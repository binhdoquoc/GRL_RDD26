"""Inherit the native loop; customize only GRL architecture and target batches."""
import csv
import math
import shutil
from copy import deepcopy
from pathlib import Path

from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import DEFAULT_CFG, yaml_save
from ultralytics.utils.torch_utils import de_parallel

from .config import grl_settings
from .data import UnlabeledYOLODataset
from .model import GRLDetectionModel


class GRLTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None, grl=None):
        self.grl_overrides = grl
        self.grl = grl_settings(grl)
        self.target_loader = self.target_iterator = None
        self.domain_rows = []
        self.domain_step = 0
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        device = self.args.device
        if (isinstance(device, (list, tuple)) and len(device) > 1) or (isinstance(device, str) and "," in device):
            raise ValueError("GRL currently supports one GPU; use device: 0")
        if self.args.batch < 1:
            raise ValueError("Specify a positive batch size; native AutoBatch excludes target graphs")
        self.add_callback("on_train_epoch_start", self._start_domain_epoch)
        self.add_callback("on_train_batch_start", self._set_coefficient)
        self.add_callback("on_train_batch_end", self._record_domain_batch)
        self.add_callback("on_train_epoch_end", self._write_domain_epoch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        architecture = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)
        inherited = architecture.get("grl", {})
        if self.resume and isinstance(weights, GRLDetectionModel):
            inherited = weights.grl_config
        self.grl = grl_settings({**inherited, **(self.grl_overrides or {})})
        if isinstance(weights, GRLDetectionModel) and self.resume and weights.grl_config != self.grl:
            raise ValueError("GRL YAML differs from the resume checkpoint")
        yaml_save(self.save_dir / "grl.yaml", self.grl)
        model = GRLDetectionModel(architecture, nc=self.data["nc"], grl=self.grl, verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if mode == "train":
            target = self.data.get("target")
            if not target:
                raise ValueError("Dataset YAML needs target: for unlabeled training images")
            root = Path(self.data["path"])
            def absolute(p):
                return str(Path(p) if Path(p).is_absolute() else root / p)
            target = absolute(target) if isinstance(target, str) else [absolute(p) for p in target]
            dataset = UnlabeledYOLODataset(
                img_path=target, imgsz=self.args.imgsz, batch_size=batch_size, augment=True,
                hyp=deepcopy(self.args), rect=self.args.rect, cache=self.args.cache or None,
                single_cls=self.args.single_cls, stride=self.stride, pad=0.0, prefix="target: ",
                task="detect", classes=self.args.classes, data=self.data, fraction=self.args.fraction,
            )
            self.target_loader = build_dataloader(dataset, batch_size, self.args.workers,
                                                  shuffle=not dataset.rect, rank=-1)
            self.target_iterator = iter(self.target_loader)
        return loader

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        try:
            target = next(self.target_iterator)
        except StopIteration:
            self.target_iterator = iter(self.target_loader)
            target = next(self.target_iterator)
        batch["target_img"] = super().preprocess_batch(target)["img"]
        return batch

    def _close_dataloader_mosaic(self):
        super()._close_dataloader_mosaic()
        if self.target_loader:
            self.target_loader.dataset.close_mosaic(hyp=deepcopy(self.args))
            self.target_loader.reset()
            self.target_iterator = iter(self.target_loader)

    def _start_domain_epoch(self, trainer):
        self.domain_rows = []
        self.domain_step = 0

    def _set_coefficient(self, trainer):
        value = self.grl["lambda"]
        if self.grl["schedule"] == "dann":
            batches = len(self.train_loader)
            progress = (self.epoch * batches + self.domain_step) / max(self.epochs * batches - 1, 1)
            value *= 2 / (1 + math.exp(-10 * progress)) - 1
        de_parallel(self.model).grl_coefficient = value

    def _record_domain_batch(self, trainer):
        model = de_parallel(self.model)
        self.domain_rows.append((model.last_domain_loss, model.last_total_mean, model.grl_coefficient))
        self.domain_step += 1

    def _write_domain_epoch(self, trainer):
        if not self.domain_rows:
            return
        filename = self.save_dir / "domain.csv"
        empty = not filename.exists()
        means = [sum(row[i] for row in self.domain_rows) / len(self.domain_rows) for i in range(3)]
        with filename.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if empty:
                writer.writerow(["epoch", "domain_loss", "total_loss_per_source_image", "grl_lambda"])
            writer.writerow([self.epoch + 1, *means])

    def save_model(self):
        super().save_model()
        # Preserve the native optimizer before final_eval strips last/best checkpoints.
        shutil.copy2(self.last, self.wdir / "resume.pt")
