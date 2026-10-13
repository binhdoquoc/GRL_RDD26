"""Verify default preservation, label-free augmentation and native checkpoints."""
import csv
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel

from grl_yolo.config import load_training_config
from grl_yolo.data import UnlabeledYOLODataset
from grl_yolo.model import GRLDetectionModel
from grl_yolo.trainer import GRLTrainer
from train import export_checkpoints


def test_training_yaml_preserves_ultralytics_defaults():
    filename = Path(__file__).resolve().parents[1] / "configs/train.yaml"
    overrides, grl = load_training_config(filename)
    defaults = vars(get_cfg())
    effective = vars(get_cfg(overrides=overrides))
    for key, value in defaults.items():
        if key not in overrides:
            assert effective[key] == value, key
    assert grl["lambda"] == 0.1 and grl["layers"] == [4, 6, 10]
    assert GRLTrainer.build_optimizer is DetectionTrainer.build_optimizer
    assert GRLTrainer._do_train is DetectionTrainer._do_train
    assert GRLTrainer.optimizer_step is DetectionTrainer.optimizer_step


def test_grl_preserves_native_detection_loss_and_adds_only_domain_term():
    torch.set_num_threads(2)
    model = GRLDetectionModel("yolo11s.yaml", nc=1, verbose=False)
    model.args = get_cfg()
    model.train()
    source = {"img": torch.rand(2, 3, 64, 64), "batch_idx": torch.tensor([0, 1]),
              "cls": torch.zeros(2, 1), "bboxes": torch.tensor([[.5, .5, .25, .25]] * 2)}
    native_loss, native_items = DetectionModel.loss(model, source)
    total, items = model({**source, "target_img": torch.rand(2, 3, 64, 64)})
    torch.testing.assert_close(items, native_items)
    expected = native_loss + source["img"].shape[0] * model.grl_config["domain_weight"] * model.last_domain_loss
    torch.testing.assert_close(total, expected)


def make_data(root):
    for split in ("source", "target", "val"):
        images, labels = root / split / "images", root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        for index in range(4):
            cv2.imwrite(str(images / f"{index}.jpg"), np.full((64, 64, 3), 50 + index * 40, dtype=np.uint8))
            (labels / f"{index}.txt").write_text(
                "0 0.5 0.5 0.25 0.25" if split != "target" else "invalid target labels",
                encoding="utf-8")
    filename = root / "data.yaml"
    filename.write_text(yaml.safe_dump({"path": str(root), "train": "source/images",
                                       "val": "val/images", "target": "target/images",
                                       "names": ["damage"]}), encoding="utf-8")
    return filename


def test_target_uses_native_augmentation_with_empty_labels(tmp_path):
    make_data(tmp_path)
    cfg = get_cfg(overrides={"imgsz": 64})
    data = {"names": {0: "damage"}, "nc": 1}
    source = build_yolo_dataset(deepcopy(cfg), str(tmp_path / "source/images"), 2, data)
    target = UnlabeledYOLODataset(img_path=str(tmp_path / "target/images"), imgsz=64,
                                 batch_size=2, augment=True, hyp=deepcopy(cfg), data=data)
    def transform_signature(transform):
        children = getattr(transform, "transforms", [])
        return (type(transform).__name__, getattr(transform, "p", None),
                [transform_signature(child) for child in children])
    assert transform_signature(source.transforms) == transform_signature(target.transforms)
    assert target[0]["img"].shape == (3, 64, 64)
    assert target[0]["cls"].numel() == 0
    assert all(label["cls"].shape == (0, 1) for label in target.labels)


def test_native_training_resume_and_plain_export(tmp_path, monkeypatch):
    # Avoid fetching a plot font in this offline CPU verification.
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *args, **kwargs: None)
    torch.set_num_threads(2)
    data = make_data(tmp_path)
    args = {"model": "yolo11s.yaml", "data": str(data), "epochs": 2, "imgsz": 64,
            "batch": 2, "device": "cpu", "workers": 0, "plots": False,
            "project": str(tmp_path / "runs"), "name": "adapted", "amp": False}
    trainer = GRLTrainer(overrides=args)
    # Keep two planned epochs, but emulate interruption after saving epoch one.
    trainer.add_callback("on_model_save", lambda t: setattr(t, "stop", True))
    trainer.train()
    resume = trainer.wdir / "resume.pt"
    state = torch.load(resume, weights_only=False)
    assert state["epoch"] == 0 and state["optimizer"] is not None
    assert isinstance(state["ema"], GRLDetectionModel)
    assert any(key.startswith("domain_heads.") for key in state["ema"].state_dict())
    restored = GRLTrainer(overrides={**args, "resume": str(resume)})
    restored.train()
    assert restored.start_epoch == 1
    export_checkpoints(restored)
    plain = YOLO(restored.wdir / "last_detector.pt").model
    assert not hasattr(plain, "domain_heads")
    with (restored.save_dir / "domain.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(float(row["domain_loss"]) > 0 for row in rows)
    # Baseline needs no target key and uses the unmodified trainer.
    baseline_data = yaml.safe_load(data.read_text())
    del baseline_data["target"]
    baseline_yaml = tmp_path / "baseline.yaml"
    baseline_yaml.write_text(yaml.safe_dump(baseline_data), encoding="utf-8")
    baseline = DetectionTrainer(overrides={**args, "data": str(baseline_yaml), "epochs": 1, "name": "baseline"})
    baseline.train()
    assert (baseline.wdir / "best.pt").is_file()
