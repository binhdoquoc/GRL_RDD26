"""Native Ultralytics training with YAML-configured GRL architecture additions."""
import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import yaml_save

from grl_yolo.config import load_training_config
from grl_yolo.model import plain_detector
from grl_yolo.outputs import clear_previous_results
from grl_yolo.trainer import GRLTrainer


def make_detector(names, weights="yolo11s.pt"):
    """Build an ordinary detector for regression tests and compatible weights."""
    base = YOLO("yolo11s.yaml").model
    model = DetectionModel(deepcopy(base.yaml), nc=len(names), verbose=False)
    if weights:
        model.load(YOLO(weights).model)
    model.names = names
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".dfl." not in name)
    return model


def save_detector(model, filename, args):
    detector = plain_detector(model)
    detector.args = deepcopy(args)
    torch.save({"model": detector, "train_args": vars(args) if not isinstance(args, dict) else args,
                "version": ultralytics.__version__}, filename)


def export_checkpoints(trainer):
    for name in ("last", "best"):
        checkpoint = trainer.wdir / f"{name}.pt"
        if checkpoint.exists():
            model, _ = attempt_load_one_weight(checkpoint, device=torch.device("cpu"))
            save_detector(model, trainer.wdir / f"{name}_detector.pt", model.args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--report", help="Write the actual run directory to JSON for notebook use")
    for key in ("data", "model", "device", "name", "project", "resume"):
        parser.add_argument(f"--{key}")
    for key in ("epochs", "batch", "imgsz", "workers"):
        parser.add_argument(f"--{key}", type=int)
    parser.add_argument("--baseline", action="store_true", help="Use the unmodified DetectionTrainer")
    cli = parser.parse_args()
    if ultralytics.__version__ != "8.3.0":
        raise RuntimeError("Install requirements.txt: this adapter targets ultralytics==8.3.0")
    overrides, grl = load_training_config(cli.config)
    for key, value in vars(cli).items():
        if key not in {"config", "baseline", "report"} and value is not None:
            overrides[key] = value
    if cli.baseline:
        grl["enabled"] = False
    print("Effective GRL settings: " + json.dumps(grl, sort_keys=True), flush=True)
    clear_previous_results(overrides, baseline=cli.baseline)
    if grl["enabled"] and grl["domain_weight"] > 0:
        trainer = GRLTrainer(overrides=overrides, grl=grl)
    else:
        trainer = DetectionTrainer(overrides=overrides)
    yaml_save(trainer.save_dir / "configuration.yaml", {"train": overrides, "grl": grl})
    trainer.train()
    export_checkpoints(trainer)
    if cli.report:
        Path(cli.report).write_text(json.dumps({"run_dir": str(trainer.save_dir.resolve())}), encoding="utf-8")
    print(f"Results: {trainer.save_dir}")


if __name__ == "__main__":
    main()
