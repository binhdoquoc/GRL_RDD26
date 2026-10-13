"""Check the installable local-extension -> model YAML -> native trainer workflow."""
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.nn.tasks import yaml_model_load

from grl_yolo.api import GRLYOLO
from grl_yolo.config import load_training_config
from grl_yolo.model import GRLDetectionModel, plain_detector
from grl_yolo.trainer import GRLTrainer
from grl_yolo.adaptation import capture_features


PROJECT = Path(__file__).resolve().parents[1]


def test_default_config_uses_model_yaml_and_native_results_settings():
    train, grl = load_training_config(PROJECT / "configs/train.yaml")
    assert Path(train["model"]).name == "yolo11s-grl.yaml"
    assert train["pretrained"] == "yolo11s.pt"
    assert "project" not in train and "name" not in train
    expected = get_save_dir(get_cfg(overrides={"model": train["model"], "data": train["data"]}))
    actual = get_save_dir(get_cfg(overrides=train))
    assert actual == expected
    assert grl["layers"] == [4, 6, 10]


def test_architecture_yaml_and_experiment_override_precedence(tmp_path):
    architecture = yaml.safe_load((PROJECT / "configs/yolo11s-grl.yaml").read_text())
    architecture["grl"]["lambda"] = 0.3
    model = tmp_path / "yolo11s-custom.yaml"
    model.write_text(yaml.safe_dump(architecture))
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"train": {"model": model.name, "data": "data.yaml"}}))
    _, settings = load_training_config(config)
    assert settings["lambda"] == 0.3
    config.write_text(yaml.safe_dump({"train": {"model": model.name}, "grl": {"lambda": 0.2}}))
    _, settings = load_training_config(config)
    assert settings["lambda"] == 0.2


def test_local_api_loads_grl_architecture_without_modifying_ultralytics(tmp_path):
    torch.set_num_threads(2)
    architecture = yaml_model_load(PROJECT / "configs/yolo11s-grl.yaml")
    architecture["grl"]["lambda"] = 0.25
    custom = tmp_path / "yolo11s-custom.yaml"
    custom.write_text(yaml.safe_dump(architecture))
    wrapper = GRLYOLO(custom, verbose=False)
    assert isinstance(wrapper.model, GRLDetectionModel)
    assert wrapper.model.grl_config["lambda"] == 0.25
    assert wrapper.task_map["detect"]["trainer"] is GRLTrainer
    assert len(wrapper.model.domain_heads.heads) == 3
    wrapper.model.eval()
    with torch.no_grad():
        inputs = torch.rand(2, 3, 64, 64)
        _, from_detector = capture_features(wrapper.model, inputs)
        from_backbone = wrapper.model.backbone_features(inputs)
    for expected, actual in zip(from_detector, from_backbone):
        torch.testing.assert_close(actual, expected)
    # A standard YOLO remains standard, proving task-map changes are local.
    base = YOLO("yolo11s.yaml").model
    assert not isinstance(base, GRLDetectionModel)
    assert wrapper.model.model[0].conv.out_channels == base.model[0].conv.out_channels
    wrapper.model.args = get_cfg()
    exported = plain_detector(wrapper.model)
    assert not hasattr(exported, "domain_heads") and "grl" not in exported.yaml


def test_native_pretrained_loading_matches_local_detector_weights(tmp_path):
    torch.set_num_threads(2)
    source = YOLO("yolo11s.yaml").model
    weights = tmp_path / "pretrained.pt"
    torch.save({"model": deepcopy(source)}, weights)
    wrapper = GRLYOLO(PROJECT / "configs/yolo11s-grl.yaml", verbose=False).load(weights)
    torch.testing.assert_close(wrapper.model.model[0].conv.weight, source.model[0].conv.weight)


def test_cli_yaml_workflow_trains_with_local_weights_and_default_output(tmp_path, monkeypatch):
    import json
    import sys
    from grl_yolo.cli import main
    from test_native_trainer import make_data
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(2)
    data = make_data(tmp_path / "dataset")
    source = YOLO("yolo11s.yaml").model
    weights = tmp_path / "local_pretrained.pt"
    torch.save({"model": source}, weights)
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump({"train": {
        "model": str(PROJECT / "configs/yolo11s-grl.yaml"), "pretrained": str(weights),
        "data": str(data), "epochs": 1, "imgsz": 64, "batch": 2, "workers": 0,
        "device": "cpu", "amp": False, "plots": False}}))
    report = tmp_path / "run.json"
    monkeypatch.setattr(sys, "argv", ["yolo11s-grl", "--config", str(config), "--report", str(report)])
    main()
    run = Path(json.loads(report.read_text())["run_dir"])
    assert run.parent.parts[-2:] == ("runs", "detect")
    assert run.name == "train" or (run.name.startswith("train") and run.name[5:].isdigit())
    assert yaml.safe_load((run / "args.yaml").read_text())["project"] is None
    assert (run / "results.csv").is_file() and (run / "args.yaml").is_file()
    assert (run / "domain.csv").is_file()
    for name in ("best.pt", "last.pt", "best_detector.pt", "last_detector.pt"):
        assert (run / "weights" / name).is_file()
