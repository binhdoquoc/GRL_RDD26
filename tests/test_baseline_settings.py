import yaml

from scripts.run_baseline import prepare_config


def test_baseline_copies_resolved_parameters_and_initial_weights(tmp_path):
    run = tmp_path / "grl"
    run.mkdir()
    model = tmp_path / "model.yaml"
    data = tmp_path / "data.yaml"
    model.write_text("nc: 4")
    data.write_text("names: [a,b,c,d]")
    args = {"epochs": 100, "batch": 4, "imgsz": 640, "seed": 7,
            "optimizer": "auto", "lr0": 0.01, "mosaic": 1.0,
            "model": str(model), "data": str(data), "pretrained": True,
            "resume": "grl/weights/last.pt", "project": "old", "name": "train"}
    (run / "args.yaml").write_text(yaml.safe_dump(args))
    (run / "configuration.yaml").write_text(yaml.safe_dump({
        "train": {"model": str(model), "data": str(data), "pretrained": "yolo11s.pt"},
        "grl": {"enabled": True}}))
    config = prepare_config(run, tmp_path)
    baseline = yaml.safe_load(config.read_text())
    for key in ("epochs", "batch", "imgsz", "seed", "optimizer", "lr0", "mosaic"):
        assert baseline["train"][key] == args[key]
    assert baseline["train"]["pretrained"] == "yolo11s.pt"
    assert baseline["grl"]["enabled"] is False
    assert baseline["train"]["resume"] is False
    assert "name" not in baseline["train"] and "project" not in baseline["train"]
