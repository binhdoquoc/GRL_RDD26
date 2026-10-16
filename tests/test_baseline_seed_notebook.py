import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.run_baseline_seeds import execute_plan, prepare_repository_configs, prepare_seed_configs, saved_runs


def saved_adaptive(root, seeds=(0, 1, 2)):
    data = root.parent / "original_data.yaml"
    model = root.parent / "original_model.yaml"
    data.write_text("path: /kaggle/input/data\ntrain: source/images\nval: target_eval/images\n")
    model.write_text("nc: 4\n")
    original = Path("/kaggle/working/old-checkout/runs/detect/train")
    records = []
    for seed in seeds:
        relative = Path(".") if seed == 0 else Path(f"seeds/{seed}/adaptive")
        run = root / relative
        run.mkdir(parents=True)
        args = {"seed": seed, "optimizer": "AdamW", "lr0": .002,
                "epochs": 100, "batch": 16, "imgsz": 640, "device": 0,
                "model": str(model), "data": str(data), "pretrained": True,
                "mosaic": 1., "close_mosaic": 10, "resume": False}
        (run / "args.yaml").write_text(yaml.safe_dump(args))
        (run / "configuration.yaml").write_text(yaml.safe_dump({
            "train": {**args, "pretrained": "yolo11s.pt"}, "grl": {"enabled": True}}))
        records.append({"seed": seed, "mode": "adaptive", "run_dir": str(original / relative),
                        "split": "val", "metrics": {"metrics/mAP50(B)": .2}})
    (root / "seed_runs.json").write_text(json.dumps({"runs": records}))


def test_downloaded_seed_paths_remap_and_all_settings_are_copied(tmp_path):
    root = tmp_path / "download/train"
    root.parent.mkdir()
    saved_adaptive(root)
    plan = prepare_seed_configs(root, tmp_path / "checkout", [0, 1, 2])
    assert len(plan) == 3
    for row in plan:
        source = yaml.safe_load((Path(row["source_run"]) / "args.yaml").read_text())
        baseline = yaml.safe_load(Path(row["config"]).read_text())
        for key in ("seed", "optimizer", "lr0", "epochs", "batch", "imgsz", "mosaic", "close_mosaic"):
            assert baseline["train"][key] == source[key]
        assert baseline["train"]["pretrained"] == "yolo11s.pt"
        assert baseline["grl"]["enabled"] is False
        assert "experiment" not in baseline
    assert len(prepare_seed_configs(root, tmp_path / "checkout", [1])) == 1


def test_missing_requested_adaptive_seed_fails(tmp_path):
    root = tmp_path / "train"
    saved_adaptive(root, (0, 1))
    with pytest.raises(ValueError, match="seed 2 is missing"):
        saved_runs(root, [0, 1, 2])


def test_archived_yaml_is_preferred_and_dataset_mount_change_is_recorded(tmp_path):
    root = tmp_path / "download/results/train"
    root.parent.mkdir(parents=True)
    saved_adaptive(root, (0,))
    archive = tmp_path / "download/project/configs"
    archive.mkdir(parents=True)
    (archive / "original_model.yaml").write_text("nc: 4\n# original saved model\n")
    (archive / "original_data.yaml").write_text("path: /kaggle/input/data\ntrain: source/images\nval: target_eval/images\n")
    plan = prepare_seed_configs(root, tmp_path / "checkout", [0], dataset_root=tmp_path / "new-input")
    config = Path(plan[0]["config"])
    copied = yaml.safe_load(config.read_text())["train"]
    assert copied["model"] == str((archive / "original_model.yaml").resolve())
    data = yaml.safe_load(Path(copied["data"]).read_text())
    assert data["train"] == str(tmp_path / "new-input/source/images")
    audit = json.loads((config.parent.parent / "baseline_settings_copy.json").read_text())
    assert Path(audit["dataset_root_override"]["from"]) == Path("/kaggle/input/data")


def test_executes_only_baselines_and_keeps_adaptive_results(tmp_path, monkeypatch):
    import ultralytics
    root = tmp_path / "train"
    saved_adaptive(root, (0, 1))
    project = tmp_path / "checkout"
    plan = prepare_seed_configs(root, project, [0, 1])
    calls = []
    def launch(command, **kwargs):
        assert "--baseline" in command
        calls.append(command)
        output = Path(command[command.index("--project") + 1]) / command[command.index("--name") + 1]
        (output / "weights").mkdir(parents=True)
        (output / "weights/best_detector.pt").write_text("fake checkpoint")
        Path(command[command.index("--report") + 1]).write_text(json.dumps({"run_dir": str(output)}))
    class Detector:
        def __init__(self, path):
            pass
        def val(self, **kwargs):
            output = Path(kwargs["project"]) / kwargs["name"]
            output.mkdir(parents=True)
            return SimpleNamespace(save_dir=output, results_dict={"metrics/mAP50(B)": .3})
    monkeypatch.setattr("scripts.run_baseline_seeds.subprocess.run", launch)
    monkeypatch.setattr(ultralytics, "YOLO", Detector)
    output = execute_plan(plan, project)
    assert len(calls) == 2
    assert (output / "seed_0/copied_settings/baseline_settings_copy.json").is_file()
    assert (output / "seed_1/weights/best_detector.pt").is_file()
    assert (root / "seeds/1/adaptive/args.yaml").is_file()
    assert (output / "seed_summary.csv").is_file()


def test_new_notebook_has_no_adaptive_training_command():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "Kaggle_BASELINE_ONLY_Same_GRL_Seeds.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    assert any("scripts/run_baseline_seeds.py" in source for source in sources)
    assert not any('command = ["train.py"' in source for source in sources)
    assert any("--check-only" in source for source in sources)
    assert not any("ADAPTIVE_RUN_ROOT" in source for source in sources)


def test_repository_configuration_needs_no_adaptive_results(tmp_path):
    project = Path(__file__).resolve().parents[1]
    filename = project / "configs/train.yaml"
    plan = prepare_repository_configs(filename, tmp_path)
    assert [r["seed"] for r in plan] == [0, 1, 2]
    for row in plan:
        config = yaml.safe_load(Path(row["config"]).read_text())
        args = config["train"]
        assert args["seed"] == row["seed"]
        assert args["optimizer"] == "AdamW" and args["lr0"] == .002
        assert args["epochs"] == 100 and args["batch"] == 16 and args["imgsz"] == 640
        assert args["pretrained"] == "yolo11s.pt"
        assert config["grl"]["enabled"] is False
        assert row["source_run"] is None
    assert len(prepare_repository_configs(filename, tmp_path, seeds=[0])) == 1
