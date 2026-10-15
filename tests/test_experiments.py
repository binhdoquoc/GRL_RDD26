import csv
import json
from types import SimpleNamespace

import pytest
import yaml

from grl_yolo.config import experiment_seeds
from grl_yolo.experiments import check_pair_settings, run_repeated


@pytest.mark.parametrize("seeds", [[], [0, 0], [-1], [True], "0,1,2"])
def test_invalid_seed_lists_fail(tmp_path, seeds):
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"experiment": {"seeds": seeds}}))
    with pytest.raises(ValueError):
        experiment_seeds(config)


def test_repeats_preserve_parameters_results_and_sample_standard_deviation(tmp_path):
    calls = []
    def train(settings, grl):
        calls.append(dict(settings))
        directory = tmp_path / "train" if "project" not in settings else settings["project"]
        from pathlib import Path
        directory = Path(directory)
        directory.mkdir(parents=True)
        (directory / "checkpoint.pt").write_text(str(settings["seed"]))
        return SimpleNamespace(save_dir=directory)
    def evaluate(trainer, seed):
        return {"seed": seed, "run_dir": str(trainer.save_dir), "split": "test",
                "metrics": {"mAP50": [.1, .3][seed]}}
    settings = {"optimizer": "AdamW", "lr0": .002, "pretrained": "yolo11s.pt"}
    primary, rows = run_repeated(settings, {"enabled": True}, [0, 1], False, train, evaluate)
    assert [call["seed"] for call in calls] == [0, 1]
    assert all(all(call[k] == v for k, v in settings.items()) for call in calls)
    assert (primary / "checkpoint.pt").read_text() == "0"
    assert (primary / "seeds/1/checkpoint.pt").read_text() == "1"
    assert "seed" not in settings
    with (primary / "seed_summary.csv").open() as stream:
        summary = next(csv.DictReader(stream))
    assert float(summary["mean"]) == pytest.approx(.2)
    assert float(summary["std"]) == pytest.approx(.1414213562)


def test_native_two_seed_adaptive_and_baseline_cli(tmp_path, monkeypatch):
    import sys
    import torch
    from test_native_trainer import make_data
    from grl_yolo.cli import main
    from pathlib import Path

    torch.set_num_threads(2)
    monkeypatch.setattr("ultralytics.data.utils.check_font", lambda *a, **kw: None)
    monkeypatch.chdir(tmp_path)
    data = make_data(tmp_path / "data")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"train": {"model": "yolo11s.yaml", "pretrained": False,
        "data": str(data), "epochs": 1, "imgsz": 64, "batch": 2, "device": "cpu", "workers": 0,
        "amp": False, "plots": False, "optimizer": "AdamW", "lr0": .002},
        "experiment": {"seeds": [0, 1]}}))
    reports = []
    for baseline in (False, True):
        report = tmp_path / ("notebook_baseline.json" if baseline else "notebook_run.json")
        args = ["train.py", "--config", str(config), "--report", str(report)]
        if baseline:
            args += ["--baseline"]
        monkeypatch.setattr(sys, "argv", args)
        main()
        content = json.loads(report.read_text())
        assert len(content["seed_runs"]) == 2
        for row in content["seed_runs"]:
            run = Path(row["run_dir"])
            assert (run / "weights/best_detector.pt").is_file()
            saved = yaml.safe_load((run / "args.yaml").read_text())
            assert saved["seed"] == row["seed"]
            assert saved["optimizer"] == "AdamW" and saved["lr0"] == .002
        reports.append(content)
    # Adaptive checkpoints survive both later adaptive seeds and all baseline seeds.
    assert all((Path(r["run_dir"]) / "weights/best_detector.pt").is_file()
               for r in reports[0]["seed_runs"])
    with (Path(reports[1]["run_dir"]) / "seed_results.csv").open() as stream:
        combined = list(csv.DictReader(stream))
    assert len(combined) == 4
    assert {r["mode"] for r in combined} == {"adaptive", "baseline"}


def test_mismatched_baseline_parameters_are_rejected(tmp_path):
    (tmp_path / "configuration.yaml").write_text(yaml.safe_dump({"train": {"lr0": .002, "seed": 0}}))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"run_dir": str(tmp_path), "seed_runs": [{"seed": 0}]}))
    assert check_pair_settings({"lr0": .002}, [0], report)
    with pytest.raises(ValueError, match="settings differ"):
        check_pair_settings({"lr0": .01}, [0], report)
    with pytest.raises(ValueError, match="seed list differs"):
        check_pair_settings({"lr0": .002}, [1], report)
