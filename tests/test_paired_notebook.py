"""Exercise both notebook training cells through the seed runner without GPU training."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import yaml


def test_one_notebook_launches_all_six_with_matching_parameters(tmp_path, monkeypatch):
    import sys
    from grl_yolo import cli
    from grl_yolo.experiments import run_repeated

    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "Kaggle_ADAPTIVE_AND_BASELINE_3Seeds.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    monkeypatch.chdir(tmp_path)
    calls = []
    def train(overrides, grl):
        mode = "adaptive" if grl["enabled"] else "baseline"
        calls.append((mode, dict(overrides)))
        if "project" in overrides:
            output = Path(overrides["project"]) / overrides["name"]
        else:
            output = tmp_path / "runs/detect" / ("train" if grl["enabled"] else "train2")
        output.mkdir(parents=True)
        (output / "configuration.yaml").write_text(yaml.safe_dump({"train": overrides, "grl": grl}))
        return SimpleNamespace(save_dir=output)
    def evaluate(trainer, seed):
        return {"seed": seed, "run_dir": str(trainer.save_dir), "split": "test",
                "metrics": {"metrics/mAP50(B)": .2}}
    monkeypatch.setattr(cli, "train_once", train)
    monkeypatch.setattr(cli, "run_repeated", lambda a, b, c, d, e: run_repeated(a, b, c, d, e, evaluate))
    def launch(arguments):
        monkeypatch.setattr(sys, "argv", list(map(str, arguments)))
        cli.main()
    namespace = {}
    exec(next(s for s in sources if "REPO_URL =" in s), namespace)
    assert namespace["RUN_BASELINE"] is True
    namespace.update(PROJECT_DIR=tmp_path, TRAIN_CONFIG=project / "configs/train.yaml",
        DATA_CONFIG=project / "configs/adaptiverdd.yaml", run_project=launch,
        subprocess=SimpleNamespace(check_output=lambda *a, **kw: "test-commit\n"))
    exec(next(s for s in sources if 'command = ["train.py"' in s), namespace)
    exec(next(s for s in sources if "if RUN_BASELINE:" in s), namespace)
    assert [(mode, args["seed"]) for mode, args in calls] == [
        ("adaptive", 0), ("adaptive", 1), ("adaptive", 2),
        ("baseline", 0), ("baseline", 1), ("baseline", 2)]
    ignored = {"project", "name"}
    for index in range(3):
        adaptive = {k: v for k, v in calls[index][1].items() if k not in ignored}
        baseline = {k: v for k, v in calls[index + 3][1].items() if k not in ignored}
        assert adaptive == baseline
        assert adaptive["hsv_v"] == .55 and adaptive["perspective"] == .0004
        assert adaptive["degrees"] == 5. and adaptive["close_mosaic"] == 20
        assert adaptive["optimizer"] == "AdamW" and adaptive["lr0"] == .002
    assert namespace["REPO_REF"] == "test-commit"
    assert len(json.loads((tmp_path / "notebook_run.json").read_text())["seed_runs"]) == 3
    assert len(json.loads((tmp_path / "notebook_baseline.json").read_text())["seed_runs"]) == 3
