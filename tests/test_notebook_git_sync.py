"""Verify updates using a local Git remote, without network or GPU training."""
import ast
import json
import subprocess
from pathlib import Path

import pytest


def git(directory, *arguments):
    subprocess.run(["git", *arguments], cwd=directory, check=True, capture_output=True)


def commit(directory):
    git(directory, "add", ".")
    git(directory, "-c", "user.name=Notebook Test", "-c", "user.email=test@example.invalid",
        "commit", "-m", "fixture")


def test_fetches_new_remote_commit_preserves_outputs_and_rejects_dirty_edits(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    git(remote, "init", "-b", "main")
    (remote / "train.py").write_text("# original\n")
    (remote / "configs").mkdir()
    architecture = remote / "configs/model.yaml"
    architecture.write_text("schedule: constant\n")
    commit(remote)
    checkout = tmp_path / "checkout"
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "kaggle_yolo11s_grl_clone.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "def sync_repository():" in "".join(c["source"]))
    definition = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    namespace = {"REPO_URL": str(remote), "REPO_REF": "", "REPO_DIR": checkout,
                 "PROJECT_DIR": checkout, "subprocess": subprocess}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "bootstrap", "exec"), namespace)
    first = namespace["sync_repository"]()
    result = checkout / "runs/detect/train/results.csv"
    result.parent.mkdir(parents=True)
    result.write_text("previous results")
    experiment = checkout / "configs/kaggle_train.yaml"
    experiment.write_text("batch: 16\n")
    architecture.write_text("schedule: dann\n")
    commit(remote)
    second = namespace["sync_repository"]()
    assert second != first
    assert (checkout / "configs/model.yaml").read_text() == "schedule: dann\n"
    assert result.read_text() == "previous results"
    assert experiment.read_text() == "batch: 16\n"
    (checkout / "train.py").write_text("# local edit\n")
    with pytest.raises(RuntimeError, match="Tracked local edits"):
        namespace["sync_repository"]()
    assert (checkout / "train.py").read_text() == "# local edit\n"


def test_training_uses_remote_configuration_not_embedded_defaults():
    project = Path(__file__).resolve().parents[1]
    notebook = json.loads((project / "kaggle_yolo11s_grl_clone.ipynb").read_text())
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in sources:
        ast.parse(source)
    assert not any("TRAINING_YAML =" in source for source in sources)
    assert any('TRAIN_CONFIG = PROJECT_DIR / "configs/train.yaml"' in source for source in sources)
    train = next(source for source in sources if 'command = ["train.py"' in source)
    assert '"--data", DATA_CONFIG' in train
    assert (project / "kaggle_yolo11s_grl.ipynb").read_bytes() == (project / "kaggle_yolo11s_grl_clone.ipynb").read_bytes()
