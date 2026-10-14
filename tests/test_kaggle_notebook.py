"""Verify notebook packaging and Kaggle dataset discovery without a Kaggle session."""
import ast
import json
from pathlib import Path

import pytest

from grl_yolo.kaggle import adaptation_split, describe_split, discover_dataset, stage_dataset


PROJECT = Path(__file__).resolve().parents[1]


def test_notebook_uses_updated_kaggle_bootstrap():
    notebook = json.loads((PROJECT / "kaggle_yolo11s_grl.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "training_environment.prepare_environment(PROJECT_DIR, torch_index=TORCH_INDEX_URL)" in code
    assert 'TORCH_INDEX_URL = "https://pypi.org/simple"' in code
    assert '"--system-site-packages"' not in code
    assert 'sys.executable, "-m", "venv"' not in code
    assert "sync_repository()" in code


def test_explicit_notebook_paths_bypass_discovery_and_read_samples(tmp_path, monkeypatch, capsys):
    import sys
    from grl_yolo import kaggle
    root = tmp_path / "prepared"
    for split, filename in (("source", "China_Drone_000000"),
                            ("target_adaptation", "China_MotorBike_000001")):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        (root / split / "images" / f"{filename}.jpg").write_bytes(b"image sample for direct read")
        (root / split / "labels" / f"{filename}.txt").write_text("0 0.5 0.5 0.2 0.2")
    def unexpected_walk(*args, **kwargs):
        raise AssertionError("Explicit paths must bypass automatic root discovery")
    monkeypatch.setattr(kaggle.os, "walk", unexpected_walk)
    notebook = json.loads((PROJECT / "kaggle_yolo11s_grl.ipynb").read_text(encoding="utf-8"))
    preflight = next("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code" and "Image read OK:" in "".join(c["source"]))
    namespace = {"Path": Path, "sys": sys, "PROJECT_DIR": PROJECT, "DATASET_ROOT": str(root),
                 "PROBE_IMAGES": ["source/images/China_Drone_000000.jpg",
                                  "target_adaptation/images/China_MotorBike_000001.jpg"]}
    exec(compile(preflight, "notebook-preflight", "exec"), namespace)
    assert namespace["root"] == root
    assert capsys.readouterr().out.count("Image read OK:") == 2


def test_notebook_writes_exact_paths_without_directory_access_gate(tmp_path, monkeypatch):
    notebook = json.loads((PROJECT / "kaggle_yolo11s_grl.ipynb").read_text(encoding="utf-8"))
    codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    preflight = next(c for c in codes if "Image read OK:" in c)
    assert "folder.is_dir()" not in preflight
    assert "raise FileNotFoundError" not in preflight
    dataset_cell = next(c for c in codes if "SOURCE_IMAGES = root" in c)
    assert "stage_dataset(" not in dataset_cell and "discover_dataset(" not in dataset_cell
    root = Path("/kaggle/input/datasets/doquocbinhbinh/adaptiverdd-prepared")
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    # Reproduce the user-visible false directory metadata result without denying real I/O.
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    namespace = {"root": root, "PROJECT_DIR": project, "NAMES": ["D00", "D10", "D20", "D40"],
                 "TARGET_SPLIT": "target_adaptation", "VALIDATION_SPLIT": "target_eval",
                 "TEST_SPLIT": "target_test", "json": json}
    exec(compile(dataset_cell, "notebook-dataset-config", "exec"), namespace)
    data = json.loads((project / "configs/kaggle_data.yaml").read_text())
    assert data["train"] == str(root / "source/images")
    assert data["target"] == str(root / "target_adaptation/images")
    assert data["val"] == str(root / "target_eval/images")
    assert data["test"] == str(root / "target_test/images")


def test_notebook_fetches_github_and_code_parses(tmp_path):
    notebook = json.loads((PROJECT / "kaggle_yolo11s_grl.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    for cell in notebook["cells"]:
        assert isinstance(cell["metadata"], dict)
        assert isinstance(cell["source"], list) and all(isinstance(s, str) for s in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for source in cells:
        tree = ast.parse(source)
        # Also check standalone subprocess scripts inside cells.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"plotting", "evaluation", "prediction"}
                for t in node.targets
            ):
                ast.parse(ast.literal_eval(node.value))
    assert not any(c.startswith("PROJECT_FILES =") for c in cells)
    bootstrap = next(c for c in cells if "def sync_repository():" in c)
    assert '"fetch", "origin", REPO_REF or "HEAD"' in bootstrap
    launcher = next(c for c in cells if "def run_project(arguments):" in c)
    assert "sync_repository()" in launcher


def test_dataset_discovery_nested_layout_and_ambiguity(tmp_path):
    root = tmp_path / "datasets/owner/adaptiverdd-prepared"
    (root / "source/images").mkdir(parents=True)
    (root / "target/images").mkdir(parents=True)
    assert discover_dataset(tmp_path) == root.resolve()
    other = tmp_path / "another"
    (other / "source/images").mkdir(parents=True)
    (other / "target/images").mkdir(parents=True)
    with pytest.raises(ValueError, match="Set DATASET_ROOT"):
        discover_dataset(tmp_path)
    assert discover_dataset(tmp_path, override=str(root)) == root.resolve()


def test_prepared_dataset_target_adaptation_layout(tmp_path):
    root = tmp_path / "datasets/owner/adaptiverdd-prepared"
    for split in ("source", "source_eval", "source_test", "target_adaptation", "target_eval", "target_test"):
        (root / split / "images").mkdir(parents=True)
    assert discover_dataset(tmp_path) == root.resolve()
    assert discover_dataset(tmp_path, override=str(root)) == root.resolve()
    assert adaptation_split(root) == "target_adaptation"


def test_dataset_directory_capitalization(tmp_path):
    root = tmp_path / "prepared"
    for split in ("Source", "Target_Adaptation", "Target_Eval"):
        (root / split / "Images").mkdir(parents=True)
        (root / split / "Labels").mkdir()
        (root / split / "Images/a.jpg").write_bytes(b"image placeholder")
        (root / split / "Labels/a.txt").write_text("0 0.5 0.5 0.2 0.2")
    assert discover_dataset(tmp_path) == root.resolve()
    assert discover_dataset(tmp_path, override=str(root)) == root.resolve()
    assert adaptation_split(root).casefold() == "target_adaptation"
    assert describe_split(root, "source", require_labels=True)["labels"] == 1
    # Prefer adaptation data even when a legacy target folder is also attached.
    (root / "target/images").mkdir(parents=True)
    assert adaptation_split(root).casefold() == "target_adaptation"


def test_target_eval_or_test_cannot_replace_adaptation_split(tmp_path):
    for split in ("source", "target_eval", "target_test"):
        (tmp_path / split / "images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="target.*images"):
        discover_dataset(tmp_path)


def test_incomplete_kaggle_mount_reports_missing_training_images(tmp_path):
    root = tmp_path / "datasets/doquocbinhbinh/adaptiverdd-prepared"
    (root / "target_adaptation/labels").mkdir(parents=True)
    (root / "target_eval/images").mkdir(parents=True)
    (root / "target_test/images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as error:
        discover_dataset(tmp_path)
    message = str(error.value)
    assert str(root / "source/images") in message
    assert str(root / "target_adaptation/images") in message
    assert "evaluation/test images cannot substitute" in message
    with pytest.raises(FileNotFoundError, match="Training dataset is incomplete"):
        discover_dataset(tmp_path, override=str(root))


def test_staging_maps_target_adaptation_to_target_alias(tmp_path, monkeypatch):
    root = tmp_path / "input"
    for split in ("source", "target_adaptation", "target_eval", "target_test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "images/a.jpg").write_bytes(b"image placeholder")
    # Windows may forbid symlinks. Copy small fixtures to exercise the path mapping.
    import shutil
    monkeypatch.setattr(Path, "symlink_to", lambda self, target, **kwargs: shutil.copytree(target, self))
    staged = stage_dataset(root, tmp_path / "working/dataset")
    assert (staged / "target/images/a.jpg").is_file()
    assert describe_split(staged, "target")["images"] == 1
    assert (staged / "target_eval/images/a.jpg").is_file()
    assert (staged / "target_test/images/a.jpg").is_file()


def test_source_labels_required_target_labels_ignored(tmp_path):
    for domain in ("source", "target"):
        (tmp_path / domain / "images").mkdir(parents=True)
        (tmp_path / domain / "labels").mkdir(parents=True)
        (tmp_path / domain / "images/a.jpg").write_bytes(b"image placeholder")
    with pytest.raises(ValueError, match="No nonempty YOLO labels"):
        describe_split(tmp_path, "source", require_labels=True)
    (tmp_path / "source/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2")
    (tmp_path / "target/labels/a.txt").write_bytes(b"\xff\xfe invalid UTF-8")
    assert describe_split(tmp_path, "source", require_labels=True)["labels"] == 1
    assert describe_split(tmp_path, "target")["labels"] == "not read"
