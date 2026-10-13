"""Check that preflight catches runtime missing/corrupt files before training."""
from PIL import Image

from scripts.check_kaggle_access import check_dataset


def prepared(root):
    for split in ("source", "target_adaptation", "target_eval"):
        images = root / split / "images"
        labels = root / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir()
        Image.new("RGB", (16, 16)).save(images / "sample.jpg")
        (labels / "sample.txt").write_bytes(
            b"\xff\xfe" if split == "target_adaptation" else b"0 0.5 0.5 0.2 0.2")


def test_actual_reads_and_output_writes_ignore_adaptation_labels(tmp_path):
    root = tmp_path / "input"
    prepared(root)
    report = check_dataset(root, tmp_path / "output", test="")
    assert report["ok"]
    assert report["splits"]["target_adaptation"]["labels_read"] == "not read"
    assert list((tmp_path / "output").iterdir()) == []


def test_missing_source_mount_and_corrupt_target_image_fail(tmp_path):
    root = tmp_path / "input"
    prepared(root)
    (root / "source/images/sample.jpg").unlink()
    (root / "source/images").rmdir()
    (root / "target_adaptation/images/sample.jpg").write_bytes(b"corrupt image")
    report = check_dataset(root, tmp_path / "output", test="")
    assert not report["ok"]
    assert any("source" in error and "FileNotFoundError" in error for error in report["errors"])
    assert any("sample.jpg" in error and "UnidentifiedImageError" in error for error in report["errors"])
