"""Check real image/label reads and output writes before starting training."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def check_dataset(root, output, target="target_adaptation", validation="target_eval", test="target_test"):
    from PIL import Image

    errors, counts = [], {}
    extensions = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm", ".heic"}

    def fail(path, error):
        errors.append(f"{path}: {type(error).__name__}: {error}")

    splits = [("source", True), (target, False), (validation, True)]
    if test:
        splits.append((test, True))
    for split, labeled in dict(splits).items():
        folder = root / split / "images"
        print(f"Checking actual image reads: {folder}", flush=True)
        images, labels = 0, 0
        # scandir distinguishes missing folders from permission errors; no is_dir gate.
        try:
            with os.scandir(folder) as entries:
                next(entries, None)
            for directory, _, filenames in os.walk(folder, followlinks=True,
                                                    onerror=lambda e: fail(folder, e)):
                for filename in filenames:
                    path = Path(directory) / filename
                    if path.suffix.lower() not in extensions:
                        continue
                    images += 1
                    try:
                        # Read every byte, then decode pixels to catch truncated images too.
                        with path.open("rb") as stream:
                            while stream.read(1024 * 1024):
                                pass
                        with Image.open(path) as image:
                            image.load()
                        if labeled:
                            label = root / split / "labels" / path.relative_to(folder).with_suffix(".txt")
                            try:
                                label.read_text(encoding="utf-8")
                                labels += 1
                            except FileNotFoundError:
                                pass  # Ultralytics permits background images without label files.
                    except Exception as error:
                        fail(path, error)
                    if images % 1000 == 0:
                        print(f"  {images} images checked", flush=True)
            if images == 0:
                raise FileNotFoundError("No supported images found in this runtime mount")
            if labeled and labels == 0:
                raise FileNotFoundError("No readable label files found for this labeled split")
        except OSError as error:
            fail(folder, error)
        counts[split] = {"images": images, "labels_read": labels if labeled else "not read"}
        print(f"  {counts[split]}", flush=True)

    try:
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="access-check-", dir=output) as probe:
            path = Path(probe) / "probe.txt"
            path.write_text("output write/read OK", encoding="utf-8")
            assert path.read_text(encoding="utf-8") == "output write/read OK"
        print(f"Output write/read/delete OK: {output}", flush=True)
    except OSError as error:
        fail(output, error)
    return {"ok": not errors, "dataset_root": str(root), "output": str(output),
            "splits": counts, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--target", default="target_adaptation")
    parser.add_argument("--validation", default="target_eval")
    parser.add_argument("--test", default="target_test", help="Empty string skips test split")
    args = parser.parse_args()
    report = check_dataset(args.root, args.output, args.target, args.validation, args.test)
    print(json.dumps(report, indent=2), flush=True)
    if not report["ok"]:
        print("FAILED: fix the listed paths or restart Kaggle and reattach the dataset before training.")
        raise SystemExit(1)
    print("PASS: all current dataset image reads and output writes succeeded.")


if __name__ == "__main__":
    main()
