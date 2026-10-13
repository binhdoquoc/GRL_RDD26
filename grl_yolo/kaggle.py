"""Standard-library helpers for discovering and staging a Kaggle dataset."""
import os
from pathlib import Path


def named_directory(parent, name):
    """Resolve dataset directory spelling while preserving its real Linux path."""
    parent = Path(parent)
    direct = parent / name
    # Explicit accessible paths need no parent-directory enumeration.
    if direct.is_dir() or not parent.is_dir():
        return direct
    matches = [p for p in parent.iterdir() if p.is_dir() and p.name.casefold() == name.casefold()]
    exact = [p for p in matches if p.name == name]
    if exact:
        return exact[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple case variants for {name} under {parent}: {matches}")
    return matches[0] if matches else direct


def split_images(root, split):
    return named_directory(named_directory(root, split), "images")


def adaptation_split(root):
    """Prefer the prepared dataset's explicit adaptation split; retain legacy support."""
    root = Path(root)
    for name in ("target_adaptation", "target"):
        folder = named_directory(root, name)
        if named_directory(folder, "images").is_dir():
            return folder.name
    raise FileNotFoundError(f"Expected target_adaptation/images or target/images under {root}")


def missing_training_paths(root):
    """Describe incomplete mounts instead of treating them as undiscovered datasets."""
    root = Path(root)
    missing = []
    source = split_images(root, "source")
    if not source.is_dir():
        missing.append(str(source))
    try:
        adaptation_split(root)
    except FileNotFoundError:
        name = "target_adaptation" if named_directory(root, "target_adaptation").is_dir() else "target"
        missing.append(str(split_images(root, name)))
    return missing


def discover_dataset(input_root="/kaggle/input", override=""):
    def valid(root):
        if not split_images(root, "source").is_dir():
            return False
        try:
            adaptation_split(root)
        except FileNotFoundError:
            return False
        return True

    if override:
        root = Path(override).resolve()
        if not valid(root):
            raise FileNotFoundError("Training dataset is incomplete. Missing image directories: "
                                    + ", ".join(missing_training_paths(root)))
        return root
    input_root = Path(input_root)
    candidates, incomplete, visited = set(), {}, set()
    # Follow Kaggle input links, but never revisit resolved directories or walk images.
    for current, directories, _ in os.walk(input_root, followlinks=True):
        root = Path(current).resolve()
        if root in visited:
            directories[:] = []
            continue
        visited.add(root)
        lower_names = {name.casefold() for name in directories}
        if lower_names & {"source", "target_adaptation", "target"}:
            if valid(root):
                candidates.add(root)
            else:
                incomplete[root] = missing_training_paths(root)
        directories[:] = [name for name in directories if name.casefold() not in {"images", "labels"}]
    candidates = sorted(candidates)
    if not candidates and incomplete:
        details = "\n".join(f"{root}: missing {', '.join(paths)}" for root, paths in incomplete.items())
        raise FileNotFoundError("Attached training dataset is incomplete:\n" + details +
                                "\nAttach source images with YOLO labels and target-adaptation images. "
                                "Target evaluation/test images cannot substitute for missing training data.")
    if len(candidates) != 1:
        raise ValueError(f"Found {len(candidates)} matching datasets: {candidates}. "
                         "Expected source/images plus target_adaptation/images (or target/images). "
                         "Set DATASET_ROOT explicitly if multiple datasets are attached.")
    return candidates[0]


def stage_dataset(root, destination):
    """Link input images/labels; keep Ultralytics label caches in writable storage."""
    root, destination = Path(root).resolve(), Path(destination).resolve()
    target_split = adaptation_split(root)
    for split in ("source", "target", "source_eval", "target_eval", "source_test", "target_test"):
        # Keep generated training YAML stable: staged target/ maps to target_adaptation/.
        original = named_directory(root, target_split if split == "target" else split)
        if not named_directory(original, "images").is_dir():
            continue
        staged = destination / split
        staged.mkdir(parents=True, exist_ok=True)
        for kind in ("images", "labels"):
            source, link = named_directory(original, kind), staged / kind
            if not source.is_dir():
                continue
            if link.is_symlink():
                if link.resolve() != source.resolve():
                    raise FileExistsError(f"Existing link points to another dataset: {link}")
            elif link.exists():
                raise FileExistsError(f"Expected a dataset symlink, found an existing path: {link}")
            else:
                link.symlink_to(source, target_is_directory=True)
    return destination


def describe_split(root, split, require_labels=False):
    root = Path(root)
    formats = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}
    image_root = split_images(root, split)
    label_root = named_directory(named_directory(root, split), "labels")
    images = sorted(p for p in image_root.rglob("*")
                    if p.is_file() and p.suffix.lower().lstrip(".") in formats)
    if not images:
        raise ValueError(f"No images in {root / split / 'images'}")
    if not require_labels:
        return {"split": split, "images": len(images), "labels": "not read"}
    labels = [label_root / p.relative_to(image_root).with_suffix(".txt") for p in images]
    existing = sum(p.is_file() for p in labels)
    nonempty = sum(p.is_file() and bool(p.read_text(encoding="utf-8").strip()) for p in labels)
    if not nonempty:
        raise ValueError(f"No nonempty YOLO labels found for {split}")
    return {"split": split, "images": len(images), "labels": existing, "nonempty_labels": nonempty}
