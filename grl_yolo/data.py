"""Source YOLO labels and a separate, strictly image-only target loader."""
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.data.dataset import YOLODataset
from PIL import Image


class UnlabeledYOLODataset(YOLODataset):
    """Use the native augmentation pipeline without reading target annotations."""
    def get_labels(self):
        labels = []
        for filename in self.im_files:
            with Image.open(filename) as image:
                width, height = image.size
            labels.append({"im_file": filename, "shape": (height, width),
                           "cls": np.zeros((0, 1), dtype=np.float32),
                           "bboxes": np.zeros((0, 4), dtype=np.float32),
                           "segments": [], "keypoints": None,
                           "normalized": True, "bbox_format": "xywh"})
        return labels


def load_config(filename):
    filename = Path(filename).resolve()
    data = yaml.safe_load(filename.read_text(encoding="utf-8"))
    root = Path(data.get("path", filename.parent))
    if not root.is_absolute():
        root = filename.parent / root
    root = root.resolve()
    for key in ("train", "val", "test", "target"):
        if key in data:
            if not isinstance(data[key], str):
                raise ValueError(f"{key} must be one image directory or .txt manifest")
            value = Path(data[key])
            data[key] = str(value if value.is_absolute() else root / value)
    names = data["names"]
    data["names"] = dict(enumerate(names)) if isinstance(names, list) else names
    if set(data["names"]) != set(range(len(data["names"]))):
        raise ValueError("Class names must use consecutive integer IDs starting at 0")
    data["nc"] = len(data["names"])
    data["path"] = str(root)
    return data


class TargetImages(Dataset):
    def __init__(self, location, imgsz):
        path = Path(location).resolve()
        if path.is_dir():
            paths = sorted(path.rglob("*"))
        elif path.is_file() and path.suffix.lower() == ".txt":
            paths = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = Path(line.strip())
                    paths.append(item if item.is_absolute() else path.parent / item)
        else:
            raise FileNotFoundError(f"Target directory or manifest not found: {path}")
        self.paths = [p for p in paths if p.suffix.lower().lstrip(".") in IMG_FORMATS]
        if not self.paths:
            raise ValueError(f"No target images found in {path}")
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Target image not found: {missing[0]}")
        self.letterbox = LetterBox((imgsz, imgsz), auto=False, scaleup=True)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise ValueError(f"Cannot decode target image: {self.paths[index]}")
        image = self.letterbox(image=image)
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        return torch.from_numpy(image).float() / 255.0
