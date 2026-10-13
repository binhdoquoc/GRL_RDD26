"""Separate GRL-only settings from unchanged Ultralytics training settings."""
from copy import deepcopy
from pathlib import Path

import yaml


GRL_DEFAULTS = {"enabled": True, "layers": [4, 6, 10], "lambda": 0.1,
                "domain_weight": 1.0, "schedule": "constant", "hidden_channels": None,
                "norm": "group", "groups": 8}


def grl_settings(values=None):
    values = values or {}
    unknown = set(values) - set(GRL_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown GRL settings: {sorted(unknown)}")
    settings = {**deepcopy(GRL_DEFAULTS), **values}
    if settings["lambda"] < 0 or settings["domain_weight"] < 0:
        raise ValueError("GRL lambda and domain_weight must be nonnegative")
    if settings["schedule"] not in {"constant", "dann"}:
        raise ValueError("GRL schedule must be constant or dann")
    layers = settings["layers"]
    if not layers or len(set(layers)) != len(layers) or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError("GRL layers must be distinct nonnegative backbone indices")
    return settings


def load_training_config(filename):
    filename = Path(filename).resolve()
    raw = yaml.safe_load(filename.read_text(encoding="utf-8")) or {}
    if set(raw) - {"train", "grl"}:
        raise ValueError("Training YAML must contain only train and grl sections")
    train = deepcopy(raw.get("train", {}))
    # Only supplied settings are overrides; all omitted keys come from Ultralytics.
    for key in ("data", "project"):
        if key in train and train[key] is not None:
            path = Path(train[key])
            train[key] = str(path if path.is_absolute() else (filename.parent / path).resolve())
    for key in ("model", "resume", "pretrained"):
        value = train.get(key)
        if isinstance(value, str):
            path = filename.parent / value
            if path.exists():
                train[key] = str(path.resolve())
    architecture_grl = {}
    model = train.get("model")
    if isinstance(model, str) and Path(model).suffix.lower() in {".yaml", ".yml"} and Path(model).is_file():
        architecture = yaml.safe_load(Path(model).read_text(encoding="utf-8")) or {}
        architecture_grl = architecture.get("grl", {})
    return train, grl_settings({**architecture_grl, **(raw.get("grl") or {})})
