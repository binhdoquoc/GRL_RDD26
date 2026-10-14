"""Start a fresh experiment without deleting its subsequent baseline results."""
import shutil
from pathlib import Path


def clear_previous_results(overrides, baseline=False, root=None):
    # Baseline follows GRL in the notebook; resume needs the existing checkpoints.
    if baseline or overrides.get("resume") or overrides.get("project"):
        return False
    root = Path(root or Path.cwd()).resolve()
    runs = root / "runs"
    target = runs / "detect"
    # Validate the exact absolute target before any recursive deletion.
    for path in (runs, target):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError(f"Refusing to clear results through a linked path: {path}")
    if target.resolve() != root / "runs" / "detect":
        raise ValueError(f"Results path escapes the intended output directory: {target}")
    for key in ("model", "pretrained", "data"):
        value = overrides.get(key)
        if isinstance(value, str) and Path(value).resolve().is_relative_to(target):
            raise ValueError(f"Training input {key} is inside previous results. Copy it outside {target} first.")
    if target.exists():
        shutil.rmtree(target)
        print(f"Cleared previous experiment results: {target}", flush=True)
        return True
    return False
