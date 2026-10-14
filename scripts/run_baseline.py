"""Run only a baseline using a previous GRL run's resolved training settings."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def prepare_config(run, project, data=None, model=None):
    args = yaml.safe_load((run / "args.yaml").read_text())
    saved = yaml.safe_load((run / "configuration.yaml").read_text())
    if not saved.get("grl", {}).get("enabled", False):
        raise ValueError("Select a GRL run with enabled GRL in configuration.yaml")
    training = dict(args)
    # Start from the same initial weights, never the trained GRL checkpoint.
    for key in ("model", "pretrained", "data"):
        if key in saved["train"]:
            training[key] = saved["train"][key]
    changes = {}
    for key, replacement in (("data", data), ("model", model)):
        if replacement:
            replacement = str(Path(replacement).resolve())
            changes[key] = {"from": training.get(key), "to": replacement}
            training[key] = replacement
    for key in ("model", "data"):
        path = Path(training[key])
        if not path.is_file():
            raise FileNotFoundError(f"Original {key} file is unavailable: {path}. Supply --{key} with a copy of that GRL run's file.")
        training[key] = str(path.resolve())
    pretrained = training.get("pretrained")
    if isinstance(pretrained, str) and Path(pretrained).is_absolute() and not Path(pretrained).is_file():
        raise FileNotFoundError(f"Original pretrained weights unavailable: {pretrained}. Restore the same initial weights.")
    # Output naming and resume state are not hyperparameters.
    for key in ("project", "name", "resume", "exist_ok", "save_dir"):
        training.pop(key, None)
    training["resume"] = False
    output = project / "configs/baseline_from_grl.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"train": training, "grl": {"enabled": False}}, sort_keys=False))
    (project / "baseline_settings_copy.json").write_text(json.dumps({
        "source_run": str(run.resolve()), "path_overrides": changes,
        "copied_settings": training,
        "note": "GRL disabled; new run with native output naming; original pretrained initialization retained."
    }, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grl-run", required=True, type=Path)
    parser.add_argument("--data", help="Copy of original dataset YAML if its old path no longer exists")
    parser.add_argument("--model", help="Copy of original architecture YAML if its old path no longer exists")
    options = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    config = prepare_config(options.grl_run, project, options.data, options.model)
    print("Copied GRL settings; baseline only:\n" + config.read_text(), flush=True)
    subprocess.run([sys.executable, str(project / "train.py"), "--config", str(config),
                    "--baseline", "--report", str(project / "notebook_baseline.json")],
                   cwd=project, check=True)


if __name__ == "__main__":
    main()
