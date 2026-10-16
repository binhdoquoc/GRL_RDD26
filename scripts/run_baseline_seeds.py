"""Train baselines from each saved adaptive seed's actual resolved settings."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# Direct execution has scripts/ on sys.path; resolve imports from the checkout.
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_baseline import prepare_config


def saved_runs(root, seeds):
    manifest = root / "seed_runs.json"
    records = json.loads(manifest.read_text())["runs"] if manifest.is_file() else []
    original_root = Path(records[0]["run_dir"]) if records else None
    indexed = {r["seed"]: r for r in records}
    plan = []
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    for seed in seeds:
        if records:
            if seed not in indexed:
                raise ValueError(f"Adaptive seed {seed} is missing from seed_runs.json")
            relative = Path(indexed[seed]["run_dir"]).relative_to(original_root)
            run = root / relative
        else:
            # Works with a downloaded train folder even without its manifest.
            run = root if yaml.safe_load((root / "args.yaml").read_text()).get("seed") == seed else root / "seeds" / str(seed) / "adaptive"
        if not run.resolve().is_relative_to(root.resolve()):
            raise ValueError("Saved run path escapes the supplied adaptive results folder")
        args = yaml.safe_load((run / "args.yaml").read_text())
        if args.get("seed") != seed:
            raise ValueError(f"Saved seed differs from requested seed {seed}: {run}")
        plan.append((seed, run, indexed.get(seed)))
    return plan


def find_saved_yaml(root, original, replacement):
    if replacement:
        return str(Path(replacement).resolve())
    # The standard download has results/train and project/configs as siblings.
    for parent in [root, *list(root.parents)[:3]]:
        candidate = parent / "project/configs" / Path(original).name
        if candidate.is_file():
            return str(candidate.resolve())
    if Path(original).is_file():
        return None  # same live session, with the original accessible file
    return None  # prepare_config gives an actionable missing-file error


def prepare_seed_configs(root, project, seeds, data=None, model=None, dataset_root=None):
    plan = []
    for seed, run, adaptive_record in saved_runs(root, seeds):
        saved = yaml.safe_load((run / "configuration.yaml").read_text())["train"]
        destination = project / "baseline_seed_settings" / str(seed)
        config = prepare_config(run, project,
            find_saved_yaml(root, saved["data"], data),
            find_saved_yaml(root, saved["model"], model), destination=destination)
        settings = yaml.safe_load(config.read_text())
        if dataset_root:
            original_data = yaml.safe_load(Path(settings["train"]["data"]).read_text())
            previous_root = Path(original_data["path"])
            new_root = Path(dataset_root).resolve()
            for key in ("train", "val", "test", "target"):
                if key in original_data:
                    def remap(value):
                        path = Path(value)
                        return str(new_root / (path.relative_to(previous_root) if path.is_absolute() else path))
                    value = original_data[key]
                    original_data[key] = [remap(v) for v in value] if isinstance(value, list) else remap(value)
            original_data["path"] = str(new_root)
            remapped = config.parent / "data.yaml"
            remapped.write_text(yaml.safe_dump(original_data, sort_keys=False))
            settings["train"]["data"] = str(remapped.resolve())
            config.write_text(yaml.safe_dump(settings, sort_keys=False))
            audit = destination / "baseline_settings_copy.json"
            record = json.loads(audit.read_text())
            record["dataset_root_override"] = {"from": str(previous_root), "to": str(new_root)}
            record["copied_settings"] = settings["train"]
            audit.write_text(json.dumps(record, indent=2))
        plan.append({"seed": seed, "source_run": str(run.resolve()), "config": str(config.resolve()),
                     "adaptive_record": adaptive_record})
        print(f"BASELINE ONLY seed {seed}: copied {run}", flush=True)
        print(yaml.safe_dump(settings, sort_keys=False), flush=True)
    return plan


def execute_plan(plan, project):
    from ultralytics.utils.files import increment_path
    from grl_yolo.experiments import write_summary

    output = increment_path(project / "runs/detect/BASELINE_ONLY", mkdir=True).resolve()
    records = []
    for row in plan:
        seed = row["seed"]
        config = Path(row["config"])
        report = output / f"seed_{seed}_run.json"
        subprocess.run([sys.executable, str(project / "train.py"), "--config", str(config),
                        "--baseline", "--project", str(output), "--name", f"seed_{seed}",
                        "--report", str(report)], cwd=project, check=True)
        run = Path(json.loads(report.read_text())["run_dir"])
        settings = yaml.safe_load(config.read_text())["train"]
        from ultralytics import YOLO
        data = yaml.safe_load(Path(settings["data"]).read_text())
        split = "test" if data.get("test") else "val"
        metrics = YOLO(run / "weights/best_detector.pt").val(
            data=settings["data"], split=split, device=settings.get("device", 0),
            imgsz=settings.get("imgsz", 640), batch=settings.get("batch", 16),
            project=str(run / "evaluation"), name=split)
        values = {k: float(v) for k, v in metrics.results_dict.items()}
        Path(metrics.save_dir, "metrics.json").write_text(json.dumps(values, indent=2))
        records.append({"mode": "baseline", "seed": seed, "split": split,
                        "run_dir": str(run), "metrics": values, "source_run": row["source_run"]})
        shutil.copytree(config.parent.parent, run / "copied_settings")
        (output / "seed_runs.json").write_text(json.dumps({"runs": records}, indent=2))
        write_summary(output, records)
        adaptive = [r["adaptive_record"] for r in plan if r["adaptive_record"]]
        if len(adaptive) == len(plan) and all(r.get("split") == split for r in adaptive):
            write_summary(output, adaptive + records)
    (project / "baseline_only_seeds_report.json").write_text(json.dumps({"run_dir": str(output), "runs": records}, indent=2))
    print("All requested baselines completed:", output, flush=True)
    return output


def prepare_repository_configs(filename, project, seeds=None, data=None, model=None):
    from ultralytics.cfg import get_cfg
    from grl_yolo.config import experiment_seeds, load_training_config

    overrides, _ = load_training_config(filename)
    if overrides.get("resume"):
        raise ValueError("Baseline comparison must start from initial weights, not resume a run")
    seeds = seeds if seeds is not None else (experiment_seeds(filename) or [overrides.get("seed", 0)])
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if data:
        overrides["data"] = str(Path(data).resolve())
    if model:
        overrides["model"] = str(Path(model).resolve())
    overrides.setdefault("device", 0)  # same GPU override as the adaptive Kaggle notebook
    training = vars(get_cfg(overrides=overrides))
    for key in ("project", "name", "exist_ok", "save_dir"):
        training.pop(key, None)
    training["resume"] = False
    plan = []
    for seed in seeds:
        directory = project / "baseline_seed_settings" / str(seed)
        config = directory / "configs/baseline_from_grl.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        current = {**training, "seed": seed}
        config.write_text(yaml.safe_dump({"train": current, "grl": {"enabled": False}}, sort_keys=False))
        (directory / "baseline_settings_copy.json").write_text(json.dumps({
            "source_config": str(Path(filename).resolve()), "copied_settings": current,
            "note": "Repository settings and native defaults; GRL disabled; same initial pretrained weights."
        }, indent=2))
        plan.append({"seed": seed, "source_run": None, "config": str(config.resolve()),
                     "adaptive_record": None})
        print(f"BASELINE ONLY seed {seed}: repository configuration {filename}", flush=True)
    print(yaml.safe_dump({"train": training, "seeds": seeds}, sort_keys=False), flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-root", type=Path, help="Optional alternative: copy saved adaptive settings")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/train.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: experiment.seeds in repository YAML")
    parser.add_argument("--data")
    parser.add_argument("--model")
    parser.add_argument("--dataset-root")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.adaptive_root:
        plan = prepare_seed_configs(args.adaptive_root.resolve(), PROJECT, args.seeds or [0, 1, 2],
                                    args.data, args.model, args.dataset_root)
    else:
        if args.dataset_root:
            raise ValueError("For repository settings, edit the dataset YAML root or pass --data")
        plan = prepare_repository_configs(args.config, PROJECT, args.seeds, args.data, args.model)
    print(f"Validated {len(plan)} baseline configurations; no adaptive training will run.", flush=True)
    if not args.check_only:
        execute_plan(plan, PROJECT)


if __name__ == "__main__":
    main()
