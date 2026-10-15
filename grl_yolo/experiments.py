"""Repeat native training by seed; retain every run under its mode's first run."""
import csv
import gc
import json
import statistics
from copy import deepcopy
from pathlib import Path

import yaml


def check_pair_settings(overrides, seeds, previous_report):
    if not previous_report.is_file():
        return []
    previous = json.loads(previous_report.read_text())
    adaptive = previous.get("seed_runs", [])
    if not adaptive:
        return []
    if [r["seed"] for r in adaptive] != seeds:
        raise ValueError("Baseline seed list differs from the current adaptive experiment")
    original = yaml.safe_load((Path(previous["run_dir"]) / "configuration.yaml").read_text())["train"]
    ignored = {"seed", "name", "project", "exist_ok"}
    original = {k: v for k, v in original.items() if k not in ignored}
    current = {k: v for k, v in overrides.items() if k not in ignored}
    if original != current:
        different = sorted(k for k in original.keys() | current.keys() if original.get(k) != current.get(k))
        raise ValueError(f"Baseline settings differ from adaptive training: {different}")
    return adaptive


def write_summary(folder, records):
    rows = [{"mode": r["mode"], "seed": r["seed"], "split": r["split"],
             "run_dir": r["run_dir"], **r["metrics"]} for r in records]
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "seed_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for mode in dict.fromkeys(r["mode"] for r in records):
        group = [r for r in records if r["mode"] == mode]
        for metric in group[0]["metrics"]:
            values = [r["metrics"][metric] for r in group]
            aggregates.append({"mode": mode, "split": group[0]["split"], "metric": metric,
                               "n": len(values), "mean": statistics.mean(values),
                               "std": statistics.stdev(values) if len(values) > 1 else None})
    with (folder / "seed_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mode", "split", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(aggregates)
    for row in aggregates:
        if "mAP" in row["metric"]:
            deviation = f'{row["std"]:.5f}' if row["std"] is not None else "N/A (one seed)"
            print(f'{row["mode"]} {row["metric"]}: {row["mean"]:.5f}; std={deviation}; n={row["n"]}', flush=True)


def evaluate_run(trainer, seed):
    from ultralytics import YOLO

    split = "test" if trainer.data.get("test") else "val"
    metrics = YOLO(trainer.wdir / "best_detector.pt").val(
        data=trainer.args.data, split=split, device=trainer.args.device,
        imgsz=trainer.args.imgsz, batch=trainer.args.batch,
        project=str(trainer.save_dir / "evaluation"), name=split)
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    Path(metrics.save_dir, "metrics.json").write_text(json.dumps(values, indent=2))
    return {"seed": seed, "run_dir": str(trainer.save_dir.resolve()),
            "evaluation_dir": str(Path(metrics.save_dir).resolve()), "split": split, "metrics": values}


def run_repeated(overrides, grl, seeds, baseline, run_one, evaluate=evaluate_run):
    if overrides.get("resume"):
        raise ValueError("Resume a particular seed with a configuration without experiment.seeds")
    mode = "baseline" if baseline else "adaptive"
    primary, records = None, []
    for index, seed in enumerate(seeds):
        settings = deepcopy(overrides)
        settings["seed"] = seed
        if primary is not None:
            settings["project"] = str(primary / "seeds" / str(seed))
            settings["name"] = mode
        print(f"\n{mode.upper()} seed {seed} ({index + 1}/{len(seeds)})", flush=True)
        trainer = run_one(settings, deepcopy(grl))
        if primary is None:
            primary = trainer.save_dir.resolve()
        record = {"mode": mode, **evaluate(trainer, seed)}
        records.append(record)
        (primary / "seed_runs.json").write_text(json.dumps({"seeds": seeds, "runs": records}, indent=2))
        write_summary(primary, records)
        del trainer
        gc.collect()
    return primary, records
