"""Generate a separate baseline notebook without modifying the frozen clone notebook."""
import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent

project = Path(__file__).resolve().parents[1]
original = json.loads((project / "kaggle_yolo11s_grl_clone.ipynb").read_text())
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": dedent(source).strip().splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", """
    # Baseline only: copy the completed GRL run's settings
    Enable GPU and Internet. Attach the original dataset and your saved GRL run outputs.
    The selected run folder must contain `args.yaml` and `configuration.yaml`.
    Source architecture/dataset YAML and original pretrained weights must remain accessible;
    supply copies below if the old paths are unavailable. Do not use trained GRL weights.
    This notebook runs no GRL training. It copies the resolved parameters, disables GRL,
    and uses native Ultralytics output naming. Changed file paths are recorded for review.
""")
# Reuse only setup cells. Preserve original dataset YAML instead of overwriting it.
for cell in original["cells"][1:]:
    source = "".join(cell["source"])
    if cell["cell_type"] == "markdown" and ("Required file access check" in source or "## 3." in source):
        break
    cells.append(deepcopy(cell))
    if cell["cell_type"] == "code" and "REPO_URL =" in source:
        add("code", '''
            GRL_RUN_DIR = "/kaggle/input/YOUR-SAVED-GRL-RUN/train"
            # In the same live session, use the actual completed runs/detect/train folder.
            ORIGINAL_DATA_YAML = ""  # optional copy of original kaggle_data.yaml
            ORIGINAL_MODEL_YAML = ""  # optional copy of original architecture YAML
            # Leave empty when original paths still exist. Preserve original YAML contents.
        ''')
add("markdown", """
    ## Run baseline only
    Settings come from the saved run, not the notebook's default training template.
    The original dataset YAML is retained; no default dataset YAML is generated.
""")
add("code", '''
    command = ["scripts/run_baseline.py", "--grl-run", GRL_RUN_DIR]
    if ORIGINAL_DATA_YAML:
        command += ["--data", ORIGINAL_DATA_YAML]
    if ORIGINAL_MODEL_YAML:
        command += ["--model", ORIGINAL_MODEL_YAML]
    run_project(command)
    BASELINE_DIR = Path(json.loads((PROJECT_DIR / "notebook_baseline.json").read_text())["run_dir"])
    print("Baseline results:", BASELINE_DIR)
    print("Parameter copy record:", PROJECT_DIR / "baseline_settings_copy.json")
''')
add("markdown", """
    ## Evaluate the best baseline checkpoint on the original test split
""")
evaluation = '''import sys, yaml
from ultralytics import YOLO
weights, config = sys.argv[1:]
settings = yaml.safe_load(open(config))["train"]
metrics = YOLO(weights).val(data=settings["data"], split="test", device=settings["device"],
                          imgsz=settings["imgsz"], batch=settings["batch"])
print(metrics.results_dict)
print("Evaluation outputs:", metrics.save_dir)
'''
add("code", "evaluation = " + repr(evaluation) + "\n" + dedent('''
    run_project(["-c", evaluation, BASELINE_DIR / "weights/best_detector.pt",
                 PROJECT_DIR / "configs/baseline_from_grl.yaml"])
'''))
for i, cell in enumerate(cells):
    cell["id"] = f"baseline-{i:02d}"
notebook = {**original, "cells": cells}
output = project / "kaggle_baseline_only.ipynb"
output.write_text(json.dumps(notebook, indent=2) + "\n")
print(output)
