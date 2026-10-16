"""Create an explicitly named baseline-seeds notebook; leave existing notebooks alone."""
import json
from pathlib import Path
from textwrap import dedent

project = Path(__file__).resolve().parents[1]
reference = json.loads((project / "kaggle_yolo11s_grl_clone.ipynb").read_text())
sources = ["".join(c["source"]) for c in reference["cells"] if c["cell_type"] == "code"]
bootstrap = next(s for s in sources if "def sync_repository():" in s)
runtime = next(s for s in sources if "def run_project(arguments):" in s).replace(
    '{"train.py", "scripts/run_baseline.py"}',
    '{"train.py", "scripts/run_baseline.py", "scripts/run_baseline_seeds.py"}')
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": dedent(source).strip().splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cell["id"] = f"baseline-seeds-{len(cells):02d}"
    cells.append(cell)


add("markdown", """
    # BASELINE ONLY — repository settings and seeds
    This trains **baseline 0, baseline 1 and baseline 2 only**. No adaptive training.
    Attach only the original image dataset. Select a GPU, enable Internet and run.
    Hyperparameters and seeds come from configs/train.yaml in GitHub, using the
    same Ultralytics version/defaults as adaptive training. No saved adaptive
    results or checkpoint upload is required. It uses the original pretrained weights.
    Use the same experiment configuration as the adaptive runs for a fair comparison.
    The frozen main notebook and adaptive results are not changed or deleted.
""")
add("code", '''
    from pathlib import Path
    import json
    import os
    import subprocess
    import sys
    import zipfile

    REPO_URL = "https://github.com/binhdoquoc/GRL_RDD26.git"
    REPO_REF = ""  # latest default branch; or a specific commit
    REPO_DIR = Path("/kaggle/working/GRL_RDD26")
    PROJECT_DIR = REPO_DIR

    TRAIN_CONFIG = PROJECT_DIR / "configs/train.yaml"
    # Change experiment.seeds in the repository to [0] for one baseline.
    RUN_TESTS = False
    TORCH_INDEX_URL = "https://pypi.org/simple"
''')
add("markdown", "## 1. Fetch the GitHub project")
add("code", bootstrap)
add("markdown", "## 2. Set up Python and verify the GPU")
add("code", runtime)
add("markdown", """
    ## 3. Review repository settings before training
    Epoch count, batch, image size, optimizer, lr0 and augmentations are loaded
    from configs/train.yaml and native defaults. experiment.seeds selects [0,1,2]
    or any requested list. GRL is disabled; output names differ from adaptive runs.
    Make configuration changes in GitHub. No results path needs to be entered.
""")
add("code", '''
    baseline_command = ["scripts/run_baseline_seeds.py", "--config", TRAIN_CONFIG]
    run_project([*baseline_command, "--check-only"])
''')
add("markdown", "## 4. Train and evaluate BASELINES ONLY")
add("code", '''
    run_project(baseline_command)
    result = json.loads((PROJECT_DIR / "baseline_only_seeds_report.json").read_text())
    BASELINE_RESULTS = Path(result["run_dir"])
    print("All baseline results:", BASELINE_RESULTS)
    print((BASELINE_RESULTS / "seed_summary.csv").read_text())
''')
add("markdown", """
    ## 5. Review and download
    BASELINE_ONLY/seed_0, seed_1, seed_2 hold separate checkpoints and evaluation
    outputs. The summary contains baseline mean/sample standard deviation. A single
    seed has no estimated standard deviation. The archive contains results and
    copied settings, not image datasets or environments.
""")
add("code", '''
    import shutil
    from IPython.display import FileLink, display
    archive = shutil.make_archive("/kaggle/working/BASELINE_ONLY_Same_GRL_Seeds", "zip",
                                  root_dir=BASELINE_RESULTS.parent, base_dir=BASELINE_RESULTS.name)
    print("Download:", archive)
    display(FileLink(archive))
''')
notebook = {**reference, "cells": cells}
output = project / "Kaggle_BASELINE_ONLY_Same_GRL_Seeds.ipynb"
output.write_text(json.dumps(notebook, indent=2) + "\n")
print(output)
