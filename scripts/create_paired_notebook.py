"""Create one notebook with adaptive and baseline stages enabled; keep frozen files unchanged."""
import json
from pathlib import Path

project = Path(__file__).resolve().parents[1]
notebook = json.loads((project / "kaggle_yolo11s_grl_clone.ipynb").read_text())
notebook["cells"][0]["source"] = [
    "# ADAPTIVE AND BASELINE — three matched seeds\n",
    "Run all cells to train Adaptive 0, 1, 2, then Baseline 0, 1, 2.\n",
    "Both stages read configs/train.yaml from GitHub; experiment.seeds selects the seeds.\n",
    "Use [0] for one adaptive/baseline pair. No extra flag needs enabling.\n",
    "Attach the image dataset, select a GPU and enable Internet.\n",
    "The baseline uses the same Git commit as adaptive training.\n",
    "A fresh adaptive experiment clears old default results; all six new runs are preserved.\n",
    "The final download includes every seed, per-seed metrics and mean/sample standard deviation.\n",
]
for cell in notebook["cells"]:
    source = "".join(cell["source"])
    if cell["cell_type"] == "code" and "REPO_URL =" in source:
        source = source.replace("RUN_BASELINE = False", "RUN_BASELINE = True")
        source = source.replace("RUN_TESTS = True", "RUN_TESTS = False")
        source += "\nREQUESTED_REPO_REF = REPO_REF  # preserve the selected branch/commit for a new experiment\n"
    if cell["cell_type"] == "code" and 'command = ["train.py"' in source:
        source = "REPO_REF = REQUESTED_REPO_REF\n" + source
        source = source.replace("run_project(command)",
            'run_project(command)\nEXPERIMENT_REVISION = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()')
    if cell["cell_type"] == "code" and "if RUN_BASELINE:" in source:
        source = "REPO_REF = EXPERIMENT_REVISION  # pin baseline to the adaptive code and settings\n" + source
    if cell["cell_type"] == "markdown" and "## 6." in source:
        source = "## 6. Run all matching baseline seeds\nEnabled by default. Uses the same repository revision, seed list, optimizer, learning rate and augmentations as adaptive training.\n"
    cell["source"] = source.splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
output = project / "Kaggle_ADAPTIVE_AND_BASELINE_3Seeds.ipynb"
output.write_text(json.dumps(notebook, indent=2) + "\n")
print(output)
