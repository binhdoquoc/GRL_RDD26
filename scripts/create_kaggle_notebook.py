"""Build a self-contained Kaggle notebook with native Ultralytics defaults."""
import json
from pathlib import Path
from textwrap import dedent

PROJECT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": dedent(text).strip().splitlines(True)})


markdown("""
    # YOLO11s + GRL using the native Ultralytics trainer

    This notebook includes the updated setup for Kaggle kernels using **Python 3.13**:
    training runs in an isolated **Python 3.12** environment created by `uv`.

    Import this `.ipynb` into Kaggle, attach **adaptiverdd-prepared**, select a **GPU**,
    enable **Internet**, and run the cells from the beginning. If an older setup cell
    failed with `ensurepip`, use this notebook's environment cell instead.
    The complete project is embedded, so no GitHub push is required.

    Only the GRL architecture, domain objective and target batch supply are added.
    Optimizer, warmup, LR scheduling, AMP, EMA, validation and source/target augmentations
    follow **Ultralytics 8.3.0 defaults** unless explicitly overridden in the YAML cell.
    Default batch size is 16 source images plus a target batch; uncomment `batch: 4`
    if GPU memory requires it. Only one GPU is used. Source labels train detection;
    target annotations are never read. Validation uses a labeled split and best
    checkpoints use native validation fitness.
""")
code("""
    from pathlib import Path
    import json
    import os
    import subprocess
    import sys
    import zipfile

    DATASET_ROOT = "/kaggle/input/datasets/doquocbinhbinh/adaptiverdd-prepared"
    # Set DATASET_ROOT="" only if you want automatic discovery.
    PROBE_IMAGES = ["source/images/China_Drone_000000.jpg",
                    "target_adaptation/images/China_MotorBike_000001.jpg"]
    # Optional direct-read diagnostics; use [] for a dataset with different filenames.
    PROJECT_DIR = Path("/kaggle/working/yolo11s-grl")
    NAMES = ["D00", "D10", "D20", "D40"]
    TARGET_SPLIT = "target_adaptation"  # use "target" for a legacy dataset
    VALIDATION_SPLIT = "target_eval"  # or a separate labeled source_eval split
    TEST_SPLIT = "target_test"       # set "" to disable final test evaluation
    RUN_TESTS = True
    RUN_BASELINE = False
    TORCH_INDEX_URL = "https://pypi.org/simple"  # avoids the timed-out download.pytorch.org host
    RESUME = ""  # trusted native weights/resume.pt from an interrupted run
""")
markdown("""
    ## 1. Restore the project
    Code is refreshed on rerun; existing outputs are preserved.
""")
files = ["train.py", "requirements.txt", "pyproject.toml", "pytest.ini", "README.md", "configs/train.yaml",
         "configs/yolo11s-grl.yaml", "configs/adaptiverdd.yaml", "grl_yolo/__init__.py", "grl_yolo/adaptation.py",
         "grl_yolo/cli.py", "grl_yolo/api.py",
         "grl_yolo/config.py", "grl_yolo/model.py", "grl_yolo/trainer.py", "grl_yolo/data.py",
         "grl_yolo/kaggle.py", "grl_yolo/environment.py", "grl_yolo/process.py",
         "tests/test_adaptation.py", "tests/test_domain_loss.py", "tests/test_native_trainer.py",
         "tests/test_environment.py", "tests/test_process.py"]
embedded = {name: (PROJECT / name).read_text(encoding="utf-8") for name in files}
code("PROJECT_FILES = " + repr(embedded) + "\n\n" + dedent("""
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    for relative, content in PROJECT_FILES.items():
        path = PROJECT_DIR / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(PROJECT_DIR)
"""))
cells[-1]["metadata"]["jupyter"] = {"source_hidden": True}
markdown("""
    ## Optional direct file-read diagnostics
    These probes print the actual file-open result; they do not stop training on a
    directory metadata check. The supplied paths are passed directly to Ultralytics.
""")
code("""
    sys.path.insert(0, str(PROJECT_DIR))
    import importlib
    import grl_yolo.kaggle as kaggle_data
    importlib.reload(kaggle_data)
    root = Path(DATASET_ROOT) if DATASET_ROOT else kaggle_data.discover_dataset()
    for relative in PROBE_IMAGES:
        sample = root / relative
        try:
            with sample.open("rb") as stream:
                header = stream.read(16)
            print("Image read OK:", sample, "bytes checked:", len(header))
        except OSError as error:
            print("Image read FAILED:", sample, type(error).__name__, str(error))
    print("Training dataset:", root)
    print("File probes are diagnostic. No custom directory-access gate is applied.")
""")
markdown("""
    ## 2. Set up the isolated Python environment
    The notebook kernel may use Python 3.13. Training uses a separate **Python 3.12**
    interpreter matching the pinned project dependencies. Standalone `uv` creates it
    without invoking the kernel's broken/missing `ensurepip`; no kernel pip is required.
    Python and package downloads need Internet. PyTorch is installed from **PyPI**
    by default to avoid timeouts reaching `download.pytorch.org`. Its Linux wheel
    includes CUDA dependencies; CUDA availability is checked after installation.
    Training subprocesses use the `Agg` Matplotlib backend to avoid inheriting
    the notebook kernel's unavailable inline backend. Saved plots display below.
    The kernel does not need restarting.
""")
code("""
    sys.path.insert(0, str(PROJECT_DIR))
    import importlib
    import grl_yolo.environment as training_environment
    # Reload restored code if this notebook is imported into an existing session.
    importlib.reload(training_environment)
    print("Notebook kernel:", sys.version)
    PYTHON, ENV = training_environment.prepare_environment(PROJECT_DIR, torch_index=TORCH_INDEX_URL)

    from grl_yolo.process import run_logged
    COMMAND_NUMBER = 0

    def run_project(arguments):
        global COMMAND_NUMBER
        COMMAND_NUMBER += 1
        filename = "train.log" if str(arguments[0]) == "train.py" else f"command-{COMMAND_NUMBER:02d}.log"
        return run_logged([str(PYTHON), *map(str, arguments)], cwd=PROJECT_DIR, env=ENV,
                          log_file=PROJECT_DIR / "logs" / filename)

    runtime = subprocess.run(
        [str(PYTHON), "-c", "import sys; print('Training Python:', sys.version, flush=True); "
         "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, flush=True); "
         "print('CUDA runtime:', torch.version.cuda, 'CUDA available:', torch.cuda.is_available(), flush=True); "
         "assert torch.cuda.is_available(), 'Enable a GPU in Kaggle settings'; "
         "print(torch.cuda.get_device_name(0), flush=True)"],
        cwd=PROJECT_DIR, env=ENV, text=True, capture_output=True,
    )
    print(runtime.stdout)
    if runtime.stderr:
        print(runtime.stderr)
    if runtime.returncode:
        raise RuntimeError("Training runtime check failed. The underlying error is printed above.")
    if RUN_TESTS:
        subprocess.run([str(PYTHON), "-m", "pytest", "-q"], cwd=PROJECT_DIR,
                       env={**ENV, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}, check=True)
""")
markdown("""
    ## 3. Write dataset YAML with the exact input paths
    Source and target images are read directly from the supplied Kaggle locations.
    No directory discovery, symlink staging or additional access checks are used here.
    Native Ultralytics loading validates the source/validation data. Target training
    uses the same image-loading pipeline without reading target annotations.
    A read-only label-cache warning can be ignored; source images do not need copying.
""")
code("""
    SOURCE_IMAGES = root / "source/images"
    TARGET_IMAGES = root / TARGET_SPLIT / "images"
    VALIDATION_IMAGES = root / VALIDATION_SPLIT / "images"
    TEST_IMAGES = root / TEST_SPLIT / "images" if TEST_SPLIT else None
    has_test = TEST_IMAGES is not None
    dataset = {"path": str(root), "train": str(SOURCE_IMAGES), "target": str(TARGET_IMAGES),
               "val": str(VALIDATION_IMAGES), "names": NAMES}
    if has_test:
        dataset["test"] = str(TEST_IMAGES)
    DATA_CONFIG = PROJECT_DIR / "configs/kaggle_data.yaml"
    DATA_CONFIG.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(DATA_CONFIG.read_text())
""")
markdown("""
    ## 4. Load the architecture YAML and edit training settings
    Omitted `train` keys inherit Ultralytics defaults. Uncomment only settings you
    want to override. `device: 0` selects one GPU; it does not change optimizer or
    augmentation defaults. Edit `configs/yolo11s-grl.yaml` to change backbone,
    detection head or GRL branch settings. The training YAML selects that file and
    initializes matching weights from `yolo11s.pt`, downloading them if needed.
    `project` and `name` are omitted so Ultralytics chooses the default run folder.
""")
code('''
    TRAINING_YAML = """
    train:
      model: yolo11s-grl.yaml
      pretrained: yolo11s.pt
      data: kaggle_data.yaml
      device: 0
      # batch: 4
      # workers: 2
      # epochs: 100
      # imgsz: 640
      # optimizer: auto
      # lr0: 0.01
      # mosaic: 1.0
      # mixup: 0.0
      # close_mosaic: 10

    # Optional GRL experiment overrides (defaults are in the model YAML):
    # grl:
    #   lambda: 0.2
    #   domain_weight: 0.5
    """
    TRAIN_CONFIG = PROJECT_DIR / "configs/kaggle_train.yaml"
    from textwrap import dedent
    TRAIN_CONFIG.write_text(dedent(TRAINING_YAML), encoding="utf-8")
    print(TRAIN_CONFIG.read_text())
''')
markdown("""
    ## 5. Native training, validation and best-model selection
    Subprocess stdout/stderr are displayed live and saved to `logs/train.log`.
    On failure, the exception includes the final training error and the log path.
    Outputs include native `results.csv`, `args.yaml`, validation plots and GRL
    `domain.csv`. `weights/resume.pt` preserves domain heads, EMA and optimizer
    for an interrupted run. Resume uses the original training budget and settings;
    completed runs should initialize a new run from a detector checkpoint instead.
    Previous standalone-loop `adaptation.pt` checkpoints are incompatible with this trainer.
""")
code("""
    report = PROJECT_DIR / "notebook_run.json"
    command = ["train.py", "--config", TRAIN_CONFIG, "--report", report]
    if RESUME:
        command += ["--resume", RESUME]
    run_project(command)
    RUN_DIR = Path(json.loads(report.read_text())["run_dir"])
    print("Completed run:", RUN_DIR)
    print("Resume:", RUN_DIR / "weights/resume.pt")
""")
markdown("""
    ## 6. Optional baseline with the same native settings
    Enable `RUN_BASELINE` in the first cell for a second run without adaptation.
""")
code("""
    BASELINE_DIR = None
    if RUN_BASELINE:
        baseline_report = PROJECT_DIR / "notebook_baseline.json"
        run_project(["train.py", "--config", TRAIN_CONFIG, "--baseline", "--report", baseline_report])
        BASELINE_DIR = Path(json.loads(baseline_report.read_text())["run_dir"])
    else:
        print("Baseline skipped.")
""")
markdown("""
    ## 7. Inspect native training results and domain loss
""")
code("""
    from IPython.display import Image, display
    native_plot = RUN_DIR / "results.png"
    if native_plot.exists():
        display(Image(filename=str(native_plot)))
    domain_csv = RUN_DIR / "domain.csv"
    if domain_csv.exists():
        plotting = '''
    import csv
    import sys
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(sys.argv[1])
    with (run / "domain.csv").open() as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key in zip(axes, ("domain_loss", "grl_lambda")):
        ax.plot(epochs, [float(r[key]) for r in rows])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
    fig.tight_layout()
    fig.savefig(run / "domain.png", dpi=150)
    plt.close(fig)
    '''
        run_project(["-c", plotting, RUN_DIR])
        display(Image(filename=str(RUN_DIR / "domain.png")))
    print("Effective settings:", RUN_DIR / "args.yaml")
""")
markdown("""
    ## 8. Evaluate the held-out target test set
    Best detector weights are selected using validation fitness. Final test
    results are separate from training and validation. Exported detector checkpoints
    contain no GRL modules and load with standard Ultralytics.
""")
code("""
    evaluation = '''
    import json
    import sys
    from pathlib import Path
    from ultralytics import YOLO
    weights, data, report = sys.argv[1:]
    metrics = YOLO(weights).val(data=data, split="test", device=0)
    result = {k: float(v) for k, v in metrics.results_dict.items()}
    Path(metrics.save_dir, "metrics.json").write_text(json.dumps(result, indent=2))
    Path(report).write_text(json.dumps({"save_dir": str(Path(metrics.save_dir).resolve())}))
    print(json.dumps(result, indent=2))
    '''
    EVALUATION_DIRS = []
    if has_test:
        models = [("adapted", RUN_DIR)] + ([("baseline", BASELINE_DIR)] if BASELINE_DIR else [])
        for label, folder in models:
            evaluation_report = PROJECT_DIR / f"evaluation_{label}.json"
            run_project(["-c", evaluation, folder / "weights/best_detector.pt", DATA_CONFIG, evaluation_report])
            EVALUATION_DIRS.append(Path(json.loads(evaluation_report.read_text())["save_dir"]))
    else:
        print("Final test split disabled; final evaluation skipped.")
""")
markdown("""
    ## 9. Preview predictions and package results
    Download `/kaggle/working/yolo11s-grl-results.zip` from notebook outputs.
    The archive excludes dataset symlinks, caches and the Python environment.
""")
code("""
    preview_images = TEST_IMAGES if has_test else VALIDATION_IMAGES
    preview = next(p for p in sorted(preview_images.rglob("*"))
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    prediction = '''
    import sys
    import json
    from pathlib import Path
    from ultralytics import YOLO
    weights, source, report = sys.argv[1:]
    results = YOLO(weights).predict(source=source, device=0, save=True)
    Path(report).write_text(json.dumps({"save_dir": str(Path(results[0].save_dir).resolve())}))
    '''
    prediction_report = PROJECT_DIR / "prediction_run.json"
    run_project(["-c", prediction, RUN_DIR / "weights/best_detector.pt", preview, prediction_report])
    PREDICTION_DIR = Path(json.loads(prediction_report.read_text())["save_dir"])
    for file in sorted(PREDICTION_DIR.glob("*"))[:1]:
        display(Image(filename=str(file)))
    archive = Path("/kaggle/working/yolo11s-grl-results.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for relative in PROJECT_FILES:
            z.write(PROJECT_DIR / relative, arcname=f"project/{relative}")
        for config in (DATA_CONFIG, TRAIN_CONFIG):
            z.write(config, arcname=f"project/configs/{config.name}")
        run_folders = [("train", RUN_DIR), ("predict", PREDICTION_DIR), ("logs", PROJECT_DIR / "logs")]
        if BASELINE_DIR:
            run_folders.append(("baseline", BASELINE_DIR))
        run_folders += [(f"val-{i}", folder) for i, folder in enumerate(EVALUATION_DIRS)]
        for label, folder in run_folders:
            for file in folder.rglob("*"):
                if file.is_file() and not file.is_symlink():
                    z.write(file, arcname=f"results/{label}/{file.relative_to(folder).as_posix()}")
    print("Download:", archive)
""")
for index, cell in enumerate(cells):
    cell["id"] = f"grl-{index:02d}"
notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}}, "cells": cells}
output = PROJECT / "kaggle_yolo11s_grl.ipynb"
output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(output)

# Keep the embedded notebook available and generate a lightweight Git checkout variant.
from copy import deepcopy

clone_notebook = deepcopy(notebook)
clone_cells = clone_notebook["cells"]
clone_cells[0]["source"] = clone_cells[0]["source"][:]
intro = "".join(clone_cells[0]["source"]).replace(
    "The complete project is embedded, so no GitHub push is required.",
    "The project is cloned from GitHub. Upload the project contents to the repository root first. "
    "Set REPO_REF to a branch, tag or commit for reproducible runs.")
clone_cells[0]["source"] = intro.splitlines(True)
settings = "".join(clone_cells[1]["source"]).replace(
    'PROJECT_DIR = Path("/kaggle/working/yolo11s-grl")',
    'REPO_URL = "https://github.com/binhdoquoc/GRL_RDD26.git"\n'
    'REPO_REF = ""  # empty uses the repository default branch; or set a tag/commit\n'
    'REPO_DIR = Path("/kaggle/working/GRL_RDD26")\n'
    'PROJECT_DIR = REPO_DIR  # project files are at the repository root')
clone_cells[1]["source"] = settings.splitlines(True)
clone_cells[2]["source"] = ["## 1. Clone the project\n",
    "Reruns reuse the checkout and preserve edited YAML and results. To update code, "
    "start a fresh session or explicitly update the checkout before setup.\n"]
clone_cells[3]["metadata"] = {}
clone_cells[3]["source"] = dedent('''
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        if REPO_REF:
            subprocess.run(["git", "checkout", REPO_REF], cwd=REPO_DIR, check=True)
    elif not (REPO_DIR / ".git").is_dir():
        raise RuntimeError(f"Existing path is not a Git checkout: {REPO_DIR}")
    if REPO_REF:
        wanted = subprocess.check_output(["git", "rev-parse", REPO_REF], cwd=REPO_DIR, text=True).strip()
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
        if actual != wanted:
            raise RuntimeError("Existing checkout differs from REPO_REF. Start a fresh session or update it explicitly.")
    if not (PROJECT_DIR / "train.py").is_file():
        raise FileNotFoundError("The cloned revision lacks train.py at the repository root. Upload the project contents, then select that revision.")
    print("Project:", PROJECT_DIR)
    print("Revision:", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip())
    PROJECT_FILES = [str(p.relative_to(PROJECT_DIR))
                     for folder in ("grl_yolo", "configs", "scripts", "tests")
                     for p in (PROJECT_DIR / folder).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".yaml"}]
    PROJECT_FILES += ["train.py", "requirements.txt", "pyproject.toml", "pytest.ini", "README.md"]
''').strip().splitlines(True)
clone_output = PROJECT / "kaggle_yolo11s_grl_clone.ipynb"
clone_output.write_text(json.dumps(clone_notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(clone_output)
