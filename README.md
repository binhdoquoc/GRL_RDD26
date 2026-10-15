# YOLO11s + local multiscale GRL extension

The workflow is: install dependencies if needed → apply the local extension →
load the model YAML → edit training hyperparameters → train using Ultralytics'
native loop and default result folders.

Local `GRLDetectionModel` and `GRLTrainer` subclasses apply the changes at runtime.
The installed Ultralytics source files are not overwritten. This follows its
custom-trainer extension mechanism and makes the base detector independently
usable. The implementation targets Ultralytics 8.3.0 / PyTorch 2.5.1.

## 1. Install the local extension

Use Python 3.10–3.12. From this folder:

```bash
python -m pip install -e .
```

This installs missing pinned dependencies and registers the `yolo11s-grl`
command. An editable install uses the files in `grl_yolo/` directly, so local
code edits take effect in a new Python process. For tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Install an appropriate PyTorch CUDA build first if your local platform needs one.
The Kaggle notebook creates an isolated Python 3.12 environment with PyPI Linux
CUDA packages using uv. The notebook kernel may remain on Python 3.13.

## 2. Load and edit the architecture YAML

`configs/yolo11s-grl.yaml` contains the YOLO11s backbone, neck/detection head,
small-model scaling and a `grl` section:

```yaml
grl:
  enabled: true
  layers: [4, 6, 10]
  hidden_channels: null
  norm: group
  groups: 8
  lambda: 0.1
  domain_weight: 1.0
  schedule: dann
```

The local extension registers a classifier for each tapped feature map. Each
branch applies GRL → Conv 1×1 → normalization → LeakyReLU → Conv 1×1.
`hidden_channels` can specify one width per branch. GroupNorm or BatchNorm is
selectable. Target images traverse only the backbone, so target data does not
update neck/head BatchNorm statistics or receive detection loss.

When changing backbone indices, update `grl.layers` and normal YOLO graph
connections accordingly. Model YAMLs must describe a valid detection graph.
The class count is overridden from dataset `names` by the native trainer.

Use the supplied entry point or `GRLYOLO` API to interpret `grl`. Plain
`ultralytics.YOLO(model_yaml)` parses the standard graph but does not activate
this extension's GRL branches or target-batch training.

## 3. Edit dataset and training hyperparameters

`configs/adaptiverdd.yaml` describes the dataset:

```yaml
path: /path/to/adaptiverdd-prepared
train: source/images
target: target_adaptation/images
val: target_eval/images
test: target_test/images
names: [D00, D10, D20, D40]
```

Source/validation/test images need parallel `labels/` folders with normalized
YOLO detection rows. Target training needs only images, and never reads target
labels. A labeled validation split is required. Keep final test images held out.

`configs/train.yaml` selects architecture, initialization and data:

```yaml
train:
  model: yolo11s-grl.yaml
  pretrained: yolo11s.pt
  data: adaptiverdd.yaml
  optimizer: AdamW
  lr0: 0.002
  # Optional overrides:
  # epochs: 100
  # batch: 4
  # imgsz: 640
  # device: 0

experiment:
  seeds: [0, 1, 2]  # use [0] for one adaptive/baseline pair
```

Paths to local model/data files resolve relative to the training YAML.
`yolo11s.pt` is downloaded by Ultralytics if absent, then matching weights are
transferred into the YAML-defined model. Set `pretrained: false` for random
initialization, or supply a compatible local checkpoint.

The supplied experiment explicitly uses AdamW and lr0=0.002. All other omitted
training settings retain Ultralytics 8.3.0 defaults, including
loss gains, warmup, LR scheduling, AMP, EMA, augmentation, early
stopping, validation and output-directory naming. Source and target both use
the native YOLODataset augmentation pipeline, including mosaic closure.
`batch` is the source batch size plus a target batch of the same configured size.
The current GRL adapter supports one GPU and a positive explicit batch size;
native AutoBatch does not account for target autograd graphs.

You may add a top-level `grl` section to training YAML for experiment-specific
overrides; these take precedence over the model YAML's GRL settings. Existing
training configs with inline GRL settings remain supported.

## 4. Train

The default `experiment.seeds: [0, 1, 2]` repeats adaptive training three times.
In Kaggle set `RUN_BASELINE = True`; the next cell repeats the matching baselines
for the same seeds, using the same source settings and initial pretrained weights.
Changing the YAML to `seeds: [0]` runs one of each. No augmentation changes are added.
Each seed initializes a fresh trainer and optimizer. The current GRL schedule
remains DANN with maximum lambda 0.1 and domain_weight 1.0.

The first seed uses the native run directory (normally `runs/detect/train` for
adaptive and `train2` for baseline). Additional seeds live inside that directory's
`seeds/<seed>/` subtree so the fixed notebook's recursive download includes every
run. Each mode writes `seed_runs.json`, `seed_results.csv` and `seed_summary.csv`.
The baseline summary also includes the adaptive runs when their seed list matches
the current `notebook_run.json`. The summary reports mean and sample standard
deviation; standard deviation is blank for a single seed. Each best detector is
evaluated on `test`, or `val` when no test split exists. The notebook's existing
plots/evaluation cells display the first seed; use the CSV summaries for all seeds.
Old results are cleared once before the adaptive sweep, never between seeds or
before its following baseline sweep. Removing the `experiment` section restores
the ordinary single-run CLI, including checkpoint resume support.

```bash
yolo11s-grl --config configs/train.yaml
```

Without an editable installation, the compatibility launcher works:

```bash
python train.py --config configs/train.yaml
```

Source-only baseline using the same native settings:

```bash
yolo11s-grl --config configs/train.yaml --baseline
```

Or use the local model API:

```python
from grl_yolo.api import GRLYOLO

model = GRLYOLO("configs/yolo11s-grl.yaml")
model.load("yolo11s.pt")  # downloads if needed
results = model.train(data="configs/adaptiverdd.yaml", epochs=100, batch=4)
```

The native trainer optimizes its original summed source detection loss plus
`source_batch_size * domain_weight * mean_domain_BCE`.
Domain BCE averages source=1/target=0 and then scales. GRL multiplies feature
gradients by `-lambda`, while classifier gradients keep their ordinary sign.

## 5. Native results and resume

With `project` and `name` omitted, native training results go to
`runs/detect/train`, `train2`, etc. relative to the process working directory.
Set standard `train.project`/`train.name` only when you want an explicit override.
Baseline runs use the same native directory incrementing. Notebook validation
and prediction also use native save directories.

Native outputs include `weights/best.pt`, `weights/last.pt`, `results.csv`,
`args.yaml`, metrics and plots. Best weights use native validation fitness.
Additional GRL outputs in the same run directory are `grl.yaml`, `domain.csv`,
`configuration.yaml` and `weights/resume.pt`. The CLI also exports standard
`weights/best_detector.pt` and `weights/last_detector.pt` without custom modules.
The API's native best/last checkpoints contain the local model class and require
this package to remain importable; use `export_checkpoints(model.trainer)` from
`grl_yolo.cli` for standard detector exports after an API run.

```bash
yolo11s-grl --config configs/train.yaml --resume runs/detect/train/weights/resume.pt
yolo detect val model=runs/detect/train/weights/best_detector.pt data=configs/adaptiverdd.yaml split=test
```

Resume an interrupted run using the same GRL settings. Native resume restores
the original epoch budget. For additional training after a completed run, start
a fresh run initialized from its detector weights. Checkpoints from the older
standalone loop (`adaptation.pt`) cannot resume this native trainer.

## Kaggle notebook

For **baseline only**, import `kaggle_baseline_only.ipynb`. Set `GRL_RUN_DIR` to
the completed GRL run folder containing `args.yaml` and `configuration.yaml`.
In a new session, attach those saved outputs as a Kaggle input. The repository
helper `scripts/run_baseline.py` copies resolved training parameters and the
original pretrained initialization, disables GRL and starts a fresh native run.
If original architecture/data paths no longer exist, supply copies using
`ORIGINAL_MODEL_YAML` and `ORIGINAL_DATA_YAML`. Preserve their original contents
and adjust only dataset mount paths when necessary. Copied settings and explicit
path replacements are recorded in `baseline_settings_copy.json`. The existing
clone notebook remains unchanged.

Both notebooks run `scripts/check_kaggle_access.py` before training. It reads and
decodes every image in the selected splits, reads existing source/evaluation
labels (never adaptation labels), and checks output write/read/delete access.
It reports exact failing paths and exits with an error if checks fail. This
checks the current runtime mount; a later session or mount failure remains possible.
You can also run it directly after installing dependencies:

```bash
python scripts/check_kaggle_access.py --root /kaggle/input/datasets/doquocbinhbinh/adaptiverdd-prepared --output /kaggle/working/GRL_RDD26/runs
```

For the GitHub workflow, upload the contents of this project folder to the root
of [GRL_RDD26](https://github.com/binhdoquoc/GRL_RDD26) (so `train.py`, `grl_yolo/`
and `configs/` are directly at the repository root). Do not upload datasets,
environments, model weights or run outputs. Then import
`kaggle_yolo11s_grl_clone.ipynb` into Kaggle. Its `REPO_URL` is configured for that
repository. Set
optionally `REPO_REF` (branch, tag or commit), attach the dataset, enable GPU and
Internet, and run in order. An empty ref clones the repository default branch.
The notebook prints the checked-out commit, uses the local extension from that
checkout. Setup and every new training command fetch the selected GitHub revision
and print its commit; the CLI prints the resolved GRL settings. Tracked local edits
stop the update rather than being discarded. Existing run outputs and untracked
experiment YAML are preserved. Results remain in the project's
native `runs/detect` folders and are packaged into the downloadable results ZIP.

Import `kaggle_yolo11s_grl.ipynb`, attach the dataset, enable GPU and Internet,
and run in order. It fetches the same GitHub project, installs pinned dependencies
if needed, loads repository `configs/train.yaml` and `configs/yolo11s-grl.yaml`, trains,
evaluates and packages the actual native run directories. Both public notebook
names now use this GitHub workflow; push repository changes before training.
The notebook overrides only the Kaggle dataset YAML path and GPU selection.
Hyperparameters and GRL settings come from the repository configuration files.
Each fresh primary training command clears the previous local `runs/detect`
experiment outputs. Download needed results before starting another experiment.
The subsequent `--baseline` command preserves the current GRL outputs for comparison.
Resume and custom project directories are not cleared. Inputs inside the old
results folder must be copied elsewhere before starting a fresh experiment.

It reads the supplied dataset paths directly and uses no symlink staging or
custom directory-access gate. Optional file probes only print diagnostics.
The native loader still needs readable source/target images in the running session.
On read-only Kaggle inputs, an unsaved label-cache warning does not require copies.

Setup avoids ensurepip through uv and uses PyPI by default with custom-index
fallback. It forces Matplotlib `Agg` for subprocesses and displays saved plots
in the kernel. Training output is streamed and saved to `logs/train.log`.

The notebook bootstrap is fixed after this correction. Future model/trainer and
architecture YAML changes belong in GitHub and require no notebook regeneration.
For maintenance of the bootstrap itself:

```bash
python scripts/create_kaggle_notebook.py
```

## Verification

Tests cover analytical GRL/domain gradients, real YOLO11 backward passes, native
training/resume, YAML architecture and experiment override precedence, pretrained
transfer, default output directories, standard exports, bootstrap/logging and
notebook packaging. Verification uses synthetic images on CPU; AdaptiveRDD
accuracy and live Kaggle GPU execution remain unmeasured.

References: [Ultralytics customization](https://docs.ultralytics.com/guides/custom-trainer),
[model YAML guide](https://docs.ultralytics.com/guides/model-yaml-config),
[MS-DAYOLO](https://github.com/Mazin-Hnewa/MS-DAYOLO).
Ultralytics and the adapted model YAML use AGPL-3.0 licensing.
