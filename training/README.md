# training/

Core training pipeline for the aerosol DDA BNN

## Pipeline

The full training pipeline runs in order:

```
hyperparm_search_driver.py  →  train.py  →  temp_tune.py  →  inference_api.py
         (sweep)              (train)      (calibrate τ)      (predict)
```

`temp_tune.py` is called automatically at the end of `train.py`, so a single
call to `train.py` or `hyperparm_search_driver.py` produces a fully calibrated
model with `taus.json` ready for inference.

## Scripts

All scripts should be run from the **project root** (`aerosol_bnn/`):

| Script | Purpose | Usage |
|---|---|---|
| `hyperparm_search_driver.py` | Grid search over hyperparameters — runs `train.py` for each combo | `python training/hyperparm_search_driver.py --grid training/configs/grid_search.yaml` |
| `train.py` | End-to-end training + evaluation + temperature calibration | `python training/train.py` |
| `temp_tune.py` | Fit per-target τ on validation predictions, apply to test, save `taus.json` | `python training/temp_tune.py` (auto-detects latest run) |
| `inference_api.py` | Production inference API — latent and physical-space predictions with auto τ loading | `python training/inference_api.py` (smoke test) |

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Global configuration loaded from `configs/default.yaml`; run-time overrides via `cfg.load()` |
| `model.py` | `HybridNet` — deterministic + Bayesian layer architecture, loss functions |
| `data_utils.py` | Dataset loading, train/val/test splitting, z-score scaling, DataLoader creation |
| `plotting.py` | Training plots — loss curves, prediction scatter plots |

## Configuration

All configuration lives in `configs/`. There are two files:

- **`default.yaml`** — Baseline configuration. Every training run starts from
  these values.  Edit this to change features, targets, architecture, learning
  rate, etc.
- **`grid_search.yaml`** — Hyperparameter sweep definition.  Each key maps to a
  **list** of values to try; the grid search driver takes the Cartesian product
  of all lists and runs `train.py` once per combination.  Any key not listed
  here keeps its `default.yaml` value.

Key settings in `default.yaml`:

| Key | Description | Default |
|---|---|---|
| `ROOT_DIR` | Project root (relative to `training/`) | `".."` |
| `DATA_FILE` | Training data CSV | `"data/DDA_dataset_extended_JAN.csv"` |
| `ARTIFACT_DIR` | Output directory for run artifacts | `"artifacts"` |
| `FEATURES` | Input feature columns | `[Npp, V/V0, coating_RI_imag, Xve, core_Df]` |
| `TARGETS` | Output target columns | `[Qext, SSA, g]` |
| `LAYER_SPEC` | Network architecture (layer sizes, types, activations) | 5 layers, 64 units each |
| `UNCERTAINTY_MODE` | `"diag"`, `"full"`, or `"none"` | `"diag"` |
| `EPOCHS` | Maximum training epochs | `5` (increase for real runs) |
| `LR` | Learning rate | `0.0003` |
| `SEED` | Global random seed | `500` |

Override any setting via a YAML file:
```python
import training.config as cfg
cfg.load("my_overrides.yaml")
```

## Artifacts

Each training run produces a timestamped directory under `artifacts/`:

```
artifacts/20260329_131854/
├── config_used.yaml          # exact config for this run
├── model.pth                 # trained model weights
├── data_meta.pt              # scalers, tf_info, raw test/val arrays
├── pyro_params.pt            # Pyro variational parameters (BNN only)
├── loss.csv                  # per-epoch train/val losses
├── loss_components.png       # loss curve plot
├── pred_plot.png             # prediction scatter plot
├── predictions_val.csv       # validation predictions + latent columns
├── predictions_test.csv      # test predictions + latent columns
├── predictions_test_cal.csv  # tau-calibrated test predictions with quantiles
└── taus.json                 # fitted temperature scaling factors per target
```

## Inference API

```python
from training.inference_api import run_inference_phys, find_latest_run_dir

run_dir = find_latest_run_dir()  # or pass a specific path

# taus="auto" (default) loads taus.json from the run dir
mean, std_ale, std_epi, std_tot, q_out = run_inference_phys(run_dir, x_raw)
```

Key functions:
- `find_latest_run_dir()` — auto-detect most recent run under `artifacts/`
- `run_inference_latent()` — predictions in latent (transformed) space
- `run_inference_phys()` — predictions in physical space with nested MC sampling
- Both default to `taus="auto"` which loads `taus.json` if present
- Pass `taus=None` explicitly to disable temperature calibration
