# MIE Aerosol Property Correction — Inference Package

Pre-trained Bayesian Neural Network for correcting low-resolution (Mie / HS)
aerosol optical properties (Qext, SSA, g) with calibrated uncertainty estimates.

## Quick start

All commands are run from the **repository root** (`aerosol_bnn/`).

### 1. Create the Conda environment

```bash
conda env create -f environment.yml      # first time
conda activate bnn_notebook_env
```

If the environment already exists, just activate it (or run
`conda env update -f environment.yml` to sync dependencies).

### 2. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name bnn_notebook_env --display-name "BNN Notebook"
```

### 3. Run the inference notebook

```bash
jupyter notebook release/inference_notebook.ipynb
```

Run all cells after selecting the "BNN Notebook" kernel.

## What's in this folder

```
release/
├── inference_notebook.ipynb   # interactive demo notebook
├── inference_api.py           # prediction API (latent + physical space)
├── config.py                  # configuration (reads configs/default.yaml)
├── model.py                   # HybridNet model definition
├── configs/
│   └── default.yaml           # feature / target / architecture settings
└── chosen_model/
    ├── model.pth              # trained weights
    ├── data_meta.pt           # scalers and transform metadata
    ├── pyro_params.pt         # Bayesian variational parameters
    ├── taus.json              # temperature calibration factors
    └── config_used.yaml       # exact config the model was trained with
```

## Python API

```python
from release.inference_api import run_inference_phys

# x_raw: (N, 8) array
# columns: [Npp, V/V0, coating_RI_imag, Xve, core_Df, Qext_HS, SSA_HS, g_HS]
mean, std_ale, std_epi, std_tot, q_out = run_inference_phys(
    model_dir="release/chosen_model",
    x_raw=x_raw,
)
```

Both inference entry points default to `taus="auto"`, which loads
`taus.json` from the resolved model run when it is present. Pass
`taus=None` explicitly to request raw, uncalibrated uncertainty.

**Returns** (all NumPy arrays of shape `(N, 3)` for targets Qext, SSA, g):

| Output | Description |
|---|---|
| `mean` | Predictive mean |
| `std_ale` | Aleatoric (data) uncertainty |
| `std_epi` | Epistemic (model) uncertainty |
| `std_tot` | Total uncertainty |
| `q_out` | Dict of quantiles, e.g. `q_out["q05"]`, `q_out["q50"]`, `q_out["q95"]` |

