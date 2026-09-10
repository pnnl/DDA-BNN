"""Production inference API for trained aerosol BNN models.

Provides ``run_inference_latent`` (predictions in latent / transformed
space) and ``run_inference_phys`` (physical-space predictions with nested
Monte Carlo sampling and optional temperature-calibrated uncertainty).
Automatically loads ``taus.json`` when ``taus="auto"`` (the default).

Example::

    from training.inference_api import run_inference_phys, find_latest_run_dir

    run_dir = find_latest_run_dir()
    mean, std_ale, std_epi, std_tot, q_out = run_inference_phys(run_dir, x_raw)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import warnings

import numpy as np
import torch
import yaml
import pyro
from scipy.special import expit

# Suppress FutureWarning from Pyro's param_store calling torch.load without weights_only=True
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

import training.config as cfg
from training.model import HybridNet
from training.train import split_output


# ---------------- project root & latest run dir -----------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_run_dir(root: str | Path | None = None) -> Path:
    """Find the most recent artifact directory that contains model.pth.

    Handles flat (artifacts/<run>/) and nested (artifacts/<sweep>/<run>/) layouts.
    """
    root_path = Path(root) if root is not None else _PROJECT_ROOT / "artifacts"
    if not root_path.is_dir():
        raise FileNotFoundError(f"Artifact root '{root_path}' does not exist")

    candidates: list[Path] = []
    for child in sorted(root_path.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        if (child / "model.pth").exists():
            candidates.append(child)
            continue
        for inner in sorted(child.iterdir(), reverse=True):
            if inner.is_dir() and (inner / "model.pth").exists():
                candidates.append(inner)
                break
    if not candidates:
        raise FileNotFoundError(
            f"No run directory with model.pth found under '{root_path}'"
        )
    return max(candidates, key=lambda p: p.name)


# ---------------- utils (keep yours) ----------------
_BASE_CONFIG = {
    key: getattr(cfg, key)
    for key in getattr(cfg, "_baseline", {})
    if hasattr(cfg, key)
}

def apply_config_used(cfg_module, cfg_dict: dict | None):
    if cfg_dict is None:
        return
    if not isinstance(cfg_dict, dict):
        raise ValueError("config_used.yaml must contain a mapping")

    namespace = getattr(cfg_module, "ns", None)
    for k, v in cfg_dict.items():
        if k.isupper() and hasattr(cfg_module, k):
            setattr(cfg_module, k, v)
            if namespace is not None:
                setattr(namespace, k, v)

def _apply_run_config(run_dir: Path) -> None:
    """Reset to baseline, then apply the selected run's saved configuration."""
    apply_config_used(cfg, _BASE_CONFIG)
    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as stream:
            apply_config_used(cfg, yaml.safe_load(stream))

def resolve_run_dir(model_dir: str | Path) -> Path:
    p = Path(model_dir).expanduser()
    if p.is_file():
        p = p.parent
    if (p / "model.pth").exists():
        return p
    runs = [d for d in p.iterdir() if d.is_dir() and (d / "model.pth").exists()] if p.is_dir() else []
    if runs:
        runs.sort()
        return runs[-1]
    raise FileNotFoundError(f"Could not locate run_dir from: {model_dir}")

def scale_like_training_np(X_raw: np.ndarray, x_scaler) -> np.ndarray:
    X_raw = np.asarray(X_raw, dtype=np.float32)
    mask = np.isnan(X_raw)
    filled = np.where(mask, np.nanmean(X_raw, axis=0), X_raw)
    X_scaled = x_scaler.transform(filled)
    X_scaled[mask] = 0.0
    return X_scaled

def inverse_phys_tf(arr: np.ndarray, tf_info: dict, tf_eps: float) -> np.ndarray:
    out = arr.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            t = np.clip(out[:, j], -50.0, 80.0)
            out[:, j] = np.exp(t) - tf_eps
        elif tf == "logit":
            t = np.clip(out[:, j], -30.0, 30.0)
            out[:, j] = expit(t)
        elif tf == "none":
            pass
        else:
            raise ValueError(f"Unknown tf '{tf}' for target '{col}'")
    return out

def _taus_to_array(
    taus: None | dict[str, float] | np.ndarray | torch.Tensor,
    dtype,
) -> torch.Tensor | None:
    if taus is None:
        return None
    if isinstance(taus, dict):
        missing = [target for target in cfg.TARGETS if target not in taus]
        if missing:
            raise ValueError(
                "taus mapping is missing target(s): " + ", ".join(missing)
            )
        arr = torch.tensor(
            [float(taus[target]) for target in cfg.TARGETS],
            dtype=dtype,
        )
    else:
        arr = torch.as_tensor(taus, dtype=dtype)
    if arr.ndim != 1 or arr.shape[0] != len(cfg.TARGETS):
        raise ValueError(f"taus must be shape (D,), got {tuple(arr.shape)}")
    if not bool(torch.isfinite(arr).all()):
        raise ValueError("taus must contain only finite values")
    if bool((arr < 0).any()):
        raise ValueError("taus must be non-negative")
    return arr


def _load_taus_from_run_dir(run_dir: Path) -> dict[str, float] | None:
    """Load taus.json from a run directory if it exists, else return None."""
    taus_path = run_dir / "taus.json"
    if taus_path.exists():
        with taus_path.open("r") as fh:
            return json.load(fh)
    return None


def _resolve_taus(taus, run_dir: Path, dtype):
    """Resolve automatic or explicit uncertainty-temperature scaling."""
    if isinstance(taus, str):
        if taus.casefold() != "auto":
            raise ValueError(
                "taus string value must be 'auto'; use None to disable scaling"
            )
        taus = _load_taus_from_run_dir(run_dir)
    return _taus_to_array(taus, dtype)


# ---------------- latent inference (now can return samples) ----------------
@torch.no_grad()
def run_inference_latent(
    model_dir: str | Path,
    x_raw: np.ndarray | torch.Tensor,
    device: torch.device | None = None,
    num_mc: int | None = None,
    seed: int = 0,
    return_samples: bool = False,
    taus: None | str | dict[str, float] | np.ndarray | torch.Tensor = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Always returns summary latent outputs (CPU):
      mu_lat_mean (N,D)
      std_ale_lat (N,D) or None
      std_epi_lat (N,D)

    If return_samples=True also returns (CPU):
      mu_lat_s (S,N,D)
      std_ale_lat_s (S,N,D) or None

    taus:
      - "auto" (default) → load from run_dir/taus.json if it exists, else no scaling
      - None → no scaling
      - dict / array / tensor → use directly
    """
    run_dir = resolve_run_dir(model_dir)
    _apply_run_config(run_dir)

    if device is None:
        device = torch.device(cfg.DEVICE) if isinstance(cfg.DEVICE, str) else cfg.DEVICE
    num_mc = int(num_mc or getattr(cfg, "BAYES_NUM_SAMPLES", 50))

    meta = torch.load(run_dir / "data_meta.pt", map_location="cpu", weights_only=False)
    x_scaler = meta["x_scaler"]
    y_scaler = meta["y_scaler"]

    X = x_raw.detach().cpu().numpy().astype(np.float32) if isinstance(x_raw, torch.Tensor) else np.asarray(x_raw, dtype=np.float32)
    n_expected = int(getattr(x_scaler, "n_features_in_", None) or x_scaler.mean_.shape[0])
    if X.ndim != 2 or X.shape[1] != n_expected:
        raise ValueError(f"Input must have shape (N,{n_expected}), got {X.shape}")
    xb = torch.tensor(scale_like_training_np(X, x_scaler), dtype=torch.float32, device=device)

    model = HybridNet(xb.shape[1]).to(device)
    state = torch.load(run_dir / "model.pth", map_location=device, weights_only=False)
    state = {k: v for k, v in state.items() if not k.startswith("guide.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    pyro.set_rng_seed(seed)
    pyro.clear_param_store()
    pyro_path = run_dir / "pyro_params.pt"
    has_pyro = pyro_path.exists() and getattr(model, "has_bayes", False)
    if has_pyro:
        pyro.get_param_store().load(str(pyro_path))
        model.guide(xb[:1])

    if has_pyro:
        pred = pyro.infer.Predictive(model, guide=model.guide, num_samples=num_mc, return_sites=["_RETURN"])(xb)
        outs = pred["_RETURN"]  # (S,N,H)
    else:
        outs = model(xb).unsqueeze(0)  # (1,N,H)

    S_eff, N, H = outs.shape
    D = len(cfg.TARGETS)

    flat = outs.reshape(S_eff * N, H)
    mu_norm_flat, std_norm_flat = split_output(flat)
    mu_norm = mu_norm_flat.reshape(S_eff, N, D)
    std_norm = std_norm_flat.reshape(S_eff, N, D) if std_norm_flat is not None else None

    if cfg.USE_ZSCORE_SCALING:
        scale = torch.as_tensor(y_scaler.scale_, device=device, dtype=mu_norm.dtype)
        mean  = torch.as_tensor(y_scaler.mean_,  device=device, dtype=mu_norm.dtype)
        mu_lat_s = mean + scale * mu_norm
        std_lat_s = scale * std_norm if std_norm is not None else None
    else:
        mu_lat_s, std_lat_s = mu_norm, std_norm

    # ------------------- tau scaling (Option B style) -------------------
    taus_arr = _resolve_taus(taus, run_dir, dtype=mu_lat_s.dtype)
    if taus_arr is not None:
        taus_arr = taus_arr.to(device)  # (D,)

        # scale epistemic spread of posterior means about their mean
        mu_bar = mu_lat_s.mean(dim=0, keepdim=True)                # (1,N,D)
        mu_lat_s = mu_bar + taus_arr.view(1, 1, D) * (mu_lat_s - mu_bar)

        # scale aleatoric per posterior draw (if available)
        if std_lat_s is not None:
            std_lat_s = taus_arr.view(1, 1, D) * std_lat_s

    # ------------------- summaries (from scaled samples) ----------------
    mu_lat_mean = mu_lat_s.mean(dim=0)  # (N,D)
    std_epi_lat = mu_lat_s.var(dim=0, unbiased=False).sqrt() if S_eff > 1 else torch.zeros((N, D), device=device)
    std_ale_lat = (std_lat_s.pow(2).mean(dim=0)).sqrt() if std_lat_s is not None else None

    mu_lat_s_cpu = mu_lat_s.detach().cpu() if return_samples else None
    std_ale_lat_s_cpu = std_lat_s.detach().cpu() if (return_samples and std_lat_s is not None) else None

    return (
        mu_lat_mean.detach().cpu(),
        (None if std_ale_lat is None else std_ale_lat.detach().cpu()),
        std_epi_lat.detach().cpu(),
        mu_lat_s_cpu,
        std_ale_lat_s_cpu,
    )


# ---------------- physical inference (Option B: use posterior samples) ----------------
def stats_from_nested_cloud(y_kl: np.ndarray, quantiles=(0.05, 0.5, 0.95)):
    K, L, N, D = y_kl.shape
    y_pool = y_kl.reshape(K * L, N, D)
    mean = np.mean(y_pool, axis=0)
    var_tot = np.var(y_pool, axis=0, ddof=0)
    std_tot = np.sqrt(np.maximum(var_tot, 1e-24))
    var_ale = np.mean(np.var(y_kl, axis=1, ddof=0), axis=0)  # E_k[Var_l]
    var_epi = np.var(np.mean(y_kl, axis=1), axis=0, ddof=0)  # Var_k[E_l]
    std_ale = np.sqrt(np.maximum(var_ale, 1e-24))
    std_epi = np.sqrt(np.maximum(var_epi, 1e-24))
    q_out = {f"q{int(round(q*100)):02d}": np.quantile(y_pool, q, axis=0) for q in quantiles}
    return mean, std_ale, std_epi, std_tot, q_out


@torch.no_grad()
def run_inference_phys(
    model_dir: str | Path,
    x_raw: np.ndarray | torch.Tensor,
    device: torch.device | None = None,
    num_mc: int | None = None,
    seed: int = 0,
    taus: None | str | dict[str, float] | np.ndarray | torch.Tensor = "auto",
    L: int = 50,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
):
    """
    Option B physical inference:
      - outer loop = posterior draws s (epistemic)
      - inner loop = aleatoric noise l
      - apply tau to BOTH:
          mu_s' = mu_bar + tau*(mu_s - mu_bar)
          std_ale_s' = tau*std_ale_s
    Returns:
      mean_phys, std_ale_phys, std_epi_phys, std_tot_phys, q_out
    """
    run_dir = resolve_run_dir(model_dir)
    # Apply the saved run configuration before reading TF_EPS. The nested
    # latent call applies the same run config again, which is idempotent.
    _apply_run_config(run_dir)

    meta = torch.load(run_dir / "data_meta.pt", map_location="cpu", weights_only=False)
    tf_info = meta["tf_info"]
    tf_eps = cfg.TF_EPS

    # NOTE: taus=None here is deliberate. run_inference_latent() defaults to
    # "auto" (auto-loading taus.json) when called on its own, but this function
    # applies its own calibration below using samples straight from the
    # posterior. Letting run_inference_latent also calibrate here would apply
    # the tau factor twice (approximately squaring it).
    mu_lat_mean, std_ale_lat, std_epi_lat, mu_lat_s, std_ale_lat_s = run_inference_latent(
        run_dir, x_raw, device=device, num_mc=num_mc, seed=seed, return_samples=True, taus=None,
    )
    if mu_lat_s is None:
        raise RuntimeError("return_samples=True failed to produce mu_lat_s")

    mu_lat_s_np = mu_lat_s.numpy()  # (S,N,D)
    if std_ale_lat_s is not None:
        std_ale_lat_s_np = std_ale_lat_s.numpy()
    else:
        std_ale_lat_s_np = np.zeros_like(mu_lat_s_np)

    S_eff, N, D = mu_lat_s_np.shape

    # tau array
    taus_arr = _resolve_taus(taus, run_dir, dtype=torch.float32)
    if taus_arr is None:
        tau_np = np.ones((D,), dtype=np.float32)
    else:
        tau_np = taus_arr.detach().cpu().numpy().astype(np.float32)

    # scale epistemic spread of posterior means
    mu_bar = mu_lat_s_np.mean(axis=0, keepdims=True)  # (1,N,D)
    mu_lat_s_cal = mu_bar + tau_np[None, None, :] * (mu_lat_s_np - mu_bar)

    # scale aleatoric std per posterior draw
    std_ale_lat_s_cal = tau_np[None, None, :] * std_ale_lat_s_np

    # nested sampling in latent space
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=(S_eff, L, N, D)).astype(np.float32)
    t = mu_lat_s_cal[:, None, :, :] + std_ale_lat_s_cal[:, None, :, :] * eps  # (S,L,N,D)

    # inverse transform to physical
    y = inverse_phys_tf(t.reshape(S_eff * L * N, D), tf_info, tf_eps).reshape(S_eff, L, N, D)

    return stats_from_nested_cloud(y, quantiles=quantiles)


# ------------------------------------------------------------------ #
# Smoke-test: python -m training.inference_api                       #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import pandas as pd

    run_dir = find_latest_run_dir()
    print(f"[INFO] run_dir → {run_dir}")

    # load taus status
    taus_path = run_dir / "taus.json"
    if taus_path.exists():
        with taus_path.open("r") as fh:
            taus_dict = json.load(fh)
        print(f"[INFO] taus loaded: {taus_dict}")
    else:
        print("[WARN] No taus.json found — running without calibration scaling")

    # determine expected input dimension from saved scaler
    meta = torch.load(run_dir / "data_meta.pt", map_location="cpu", weights_only=False)
    Din = int(getattr(meta["x_scaler"], "n_features_in_", None)
              or meta["x_scaler"].mean_.shape[0])

    N = 16
    x_dummy = torch.rand(N, Din)
    print(f"[INFO] dummy input: ({N}, {Din})")

    # --- latent ---
    print("\n--- Latent inference ---")
    mu_lat, std_ale_lat, std_epi_lat, mu_lat_s, std_ale_lat_s = run_inference_latent(
        run_dir, x_dummy, num_mc=50, seed=0, return_samples=True,
    )
    print(f"  mu_lat:      {mu_lat.shape}")
    print(f"  std_ale_lat: {None if std_ale_lat is None else std_ale_lat.shape}")
    print(f"  std_epi_lat: {std_epi_lat.shape}")
    print(f"  mu_lat_s:    {None if mu_lat_s is None else mu_lat_s.shape}")

    # --- physical ---
    print("\n--- Physical inference ---")
    mean_phys, std_ale_phys, std_epi_phys, std_tot_phys, q_out = run_inference_phys(
        run_dir, x_dummy, num_mc=50, seed=0, L=50,
    )
    print(f"  mean_phys:    {mean_phys.shape}")
    print(f"  std_tot_phys: {std_tot_phys.shape}")
    print(f"  q_out keys:   {list(q_out.keys())}")

    # --- variance decomposition sanity check ---
    max_err = np.max(np.abs(std_tot_phys**2 - (std_ale_phys**2 + std_epi_phys**2)))
    print(f"\n  Var decomposition max error: {max_err:.2e}",
          "✓" if max_err < 1e-4 else "⚠ large")

    # --- summary table ---
    targets = cfg.TARGETS
    out = {}
    for j, t in enumerate(targets):
        out[f"mean_{t}"]    = mean_phys[:, j]
        out[f"std_tot_{t}"] = std_tot_phys[:, j]
        out[f"std_ale_{t}"] = std_ale_phys[:, j]
        out[f"std_epi_{t}"] = std_epi_phys[:, j]
        out[f"q05_{t}"]     = q_out["q05"][:, j]
        out[f"q50_{t}"]     = q_out["q50"][:, j]
        out[f"q95_{t}"]     = q_out["q95"][:, j]

    df = pd.DataFrame(out)
    print(f"\n--- Results (first 5 rows) ---\n{df.head().to_string(float_format='{:.6f}'.format)}")
    print("\n[OK] Smoke test passed.")
