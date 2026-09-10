from __future__ import annotations
from pathlib import Path
from typing import Union, Optional, Dict, Tuple
import json
import numpy as np
import yaml
import torch
import pyro
from scipy.special import expit

from . import config as cfg
from .model import HybridNet, build_cov_from_params


def smooth_logvar(raw):
    """Map raw network output to bounded log-variance."""
    return cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)


def smooth_offdiag(raw):
    """Map raw network output to bounded off-diagonal correlation."""
    return cfg.OFF_DIAG_MAX * torch.tanh(raw)


def split_output(out: torch.Tensor):
    """Split network output into mean and std based on uncertainty mode."""
    d = len(cfg.TARGETS)
    mode = cfg.UNCERTAINTY_MODE.lower()

    if mode == "none":
        return out, None

    if mode == "diag":
        mu, raw = out[:, :d], out[:, d:]
        log_var = smooth_logvar(raw)
        std = torch.exp(0.5 * log_var)
        return mu, std

    mu, params = out[:, :d], out[:, d:]
    raw_log_var = params[:, :d]
    rho_raw = params[:, d:]

    sigma = torch.exp(0.5 * smooth_logvar(raw_log_var))
    cov = build_cov_from_params(sigma, rho_raw)

    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return mu, std


# Capture the fully post-processed import-time configuration (resolved paths
# and torch.device included). Each inference call resets to this state before
# applying a run's saved configuration so values omitted by one run cannot
# leak in from a previous run.
_BASE_CONFIG = {
    key: getattr(cfg, key)
    for key in getattr(cfg, "_baseline", {})
    if hasattr(cfg, key)
}

def apply_config_used(cfg_module, cfg_dict: dict | None):
    """Apply saved config values to module globals and its namespace mirror."""
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
    """Reset to baseline, then apply config_used.yaml for one model run."""
    apply_config_used(cfg, _BASE_CONFIG)

    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        with cfg_path.open("r") as stream:
            apply_config_used(cfg, yaml.safe_load(stream))

def resolve_run_dir(model_dir: Union[str, Path]) -> Path:
    """Resolve a model directory path, finding the latest run if needed."""
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
    """Scale input features using the training scaler, handling NaN values."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    mask = np.isnan(X_raw)
    filled = np.where(mask, np.nanmean(X_raw, axis=0), X_raw)
    X_scaled = x_scaler.transform(filled)
    X_scaled[mask] = 0.0
    return X_scaled


def inverse_phys_tf(arr: np.ndarray, tf_info: dict, tf_eps: float) -> np.ndarray:
    """Apply inverse physical transforms to convert from latent to physical space."""
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
    taus: Union[None, Dict[str, float], np.ndarray, torch.Tensor],
    dtype
) -> Optional[torch.Tensor]:
    """Convert tau values (dict, array, or tensor) to a tensor of shape (D,)."""
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


def _load_taus_from_run_dir(run_dir: Path) -> Optional[Dict[str, float]]:
    """Load taus.json from a run directory if it exists, else return None."""
    taus_path = run_dir / "taus.json"
    if taus_path.exists():
        with taus_path.open("r") as fh:
            return json.load(fh)
    return None


def _resolve_taus(
    taus: Union[None, str, Dict[str, float], np.ndarray, torch.Tensor],
    run_dir: Path,
    dtype,
) -> Optional[torch.Tensor]:
    """Resolve automatic or explicit uncertainty-temperature scaling."""
    if isinstance(taus, str):
        if taus.casefold() != "auto":
            raise ValueError(
                "taus string value must be 'auto'; use None to disable scaling"
            )
        taus = _load_taus_from_run_dir(run_dir)
    return _taus_to_array(taus, dtype)


@torch.no_grad()
def run_inference_latent(
    model_dir: Union[str, Path],
    x_raw: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    num_mc: Optional[int] = None,
    seed: int = 0,
    return_samples: bool = False,
    taus: Union[None, str, Dict[str, float], np.ndarray, torch.Tensor] = "auto",
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Run inference in latent space.

    Returns (all CPU tensors):
        mu_lat_mean: (N, D) mean predictions
        std_ale_lat: (N, D) aleatoric std or None
        std_epi_lat: (N, D) epistemic std

    If return_samples=True, also returns:
        mu_lat_s: (S, N, D) posterior samples
        std_ale_lat_s: (S, N, D) aleatoric std per sample or None

    taus:
        - "auto" (default) -> load run_dir/taus.json if present, else no scaling
        - None             -> no calibration scaling
        - dict / array / tensor -> use directly
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
        outs = pred["_RETURN"]
    else:
        outs = model(xb).unsqueeze(0)

    S_eff, N, H = outs.shape
    D = len(cfg.TARGETS)

    flat = outs.reshape(S_eff * N, H)
    mu_norm_flat, std_norm_flat = split_output(flat)
    mu_norm = mu_norm_flat.reshape(S_eff, N, D)
    std_norm = std_norm_flat.reshape(S_eff, N, D) if std_norm_flat is not None else None

    if cfg.USE_ZSCORE_SCALING:
        scale = torch.as_tensor(y_scaler.scale_, device=device, dtype=mu_norm.dtype)
        mean = torch.as_tensor(y_scaler.mean_, device=device, dtype=mu_norm.dtype)
        mu_lat_s = mean + scale * mu_norm
        std_lat_s = scale * std_norm if std_norm is not None else None
    else:
        mu_lat_s, std_lat_s = mu_norm, std_norm

    taus_arr = _resolve_taus(taus, run_dir, dtype=mu_lat_s.dtype)
    if taus_arr is not None:
        taus_arr = taus_arr.to(device)
        mu_bar = mu_lat_s.mean(dim=0, keepdim=True)
        mu_lat_s = mu_bar + taus_arr.view(1, 1, D) * (mu_lat_s - mu_bar)
        if std_lat_s is not None:
            std_lat_s = taus_arr.view(1, 1, D) * std_lat_s

    mu_lat_mean = mu_lat_s.mean(dim=0)
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


def stats_from_nested_cloud(y_kl: np.ndarray, quantiles=(0.05, 0.5, 0.95)):
    """
    Compute statistics from nested Monte Carlo samples.

    Args:
        y_kl: (K, L, N, D) array where K=epistemic samples, L=aleatoric samples
        quantiles: quantiles to compute

    Returns:
        mean, std_ale, std_epi, std_tot, quantile_dict
    """
    K, L, N, D = y_kl.shape
    y_pool = y_kl.reshape(K * L, N, D)
    mean = np.mean(y_pool, axis=0)
    var_tot = np.var(y_pool, axis=0, ddof=0)
    std_tot = np.sqrt(np.maximum(var_tot, 1e-24))
    var_ale = np.mean(np.var(y_kl, axis=1, ddof=0), axis=0)
    var_epi = np.var(np.mean(y_kl, axis=1), axis=0, ddof=0)
    std_ale = np.sqrt(np.maximum(var_ale, 1e-24))
    std_epi = np.sqrt(np.maximum(var_epi, 1e-24))
    q_out = {f"q{int(round(q*100)):02d}": np.quantile(y_pool, q, axis=0) for q in quantiles}
    return mean, std_ale, std_epi, std_tot, q_out


@torch.no_grad()
def run_inference_phys(
    model_dir: Union[str, Path],
    x_raw: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    num_mc: Optional[int] = None,
    seed: int = 0,
    taus: Union[None, str, Dict[str, float], np.ndarray, torch.Tensor] = "auto",
    L: int = 50,
    quantiles=(0.05, 0.5, 0.95),
):
    """
    Run inference in physical space using nested Monte Carlo sampling.

    Outer loop (K) = posterior/epistemic samples
    Inner loop (L) = aleatoric noise samples

    taus:
        - "auto" (default) -> load run_dir/taus.json if present, else no scaling
        - None             -> no calibration scaling
        - dict / array / tensor -> use directly
    Applies temperature scaling to both epistemic and aleatoric spread.

    Returns:
        mean_phys, std_ale_phys, std_epi_phys, std_tot_phys, quantile_dict
    """
    run_dir = resolve_run_dir(model_dir)
    _apply_run_config(run_dir)

    meta = torch.load(run_dir / "data_meta.pt", map_location="cpu", weights_only=False)
    tf_info = meta["tf_info"]
    tf_eps = cfg.TF_EPS

    # NOTE: taus=None here is deliberate. run_inference_latent() defaults to
    # "auto" (auto-loading taus.json) when called on its own, but this function
    # applies its own calibration below (lines further down) using samples
    # straight from the posterior. Letting run_inference_latent also calibrate
    # here would apply the tau factor twice (approximately squaring it).
    mu_lat_mean, std_ale_lat, std_epi_lat, mu_lat_s, std_ale_lat_s = run_inference_latent(
        run_dir, x_raw, device=device, num_mc=num_mc, seed=seed,
        return_samples=True, taus=None,
    )
    if mu_lat_s is None:
        raise RuntimeError("return_samples=True failed to produce mu_lat_s")

    mu_lat_s_np = mu_lat_s.numpy()
    if std_ale_lat_s is not None:
        std_ale_lat_s_np = std_ale_lat_s.numpy()
    else:
        std_ale_lat_s_np = np.zeros_like(mu_lat_s_np)

    S_eff, N, D = mu_lat_s_np.shape

    taus_arr = _resolve_taus(taus, run_dir, dtype=torch.float32)
    if taus_arr is None:
        tau_np = np.ones((D,), dtype=np.float32)
    else:
        tau_np = taus_arr.detach().cpu().numpy().astype(np.float32)

    mu_bar = mu_lat_s_np.mean(axis=0, keepdims=True)
    mu_lat_s_cal = mu_bar + tau_np[None, None, :] * (mu_lat_s_np - mu_bar)
    std_ale_lat_s_cal = tau_np[None, None, :] * std_ale_lat_s_np

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(size=(S_eff, L, N, D)).astype(np.float32)
    t = mu_lat_s_cal[:, None, :, :] + std_ale_lat_s_cal[:, None, :, :] * eps

    y = inverse_phys_tf(t.reshape(S_eff * L * N, D), tf_info, tf_eps).reshape(S_eff, L, N, D)

    return stats_from_nested_cloud(y, quantiles=quantiles)

