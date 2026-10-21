# training/data_utils.py
"""Dataset loading, splitting, scaling, and DataLoader construction.

Handles CSV ingestion, engineered features (``Xve``, ``coating_RI_imag``),
forward target transforms (log / logit), train-only z-score scaling, and
optional balanced sampling.  Returns ready-to-use PyTorch DataLoaders
plus a ``meta`` dict containing scalers and raw arrays for evaluation.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import training.config as cfg

# ──────────────────────────────────────────────────────────────────────
# 0)  Global seeding helper
# ──────────────────────────────────────────────────────────────────────
def _set_global_seed(seed: int):
    """Seed Python, NumPy, Torch (CPU & CUDA) and CuDNN for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python ≥3.3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # deterministic CuDNN (optional but often helpful)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if cfg.SEED is not None:
    _set_global_seed(cfg.SEED)


def _apply_target_forward(df_targets):
    """Return transformed ndarray and per-target metadata."""
    Y = df_targets.copy()
    info = {}
    for col in df_targets.columns:
        tf = cfg.TARGET_TF.get(col, "none")
        if tf == "log":
            Y[col] = np.log(df_targets[col] + cfg.TF_EPS)
        elif tf == "logit":
            Y[col] = logit(np.clip(df_targets[col], cfg.TF_EPS, 1 - cfg.TF_EPS))
        info[col] = tf
    return Y.values, info


# ──────────────────────────────────────────────────────────────────────
# helper: dummy scaler when Z-score is disabled
# ──────────────────────────────────────────────────────────────────────
class IdentityScaler:
    """No-op scaler; NaN means missing and maps to zero at the caller boundary."""
    def fit(self, x):
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("IdentityScaler expects a 2-D input array")
        self._dim = x.shape[1]
        return self

    def transform(self, x):         return x

    def fit_transform(self, x):     return self.fit(x).transform(x)

    def inverse_transform(self, x): return x

    @property
    def scale_(self):               return np.ones(self._dim)

    @scale_.setter
    def scale_(self, v):            self._dim = len(v)

    @property
    def n_features_in_(self):      return self._dim

    @property
    def mean_(self):               return np.zeros(self._dim)


# ──────────────────────────────────────────────────────────────────────
# feature helpers
# ──────────────────────────────────────────────────────────────────────
def _extract_imaginary(col: pd.Series) -> np.ndarray:
    def one(x):
        if pd.isna(x):             return np.nan
        if isinstance(x, complex): return x.imag
        try:
            return complex(x).imag
        except Exception:
            return np.nan

    return col.apply(one).to_numpy(dtype=float)


def _scale_and_zero_missing(arr: np.ndarray,
                            scaler: StandardScaler | None = None,
                            fit: bool = True) -> tuple[np.ndarray, StandardScaler]:
    """
    Fill NaNs with column means, StandardScaler transform, then set missing entries to 0.

    If fit=True: fit a new scaler on this array.
    If fit=False: use the provided scaler (must not be None).
    """
    mask = np.isnan(arr)
    filled = np.where(mask, np.nanmean(arr, axis=0), arr)

    if fit:
        scaler = StandardScaler().fit(filled)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")

    scaled = scaler.transform(filled)
    scaled[mask] = 0.0
    return scaled, scaler


# ──────────────────────────────────────────────────────────────────────
# scientific-transform helpers
# ──────────────────────────────────────────────────────────────────────
BOUNDED_TARGETS = {"SSA", "g"}
LOG_TARGETS_POS = {"Qext"}


def _forward_tf(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    y = df.copy()
    eps = 1e-6
    for col in y.columns:
        if col in BOUNDED_TARGETS:
            y[col] = logit(np.clip(y[col], eps, 1 - eps))
        elif col in LOG_TARGETS_POS:
            y[col] = np.log(y[col] + eps)
    return y.values, {"eps": eps}


# ──────────────────────────────────────────────────────────────────────
# DataLoader helpers
# ──────────────────────────────────────────────────────────────────────
def _seed_worker(worker_id: int):
    """Ensures each DataLoader worker has a deterministic and unique seed."""
    worker_seed = (cfg.SEED + worker_id) % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Shared generator for all DataLoaders
_dl_generator = torch.Generator()
_dl_generator.manual_seed(cfg.SEED if cfg.SEED is not None else 0)


def load_dataset(path: Path,
                 features: list[str],
                 targets: list[str],
                 train_fraction: float,
                 val_fraction: float = 0.1,
                 device=cfg.DEVICE):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # ---- deduplicate exact input rows ----

    dedup_cols = ["Npp", "V/V0", "core_Df", "coating_RI"]
    if {"Dve", "wavelength"}.issubset(df.columns):
        dedup_cols += ["Dve", "wavelength"]

    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
    after = len(df)
    print(f"[INFO] Deduplicated dataset: dropped {before - after} rows (exact duplicates by {dedup_cols})")


    # engineered feature ------------------------------------------------
    if {"Dve", "wavelength"}.issubset(df.columns):
        df["Xve"] = np.pi * df["Dve"] / df["wavelength"]

    # ------------------ build X_raw -----------------------------------
    cols = []
    for f in features:
        if f == "coating_RI_imag":
            cols.append(_extract_imaginary(df["coating_RI"]))
        else:
            cols.append(df[f].to_numpy(dtype=float))

    low_res_cols = [df[f"{t}_HS"].to_numpy(dtype=float) for t in targets]
    cols.extend(low_res_cols)

    X_raw = np.column_stack(cols)
    # Missing values use NaN; infinity is never a valid input to the network.
    if np.isinf(X_raw).any():
        raise ValueError("Infinite feature values are invalid; use NaN for missing")
    low_res_raw = X_raw[:, -len(targets):]

    # ------------------ build y_raw (latent/forward-transformed) -------
    y_raw_np, tf_info = _apply_target_forward(df[targets])  # (n, D) in latent t-space (unscaled)

    # ------------------ split FIRST -----------------------------------
    n = len(X_raw)
    n_train = int(train_fraction * n)
    n_val = int(val_fraction * n)

    idx = np.random.permutation(n)
    idx_tr = idx[:n_train]
    idx_va = idx[n_train: n_train + n_val]
    idx_te = idx[n_train + n_val:]

    # ------------------ scale using TRAIN ONLY -------------------------
    if cfg.USE_ZSCORE_SCALING:
        # fit scalers on train subset only, then transform all splits
        X_tr, x_scaler = _scale_and_zero_missing(X_raw[idx_tr], fit=True)
        X_va, _ = _scale_and_zero_missing(X_raw[idx_va], scaler=x_scaler, fit=False)
        X_te, _ = _scale_and_zero_missing(X_raw[idx_te], scaler=x_scaler, fit=False)

        y_tr, y_scaler = _scale_and_zero_missing(y_raw_np[idx_tr], fit=True)
        y_va, _ = _scale_and_zero_missing(y_raw_np[idx_va], scaler=y_scaler, fit=False)
        y_te, _ = _scale_and_zero_missing(y_raw_np[idx_te], scaler=y_scaler, fit=False)
    else:
        x_scaler = IdentityScaler(); x_scaler.scale_ = np.ones(X_raw.shape[1])
        y_scaler = IdentityScaler(); y_scaler.scale_ = np.ones(y_raw_np.shape[1])

        # In no-scaling mode, missing features still need the same zero fill
        # as production inference; never pass NaNs through to the network.
        # This zero is in raw units and may not be physically neutral for
        # size/morphology features; changing the imputation policy requires
        # a separate modeling decision and retraining.
        X_tr, X_va, X_te = (np.where(np.isnan(X_raw[idx]), 0.0, X_raw[idx])
                             for idx in (idx_tr, idx_va, idx_te))
        y_tr, y_va, y_te = y_raw_np[idx_tr], y_raw_np[idx_va], y_raw_np[idx_te]

    # -----------------------------------------------------------------
    # Balanced sampler (optional, can handle multiple targets)
    # NOTE: y_tr is already in transformed space (and scaled if enabled).
    # -----------------------------------------------------------------
    sampler = None
    shuffle_flag = True
    if cfg.BALANCE_TARGETS:
        weights = np.ones(len(y_tr), dtype=np.float32)
        for tgt in cfg.BALANCE_TARGETS:
            if tgt not in targets:
                raise ValueError(f"{tgt} not among TARGETS {targets}")
            col_idx = targets.index(tgt)
            vals = y_tr[:, col_idx]  # transformed space (scaled if enabled)
            bins = np.linspace(vals.min(), vals.max(), cfg.N_BINS + 1)
            bin_idx = np.digitize(vals, bins) - 1
            freq = np.bincount(bin_idx, minlength=cfg.N_BINS) + 1e-8
            inv_f = 1.0 / freq[bin_idx]
            if cfg.BALANCE_STRATEGY == "prod":
                weights *= inv_f
            elif cfg.BALANCE_STRATEGY == "mean":
                weights += inv_f
            elif cfg.BALANCE_STRATEGY == "max":
                weights = np.maximum(weights, inv_f)
            else:
                raise ValueError("cfg.BALANCE_STRATEGY must be 'prod', 'mean', or 'max'")

        weights = np.clip(weights, 0, 10 * weights.mean())
        if cfg.BALANCE_STRATEGY == "mean":
            weights /= len(cfg.BALANCE_TARGETS)
        weights /= weights.mean()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle_flag = False

    # ------------------ DataLoaders -----------------------------------
    def _make_loader(x, y, shuffle=False, sampler=None):
        ds = TensorDataset(torch.tensor(x, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.float32))
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=0,
            generator=_dl_generator,
            worker_init_fn=_seed_worker if sampler is None else None,
        )

    train_loader = _make_loader(X_tr, y_tr, shuffle=shuffle_flag, sampler=sampler)
    val_loader = _make_loader(X_va, y_va)
    test_loader = _make_loader(X_te, y_te)

    # --- move tensors to device --------------------------------------
    if device != torch.device("cpu"):
        for dl in (train_loader, val_loader, test_loader):
            dl.dataset.tensors = tuple(t.to(device) for t in dl.dataset.tensors)

    meta = {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "tf_info": tf_info,
        # raw physical truth on test set:
        "y_test_raw": df[cfg.TARGETS].values[idx_te],
        "low_res_test_raw": low_res_raw[idx_te],
        "x_test_raw": X_raw[idx_te],
        # (optional but helpful)
        "idx_tr": idx_tr,
        "idx_va": idx_va,
        "idx_te": idx_te,
    }
    return train_loader, val_loader, test_loader, meta