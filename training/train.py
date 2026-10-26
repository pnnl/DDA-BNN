# training/train.py
"""End-to-end training and evaluation for the aerosol BNN.

Loads data, builds ``HybridNet``, runs the training loop (SVI for
Bayesian layers, deterministic NLL otherwise), evaluates on val/test
splits, saves all artifacts, and calls ``temp_tune`` for temperature
calibration.  Can be invoked directly or via ``hyperparm_search_driver``.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim
import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from scipy.special import expit
import yaml

from training.data_utils import load_dataset
from training.model import HybridNet, mse_loss, gaussian_nll_diag, gaussian_nll_full, build_cov_from_params
from training.plotting import plot_results, plot_loss
import training.config as cfg


# ---------- inverse scientific transform --------------------------------
def inverse_phys_tf(arr: np.ndarray, tf_info: dict[str, str]):
    """
    Undo per-target transforms (log, logit, none).
    """
    out = arr.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            out[:, j] = np.exp(out[:, j]) - cfg.TF_EPS
        elif tf == "logit":
            out[:, j] = expit(out[:, j])
    return out


# ---------- delta-method std propagation --------------------------------
def propagate_std(std_in: np.ndarray,
                  mu_in: np.ndarray,
                  tf_info):
    """
    Propagate std through inverse transforms using first-order Taylor
    expansion (delta method).
    """
    out = std_in.copy()
    idx = {c: i for i, c in enumerate(cfg.TARGETS)}
    for col, tf in tf_info.items():
        j = idx[col]
        if tf == "log":
            out[:, j] = std_in[:, j] * np.exp(mu_in[:, j])
        elif tf == "logit":
            s = expit(mu_in[:, j])
            out[:, j] = std_in[:, j] * s * (1 - s)
    return out


# --- shared smooth maps --------------------------------------------
def smooth_logvar(raw):
    return cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)


def smooth_offdiag(raw):
    return cfg.OFF_DIAG_MAX * torch.tanh(raw)

def early_stop_start_epoch() -> int:
    """
    Start counting early-stopping patience only after deterministic warmup.
    Example: DET_WARMUP=100, PATIENCE=50 => earliest stop at epoch 150.
    """
    return int(getattr(cfg, "DET_WARMUP", 0))

# ----------------------------------------------------------------------
# SVI helper (KL warm-up + per-parameter LR)
# ----------------------------------------------------------------------
def make_svi(model, epoch, warmup=cfg.WARMUP):
    beta = min(1.0, epoch / max(1, warmup))

    EPS = 1.0e-8  # already defined above

    def scaled_guide(*args, **kwargs):
        if beta > 0.0:  # KL active
            with pyro.poutine.scale(scale=beta):
                return model.guide(*args, **kwargs)
        else:  # β = 0 → no scale()
            return model.guide(*args, **kwargs)

    def lr_cfg(name, _):
        """Per-parameter optimiser settings."""
        lr = cfg.LR * cfg.LR_SCALE_FACTOR if "scale" in name else cfg.LR
        cfg_ = {"lr": lr}  # FIXME - Terrible naming clash
        if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
            cfg_["clip_norm"] = cfg.GRAD_CLIP_NORM
        return cfg_

    return SVI(
        model,
        scaled_guide if beta < 1.0 else model.guide,
        ClippedAdam(lr_cfg),  # <- use the clipped optimiser
        loss=TraceMeanField_ELBO(),
    )


# ----------------------------------------------------------------------
# Loss for deterministic training
# ----------------------------------------------------------------------
def make_det_loss():
    d = len(cfg.TARGETS)
    if cfg.UNCERTAINTY_MODE == "none":
        return mse_loss
    if cfg.UNCERTAINTY_MODE == "diag":
        return lambda o, t: gaussian_nll_diag(o, t, d)
    if cfg.UNCERTAINTY_MODE == "full":
        return lambda o, t: gaussian_nll_full(o, t, d)
    raise ValueError("bad UNCERTAINTY_MODE")


# ----------------------------------------------------------------------
# Split raw network output into μ and σ
# ----------------------------------------------------------------------
def split_output(out: torch.Tensor):
    d = len(cfg.TARGETS)
    mode = cfg.UNCERTAINTY_MODE.lower()

    if mode == "none":
        return out, None

    if mode == "diag":
        mu, raw = out[:, :d], out[:, d:]
        log_var = smooth_logvar(raw)  # <- no clamp!
        std = torch.sqrt(torch.exp(log_var) + 1e-6)
        return mu, std

    # -------- full ---------------------------------------------------
    mu, params = out[:, :d], out[:, d:]
    raw_log_var = params[:, :d]  # d components
    rho_raw = params[:, d:]  # remaining

    sigma = torch.sqrt(torch.exp(smooth_logvar(raw_log_var)) + 1e-6)
    cov = build_cov_from_params(sigma, rho_raw)

    # MARGINAL-ONLY API: cross-target covariance is discarded by this return.
    # Joint predictions require carrying cov (or its Cholesky factor) forward.
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return mu, std  # keep interface unchanged (mu, std)


# ----------------------------------------------------------------------
# Training / evaluation helpers
# ----------------------------------------------------------------------

# ---------------- helper ------------------------------------------------

def elbo_parts(model, guide, x, y):
    """
    Return
        nll     :  -log p(y | x , w)             (>=0)
        kl_raw  :   KL(q(w) || p(w))             (>=0)

    Works with any guide (AutoDiagonalNormal, AutoMultivariateNormal, …).
    No beta-scaling is applied here.
    """
    # sample weights  w ~ q(w|φ)
    guide_trace = poutine.trace(guide).get_trace(x, y)
    # run the model with those weights
    model_trace = poutine.trace(
        poutine.replay(model, guide_trace)).get_trace(x, y)

    # make .log_prob fields available
    guide_trace.compute_log_prob()
    model_trace.compute_log_prob()

    # complete guide log-prob  log q(w)
    log_q = guide_trace.log_prob_sum()

    log_p = 0.0     # priors
    log_like = 0.0  # likelihood

    for site in model_trace.nodes.values():
        if site["type"] != "sample":
            continue
        lp = site["log_prob"].sum()
        if site["is_observed"]:
            log_like = lp
        else:
            log_p += lp

    kl_raw = (log_q - log_p).detach()          # >= 0
    nll    = (-log_like).detach()               # >= 0
    return nll.item(), kl_raw.item()

# ---------------- training loop for one epoch ---------------------------
def train_one_epoch(model,
                    loader,
                    beta: float = 1.0,
                    optimiser=None,
                    svi: SVI | None = None,
                    loss_fn=None):
    """
    Returns per-sample averages (tot, nll, kl_beta)

    For BNN:
      - tot is the SVI/ELBO loss (as returned by svi.step) averaged per sample
      - nll and kl_beta are logged from a single posterior draw via elbo_parts(),
        also averaged per sample (so they are on a comparable scale).
    """
    model.train()

    # ---- Bayesian ----------------------------------------------------
    if svi is not None:
        tot_acc = 0.0
        nll_acc = 0.0
        kl_acc  = 0.0
        n_seen  = 0

        for xb, yb in loader:
            xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)

            # logging only (one posterior sample)
            nll_v, kl_raw_v = elbo_parts(model, model.guide, xb, yb)
            nll_acc += nll_v
            kl_acc  += beta * kl_raw_v

            # optimisation
            tot_acc += svi.step(xb, yb)

            n_seen += xb.size(0)

        return tot_acc / n_seen, nll_acc / n_seen, kl_acc / n_seen

    # ---- Deterministic ----------------------------------------------
    tot_acc = 0.0
    N = len(loader.dataset)

    for xb, yb in loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        loss = loss_fn(model(xb), yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        optimiser.step()
        tot_acc += loss.item() * xb.size(0)

    # deterministic: kl = 0 , nll = tot
    return tot_acc / N, tot_acc / N, 0.0


@torch.no_grad()
def evaluate(model, loader, meta, svi=None, run_dir=cfg.ARTIFACT_DIR,
             split_name: str = "test", dump_latent: bool = False):
    """
    Evaluate on `loader` and write predictions_{split_name}.csv.

    - Always writes: true_* (physical), pred_* (physical).
    - For split_name=="test": also writes input features from meta["x_test_raw"].
    - For split_name=="val": if meta lacks x_val_raw/y_val_raw, reconstructs truth
      from loader.dataset.tensors (no feature columns written).
    - If dump_latent=True: also writes mu_lat_mean_* and std_*_lat_* plus std_tot_lat_*.
    """
    y_scaler = meta["y_scaler"]
    tf_info = meta["tf_info"]
    bayes = svi is not None
    model.eval()

    # --- accumulators (physical) ---
    mu_phys_ls, ale_phys_ls, epi_phys_ls = [], [], []

    # --- accumulators (latent t-space) ---
    mu_lat_ls, std_ale_lat_ls, std_epi_lat_ls = [], [], []

    for xb, _ in loader:
        xb = xb.to(cfg.DEVICE)

        # --- forward / MC sampling ---
        if bayes:
            pred = pyro.infer.Predictive(
                model,
                guide=model.guide,
                num_samples=cfg.BAYES_NUM_SAMPLES,
                return_sites=["_RETURN"],
            )(xb)
            outs = pred["_RETURN"].transpose(0, 1)  # (B,S,H)

            mus_s, vars_s = [], []
            for s in range(outs.shape[1]):
                mu_s, std_s = split_output(outs[:, s, :])
                mus_s.append(mu_s)
                if std_s is not None:
                    vars_s.append(std_s.pow(2))

            mus_s = torch.stack(mus_s, 0)          # (S,B,D)
            mu_m = mus_s.mean(0)                   # (B,D)
            std_e = mus_s.var(0, unbiased=False).sqrt()  # (B,D)
            std_a = torch.stack(vars_s, 0).mean(0).sqrt() if vars_s else None
        else:
            out = model(xb)
            mu_m, std_a = split_output(out)
            std_e = None

        # --- undo z-score to latent t-space ---
        mu_lat = (y_scaler.inverse_transform(mu_m.cpu().numpy())
                  if cfg.USE_ZSCORE_SCALING else mu_m.cpu().numpy())

        if dump_latent:
            mu_lat_ls.append(mu_lat)

        # latent aleatoric std (t-space)
        std_a_lat = None
        if std_a is not None:
            std_a_lat = (y_scaler.scale_ * std_a.cpu().numpy()
                         if cfg.USE_ZSCORE_SCALING else std_a.cpu().numpy())
            if dump_latent:
                std_ale_lat_ls.append(std_a_lat)

        # latent epistemic std (t-space)
        std_e_lat = None
        if std_e is not None:
            std_e_lat = (y_scaler.scale_ * std_e.cpu().numpy()
                         if cfg.USE_ZSCORE_SCALING else std_e.cpu().numpy())
            if dump_latent:
                std_epi_lat_ls.append(std_e_lat)

        # --- convert mean to physical ---
        mu_phys = inverse_phys_tf(mu_lat, tf_info)
        mu_phys_ls.append(mu_phys)

        # --- propagate stds to physical (delta method) ---
        if std_a_lat is not None:
            ale_phys_ls.append(propagate_std(std_a_lat, mu_lat, tf_info))
        if std_e_lat is not None:
            epi_phys_ls.append(propagate_std(std_e_lat, mu_lat, tf_info))

    # --- concatenate outputs ---
    mu_phys = np.concatenate(mu_phys_ls, 0)
    std_ale_phys = np.concatenate(ale_phys_ls, 0) if ale_phys_ls else None
    std_epi_phys = np.concatenate(epi_phys_ls, 0) if epi_phys_ls else None

    # --- get truths (physical) and (optionally) features ---
    true_cols = [f"true_{t}" for t in cfg.TARGETS]
    pred_cols = [f"pred_{t}" for t in cfg.TARGETS]

    X_all = None
    y_true_phys = None

    if split_name == "test":
        # full output with feature columns
        X_all = meta["x_test_raw"]
        y_true_phys = meta["y_test_raw"]

        in_cols = list(cfg.FEATURES) + [f"{t}_HS" for t in cfg.TARGETS]
        df = pd.DataFrame(
            np.hstack([X_all, y_true_phys, mu_phys]),
            columns=in_cols + true_cols + pred_cols
        )

    elif split_name == "val":
        # if val raw arrays exist, use them; otherwise reconstruct truth from loader tensors
        X_all = meta.get("x_val_raw", None)
        y_true_phys = meta.get("y_val_raw", None)

        if y_true_phys is None:
            # loader targets are the training targets (t_norm if z-scored)
            y_lat_norm = loader.dataset.tensors[1].detach().cpu().numpy()
            if cfg.USE_ZSCORE_SCALING:
                t_true = y_scaler.mean_ + y_scaler.scale_ * y_lat_norm
            else:
                t_true = y_lat_norm
            y_true_phys = inverse_phys_tf(t_true, tf_info)

        if X_all is None:
            # no features available -> write minimal CSV (sufficient for tau fitting)
            df = pd.DataFrame(
                np.hstack([y_true_phys, mu_phys]),
                columns=true_cols + pred_cols
            )
        else:
            in_cols = list(cfg.FEATURES) + [f"{t}_HS" for t in cfg.TARGETS]
            df = pd.DataFrame(
                np.hstack([X_all, y_true_phys, mu_phys]),
                columns=in_cols + true_cols + pred_cols
            )
    else:
        raise ValueError("split_name must be 'val' or 'test'")

    # --- add optional physical std columns (unchanged behaviour from old evaluate) ---
    if std_ale_phys is not None:
        df = pd.concat(
            [df, pd.DataFrame(std_ale_phys, columns=[f"std_ale_{t}" for t in cfg.TARGETS])],
            axis=1
        )
    if std_epi_phys is not None:
        df = pd.concat(
            [df, pd.DataFrame(std_epi_phys, columns=[f"std_epi_{t}" for t in cfg.TARGETS])],
            axis=1
        )

    # --- add latent columns for temperature tuning ---
    if dump_latent:
        mu_lat_all = np.concatenate(mu_lat_ls, 0)
        for j, t in enumerate(cfg.TARGETS):
            df[f"mu_lat_mean_{t}"] = mu_lat_all[:, j]

        if std_ale_lat_ls:
            std_ale_lat_all = np.concatenate(std_ale_lat_ls, 0)
            for j, t in enumerate(cfg.TARGETS):
                df[f"std_ale_lat_{t}"] = std_ale_lat_all[:, j]

        if std_epi_lat_ls:
            std_epi_lat_all = np.concatenate(std_epi_lat_ls, 0)
            for j, t in enumerate(cfg.TARGETS):
                df[f"std_epi_lat_{t}"] = std_epi_lat_all[:, j]

        # total latent std (what your tuner expects as std_tot_lat_*)
        for t in cfg.TARGETS:
            a = df[f"std_ale_lat_{t}"].to_numpy() if f"std_ale_lat_{t}" in df else None
            e = df[f"std_epi_lat_{t}"].to_numpy() if f"std_epi_lat_{t}" in df else None
            if a is not None and e is not None:
                df[f"std_tot_lat_{t}"] = np.sqrt(a**2 + e**2)
            elif a is not None:
                df[f"std_tot_lat_{t}"] = a
            elif e is not None:
                df[f"std_tot_lat_{t}"] = e
            else:
                # no uncertainty available in this mode
                df[f"std_tot_lat_{t}"] = np.nan

    preds_csv = Path(run_dir) / f"predictions_{split_name}.csv"
    df.to_csv(preds_csv, index=False)
    print(f"[INFO] saved predictions → {preds_csv}")

    return mu_phys, std_ale_phys, std_epi_phys

@torch.no_grad()
def loss_on_loader(model,
                   loader,
                   beta: float = 1.0,
                   loss_fn=None,
                   svi: SVI | None = None):
    """
    Returns (tot , nll , kl_beta) without changing any parameters.
    """
    model.eval()
    tot = nll = kl = 0.0
    N = len(loader.dataset)

    if svi is not None:
        tot_acc = 0.0
        nll_acc = 0.0
        kl_acc = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)

            # logging only
            nll_v, kl_raw_v = elbo_parts(model, model.guide, xb, yb)
            nll_acc += nll_v
            kl_acc += beta * kl_raw_v

            # objective
            tot_acc += svi.evaluate_loss(xb, yb)
            n_seen += xb.size(0)

        return tot_acc / n_seen, nll_acc / n_seen, kl_acc / n_seen

    # deterministic
    for xb, yb in loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        l = loss_fn(model(xb), yb)
        tot += l.item() * xb.size(0)

    return tot / N, tot / N, 0.0


def _is_better(new, best, min_delta):
    """Return True if `new` is better than `best` by at least `min_delta` (relative)."""
    return (best - new) / max(abs(best), 1e-12) > min_delta


# --- helper: turn non-serialisable objects into strings -------------
def _serialisable(x):
    """Turn non-serialisable objects (Path, device, numpy scalar) into strings/scalars."""
    if isinstance(x, (Path, torch.device)):
        return str(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x

def make_deterministic_guide(bnn):
    # assume guide has parameters "loc" and "scale_unconstrained"
    def guide(*args, **kwargs):
        for name, value in pyro.get_param_store().items():
            if name.endswith("scale_unconstrained"):
                pyro.param(name).data.fill_(-10.0)     # sigma ≈ 0
        return bnn.guide(*args, **kwargs)
    return guide

# ----------------------------------------------------------------------
def main():
    # 0)  run dir -----------------------------------------------------
    run_dir = cfg.ARTIFACT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"[INFO] artifacts → {run_dir}")

    # Save config used for this run
    cfg_dict = {k: _serialisable(getattr(cfg, k))
                for k in dir(cfg) if k.isupper()}
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"[INFO] saved config   → {run_dir / 'config_used.yaml'}")


    # 1)  data --------------------------------------------------------
    train_loader, val_loader, test_loader, meta = load_dataset(
        cfg.DATA_FILE, cfg.FEATURES, cfg.TARGETS,
        cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.DEVICE)

    model = HybridNet(next(iter(train_loader))[0].shape[1]).to(cfg.DEVICE)
    is_bnn = getattr(model, "has_bayes", False)

    if getattr(model, "has_bayes", False):
        with torch.no_grad():
            # produce a single dummy input to trigger guide initialisation
            dummy_x = next(iter(train_loader))[0][:1].to(cfg.DEVICE)
            model.guide(dummy_x)  # <-- instantiates guide parameters

    det_optim = None
    if is_bnn:  # BNN -> we will need a
        det_optim = optim.Adam(  # vanilla torch optimiser
            model.parameters(), lr=cfg.LR)  # for the µ-only phase

    det_loss = make_det_loss()
    optimiser = optim.Adam(model.parameters(), lr=cfg.LR) if not is_bnn else None
    svi = None

    # ------ early-stopping bookkeeping ------------------------------
    best_val = 99999999
    best_state = None
    # Default best_epoch to the DET_WARMUP epoch if applicable, else 0
    best_epoch = cfg.DET_WARMUP if cfg.DET_WARMUP > 0 else 0

    tr_tot_hist, tr_nll_hist, tr_kl_hist = [], [], []
    va_tot_hist, va_nll_hist, va_kl_hist = [], [], []

    if is_bnn:
        def lr_cfg(name, _):
            lr = cfg.LR * cfg.LR_SCALE_FACTOR if "scale" in name else cfg.LR
            d = {"lr": lr}
            if cfg.GRAD_CLIP_NORM and cfg.GRAD_CLIP_NORM > 0:
                d["clip_norm"] = cfg.GRAD_CLIP_NORM
            return d

        adam = ClippedAdam(lr_cfg)  # ONE optimiser object, reused


    ALPHA0 = 0.05  # small but > 0


    def set_scale_requires_grad(require_grad: bool, init_val: float = None):
        """
        enable/disable gradients of all variational log-scale parameters.
        If init_val is given, also set their value.
        """
        for name, p in pyro.get_param_store().items():
            if name.endswith("scale_unconstrained"):
                p.requires_grad = require_grad
                if init_val is not None:
                    p.data.fill_(init_val)

    if is_bnn and cfg.DET_WARMUP > 0:
        set_scale_requires_grad(False, init_val=-10.0)  # σ ≈ 0

    # ------------------------------------------------------------------

    def alpha_beta(epoch: int):
        # no beta scheduling; standard ELBO always
        return 1.0, 1.0


    if is_bnn and cfg.DET_WARMUP > 0:
        for name, p in pyro.get_param_store().items():
            if name.endswith("scale_unconstrained"):
                p.requires_grad = False
                p.data.fill_(-10.0)  # σw ≈ 0
        model.use_predicted_std = False  # ignore σ̂ during DET phase

    frozen_flag = False

    for epoch in range(1, cfg.EPOCHS + 1):

        if is_bnn:
            model.alpha = 1.0  # no likelihood scaling

            # standard ELBO, no beta tricks
            beta = min(1.0, epoch / max(1, cfg.WARMUP))
            svi = make_svi(model, epoch, warmup=cfg.WARMUP)



            # unfreeze variational scales after deterministic warmup
            if epoch > cfg.DET_WARMUP and not any(
                    p.requires_grad for n, p in pyro.get_param_store().items()
                    if n.endswith("scale_unconstrained")):
                for name, p in pyro.get_param_store().items():
                    if name.endswith("scale_unconstrained"):
                        p.requires_grad = True
                        p.data.fill_(cfg.LOG_VAR_MIN + 0.1)
                model.use_predicted_std = True
                if not frozen_flag:
                    frozen_flag = True
                    print(f"[INFO] Variance parameters unfrozen at epoch {epoch}")

            tr_tot, tr_nll, tr_kl = train_one_epoch(model, train_loader, beta=beta, svi=svi)
            va_tot, va_nll, va_kl = loss_on_loader(model, val_loader, beta=beta, svi=svi)

        else:
            tr_tot, tr_nll, tr_kl = train_one_epoch(model, train_loader,
                                                    optimiser=optimiser, loss_fn=det_loss)
            va_tot, va_nll, va_kl = loss_on_loader(model, val_loader, loss_fn=det_loss)
            svi = None

        # logging...
        tr_tot_hist.append(tr_tot)
        tr_nll_hist.append(tr_nll)
        tr_kl_hist.append(tr_kl)
        va_tot_hist.append(va_tot)
        va_nll_hist.append(va_nll)
        va_kl_hist.append(va_kl)

        print(f"epoch {epoch:3d}/{cfg.EPOCHS} | "
              f"train tot {tr_tot:.4e}  nll {tr_nll:.4e}  kl {tr_kl:.4e} | "
              f"val   tot {va_tot:.4e}  nll {va_nll:.4e}  kl {va_kl:.4e} | beta {beta:.3f}")

        # ---- early–stopping logic ---------------------------------
        if cfg.EARLY_STOP_PATIENCE > 0:
            start_es = early_stop_start_epoch()

            if epoch < start_es:
                # do not update best_epoch / best_val and do not count patience yet
                pass
            else:
                # initialise best at the moment ES becomes active
                if best_epoch < start_es:
                    best_val = va_tot
                    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch

                if _is_better(va_tot, best_val, cfg.EARLY_STOP_MIN_DELTA):
                    best_val = va_tot
                    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                elif epoch - best_epoch >= cfg.EARLY_STOP_PATIENCE:
                    print(f"[INFO] Early stopping at epoch {epoch} "
                          f"(no val improvement for {cfg.EARLY_STOP_PATIENCE} epochs; "
                          f"counting started at epoch {start_es}).")
                    break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
        print(f"[INFO] Loaded best model from epoch {best_epoch} (val loss {best_val:.4e})")

    # 3)  evaluation --------------------------------------------------
    _ = evaluate(model, val_loader, meta, svi, run_dir, split_name="val", dump_latent=True)
    mu, std_ale, std_epi = evaluate(model, test_loader, meta, svi, run_dir, split_name="test", dump_latent=True)

    # 4)  store & plot -----------------------------------------------
    epochs_done = len(tr_tot_hist)
    df = pd.DataFrame({
        "epoch": np.arange(1, len(tr_tot_hist) + 1),
        "train_tot": tr_tot_hist,
        "train_nll": tr_nll_hist,
        "train_kl": tr_kl_hist,
        "val_tot": va_tot_hist,
        "val_nll": va_nll_hist,
        "val_kl": va_kl_hist,
        # legacy columns for old scripts
        "train": tr_tot_hist,
        "val": va_tot_hist,
    })
    df.to_csv(run_dir / "loss.csv", index=False)

    plot_loss(
        {
            "Train (tot)": tr_tot_hist,
            "Val   (tot)": va_tot_hist,
            "Train (NLL)": tr_nll_hist,
            "Val   (NLL)": va_nll_hist,
            "Train (KL)": tr_kl_hist,
            "Val   (KL)": va_kl_hist,
        },
        run_dir,
        fname="loss_components.png"
    )
    plot_results(meta["y_test_raw"], mu, std_ale, std_epi,
                 meta["low_res_test_raw"], cfg.TARGETS,
                 run_dir / "pred_plot.png")

    torch.save(model.state_dict(), run_dir / "model.pth")
    torch.save(meta, run_dir / "data_meta.pt")
    if svi is not None:
        pyro.get_param_store().save(run_dir / "pyro_params.pt")

    print("[INFO] training complete")

    # 5)  temperature calibration (fit taus on val, apply to test) -----
    from training.temp_tune import run_temp_tune
    print("\n[INFO] running temperature calibration …")
    run_temp_tune(run_dir)

    return run_dir


if __name__ == "__main__":
    main()
