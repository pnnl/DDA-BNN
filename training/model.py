"""HybridNet model definition and associated loss functions.

Defines ``HybridNet``, a Pyro-compatible MLP whose layers can be
deterministic or Bayesian (per ``cfg.LAYER_SPEC``).  Supports diagonal
and full-covariance aleatoric uncertainty heads.  Also provides
standalone loss helpers (``mse_loss``, ``gaussian_nll_diag``,
``gaussian_nll_full``) for non-Bayesian training modes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal

import training.config as cfg


# ---------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------
def make_activation(name):
    if name is None:
        return nn.Identity()
    if not hasattr(nn, name):
        raise ValueError(f"activation '{name}' not in torch.nn")
    return getattr(nn, name)()

def head_size(d, mode):
    mode = mode.lower()
    if mode == "none":
        return d
    if mode == "diag":
        return 2 * d                   # μ + per-dim log σ²
    if mode == "full":
        return d + d * (d + 1) // 2    # μ + packed lower-tri
    raise ValueError("uncertainty mode must be 'none' | 'diag' | 'full'")


# ---------------- helper to build correlation matrix ---------------
def build_cov_from_params(sigma, rho_raw):
    """Build a positive-definite covariance using a lower-triangular factor.

    sigma gives positive Cholesky diagonal entries; bounded off-diagonals
    are scaled by their row's sigma. RHO_MAX is a factor bound, not a
    bound on pairwise Pearson correlation. Specifically, for zero-indexed
    target i, marginal variance is sigma[i]**2 * (1 + sum(r[i,j]**2,
    j < i)); LOG_VAR_MIN/MAX bound the Cholesky diagonal variance, not
    every marginal variance. The marginal scale depends on TARGETS order.
    """
    _, d = sigma.shape
    L = torch.diag_embed(sigma)
    idx = 0
    for i in range(1, d):
        for j in range(i):
            L[:, i, j] = sigma[:, i] * cfg.RHO_MAX * torch.tanh(rho_raw[:, idx])
            idx += 1
    return L @ L.transpose(-1, -2)


# ---------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------
class HybridNet(PyroModule):
    """
    Hybrid MLP whose layers can be deterministic ("det") or Bayesian
    ("bnn") according to cfg.LAYER_SPEC.

    When self.alpha < 1 the contribution of the data-likelihood to the
    ELBO is down-weighted (used for deterministic and NLL warm-up).
    """
    def __init__(self, n_in: int):
        super().__init__()
        d      = len(cfg.TARGETS)
        h_out  = head_size(d, cfg.UNCERTAINTY_MODE)
        prev   = n_in
        self.has_bayes = False
        self.alpha = 1.0          # <<< likelihood scaling factor
        self.use_predicted_std = True

        # --------------- build backbone --------------------------------
        backbone = PyroModule[nn.Sequential]()
        idx = 0
        for spec in cfg.LAYER_SPEC:
            units = spec["units"]
            ltype = spec["type"].lower()
            act   = make_activation(spec.get("act"))
            if ltype == "det":
                lin = nn.Linear(prev, units)
            elif ltype == "bnn":
                lin = self._make_bayes_linear(prev, units)
            else:
                raise ValueError("layer type must be 'det' or 'bnn'")
            backbone.add_module(str(idx), lin); idx += 1
            if not isinstance(act, nn.Identity):
                backbone.add_module(str(idx), act); idx += 1
            prev = units
        self.backbone = backbone

        # --------------- head -----------------------------------------
        head_typ = "bnn" if (cfg.LAYER_SPEC and cfg.LAYER_SPEC[-1]["type"].lower() == "bnn") else "det"
        self.head = (
            self._make_bayes_linear(prev, h_out) if head_typ == "bnn"
            else nn.Linear(prev, h_out)
        )

        # --------------- auto-guide if Bayesian -----------------------
        if self.has_bayes:
            self.guide = AutoDiagonalNormal(self)

        # --------------- init deterministic weights -------------------
        self.apply(self._init_weights)

    # -----------------------------------------------------------------
    def forward(self, x, y=None):
        """
        If y is provided the method contributes the (scaled) likelihood
        term to the ELBO and returns the raw head output.  If y is None
        it just returns the raw output (for prediction).
        """
        out = self.head(self.backbone(x))
        if y is None:
            return out

        d = len(cfg.TARGETS)
        eps = 1e-6
        mode = cfg.UNCERTAINTY_MODE.lower()

        if mode == "diag":
            mu, raw = out[:, :d], out[:, d:]

            # Map raw -> bounded log-variance, then to variance/std for the Normal
            log_var = cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)
            var = torch.exp(log_var) + eps
            scale = torch.sqrt(var)
            dist_y = dist.Normal(mu, scale).to_event(1)

        elif mode == "full":
            # TODO - not tested yet
            mu, params    = out[:, :d], out[:, d:]
            raw_log_var   = params[:, :d]
            rho_raw       = params[:, d:]
            sigma         = torch.sqrt(torch.exp(cfg.LOG_VAR_MIN +
                                              (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN)
                                              * torch.sigmoid(raw_log_var)) + eps)
            cov           = build_cov_from_params(sigma, rho_raw)
            dist_y        = dist.MultivariateNormal(mu, covariance_matrix=cov)

        else:
            raise RuntimeError(f"bad cfg.UNCERTAINTY_MODE '{cfg.UNCERTAINTY_MODE}'")

        # ---------------- scaled likelihood contribution -------------
        B = x.size(0)
        with pyro.plate("data", B):
            with pyro.poutine.scale(scale=getattr(self, "alpha", 1.0)):
                pyro.sample("obs", dist_y, obs=y)

        return out

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        """Kaiming-normal init for deterministic Linear layers."""
        if isinstance(m, nn.Linear) and not isinstance(m, PyroModule):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _make_bayes_linear(self, in_features, out_features):
        """Return Bayesian Linear layer with Normal(0, cfg.PRIOR_STD) prior."""
        lin = PyroModule[nn.Linear](in_features, out_features)

        loc_w   = torch.zeros(out_features, in_features, device=cfg.DEVICE)
        scale_w = cfg.PRIOR_STD * torch.ones_like(loc_w)
        loc_b   = torch.zeros(out_features, device=cfg.DEVICE)
        scale_b = cfg.PRIOR_STD * torch.ones_like(loc_b)

        lin.weight = PyroSample(dist.Normal(loc_w, scale_w).to_event(2))
        lin.bias   = PyroSample(dist.Normal(loc_b, scale_b).to_event(1))

        self.has_bayes = True
        return lin

# ----------------------------------------------------------------------
# basic deterministic losses – required when the network is trained
# in non-Bayesian ("det") mode or when cfg.UNCERTAINTY_MODE == "none/diag/full"
# ----------------------------------------------------------------------
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error (per-sample mean)."""
    return nn.functional.mse_loss(pred, target)


def gaussian_nll_diag(output: torch.Tensor, target: torch.Tensor, d: int):
    """
    Negative log-likelihood for independent Gaussians
    (μ + per-dimension variance in `output`).
    """
    mu, raw = output[:, :d], output[:, d:]
    # same smooth mapping as in HybridNet.forward
    log_var = cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)
    var = torch.exp(log_var) + 1e-6
    inv_var = var.reciprocal()
    # Use the stabilized variance in both terms (Gaussian constant omitted).
    nll = 0.5 * (inv_var * (target - mu).pow(2) + var.log())
    # sum over target dims to match Pyro's to_event(1), then mean over batch
    return nll.sum(dim=-1).mean()


def gaussian_nll_full(output: torch.Tensor, target: torch.Tensor, d: int):
    """NLL using exactly the covariance parameterization of HybridNet.forward."""
    mu, params = output[:, :d], output[:, d:]
    raw_log_var, rho_raw = params[:, :d], params[:, d:]
    sigma = torch.sqrt(torch.exp(cfg.LOG_VAR_MIN +
                             (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) *
                             torch.sigmoid(raw_log_var)) + 1e-6)
    cov = build_cov_from_params(sigma, rho_raw)
    return -torch.distributions.MultivariateNormal(mu, covariance_matrix=cov).log_prob(target).mean()

__all__ = [
    "HybridNet",
    "mse_loss",
    "gaussian_nll_diag",
    "gaussian_nll_full",
    "build_cov_from_params",
]
