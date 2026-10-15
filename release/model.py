import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal

from . import config as cfg


def make_activation(name):
    """Return a torch.nn activation module by name, or Identity if None."""
    if name is None:
        return nn.Identity()
    if not hasattr(nn, name):
        raise ValueError(f"activation '{name}' not in torch.nn")
    return getattr(nn, name)()


def head_size(d, mode):
    """Compute output head size based on uncertainty mode."""
    mode = mode.lower()
    if mode == "none":
        return d
    if mode == "diag":
        return 2 * d
    if mode == "full":
        return d + d * (d + 1) // 2
    raise ValueError("uncertainty mode must be 'none' | 'diag' | 'full'")


def build_cov_from_params(sigma, rho_raw):
    """
    Build covariance matrix from standard deviations and correlation parameters.

    Args:
        sigma: (B, d) positive standard deviations
        rho_raw: (B, d*(d-1)//2) unconstrained correlation parameters

    Returns:
        Full covariance matrix Σ = D R D
    """
    B, d = sigma.shape
    R = torch.eye(d, device=sigma.device).repeat(B, 1, 1)
    idx = 0
    for i in range(1, d):
        for j in range(i):
            rho = cfg.RHO_MAX * torch.tanh(rho_raw[:, idx])
            R[:, i, j] = rho
            R[:, j, i] = rho
            idx += 1
    D = torch.diag_embed(sigma)
    cov = D @ R @ D
    return cov


class HybridNet(PyroModule):
    """
    Hybrid MLP with deterministic or Bayesian layers as specified in cfg.LAYER_SPEC.

    When alpha < 1, the data-likelihood contribution to the ELBO is down-weighted
    (used for deterministic and NLL warm-up).
    """

    def __init__(self, n_in: int):
        super().__init__()
        d = len(cfg.TARGETS)
        h_out = head_size(d, cfg.UNCERTAINTY_MODE)
        prev = n_in
        self.has_bayes = False
        self.alpha = 1.0
        self.use_predicted_std = True

        backbone = PyroModule[nn.Sequential]()
        idx = 0
        for spec in cfg.LAYER_SPEC:
            units = spec["units"]
            ltype = spec["type"].lower()
            act = make_activation(spec.get("act"))

            if ltype == "det":
                lin = nn.Linear(prev, units)
            elif ltype == "bnn":
                lin = self._make_bayes_linear(prev, units)
            else:
                raise ValueError("layer type must be 'det' or 'bnn'")

            backbone.add_module(str(idx), lin)
            idx += 1
            if not isinstance(act, nn.Identity):
                backbone.add_module(str(idx), act)
                idx += 1
            prev = units
        self.backbone = backbone

        head_typ = "bnn" if (cfg.LAYER_SPEC and cfg.LAYER_SPEC[-1]["type"].lower() == "bnn") else "det"
        self.head = (
            self._make_bayes_linear(prev, h_out) if head_typ == "bnn"
            else nn.Linear(prev, h_out)
        )

        if self.has_bayes:
            self.guide = AutoDiagonalNormal(self)

        self.apply(self._init_weights)

    def forward(self, x, y=None):
        """
        Forward pass. If y is provided, contributes scaled likelihood to ELBO.
        Returns raw head output.
        """
        out = self.head(self.backbone(x))
        if y is None:
            return out

        d = len(cfg.TARGETS)
        eps = 1e-6
        mode = cfg.UNCERTAINTY_MODE.lower()

        if mode == "none":
            # No learned aleatoric head: use a fixed unit observation scale
            # in training's target space. Epistemic uncertainty is still learned.
            dist_y = dist.Normal(out, torch.ones_like(out)).to_event(1)
        elif mode == "diag":
            mu, raw = out[:, :d], out[:, d:]
            log_var = cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)
            var = torch.exp(log_var) + eps
            scale = torch.sqrt(var)
            dist_y = dist.Normal(mu, scale).to_event(1)

        elif mode == "full":
            mu, params = out[:, :d], out[:, d:]
            raw_log_var = params[:, :d]
            rho_raw = params[:, d:]
            sigma = torch.exp(0.5 * (cfg.LOG_VAR_MIN +
                                     (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN)
                                     * torch.sigmoid(raw_log_var))) + eps
            cov = build_cov_from_params(sigma, rho_raw)
            dist_y = dist.MultivariateNormal(mu, covariance_matrix=cov)

        else:
            raise RuntimeError(f"bad cfg.UNCERTAINTY_MODE '{cfg.UNCERTAINTY_MODE}'")

        B = x.size(0)
        with pyro.plate("data", B):
            with pyro.poutine.scale(scale=getattr(self, "alpha", 1.0)):
                pyro.sample("obs", dist_y, obs=y)

        return out

    @staticmethod
    def _init_weights(m):
        """Kaiming-normal init for deterministic Linear layers."""
        if isinstance(m, nn.Linear) and not isinstance(m, PyroModule):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _make_bayes_linear(self, in_features, out_features):
        """Create Bayesian Linear layer with Normal(0, cfg.PRIOR_STD) prior."""
        lin = PyroModule[nn.Linear](in_features, out_features)

        loc_w = torch.zeros(out_features, in_features, device=cfg.DEVICE)
        scale_w = cfg.PRIOR_STD * torch.ones_like(loc_w)
        loc_b = torch.zeros(out_features, device=cfg.DEVICE)
        scale_b = cfg.PRIOR_STD * torch.ones_like(loc_b)

        lin.weight = PyroSample(dist.Normal(loc_w, scale_w).to_event(2))
        lin.bias = PyroSample(dist.Normal(loc_b, scale_b).to_event(1))

        self.has_bayes = True
        return lin


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean-squared-error loss."""
    return nn.functional.mse_loss(pred, target)


def gaussian_nll_diag(output: torch.Tensor, target: torch.Tensor, d: int):
    """Negative log-likelihood for independent Gaussians with diagonal covariance."""
    mu, raw = output[:, :d], output[:, d:]
    log_var = cfg.LOG_VAR_MIN + (cfg.LOG_VAR_MAX - cfg.LOG_VAR_MIN) * torch.sigmoid(raw)
    var = torch.exp(log_var) + 1e-6
    inv_var = var.reciprocal()
    nll = 0.5 * (inv_var * (target - mu).pow(2) + log_var)
    return nll.sum(dim=-1).mean()


def gaussian_nll_full(output: torch.Tensor, target: torch.Tensor, d: int):
    """Negative log-likelihood for full covariance Gaussian with packed Cholesky factor."""
    B = output.size(0)
    mu, pack = output[:, :d], output[:, d:]

    L = torch.zeros(B, d, d, device=output.device)
    k = 0
    for i in range(d):
        for j in range(i + 1):
            v = pack[:, k]
            L[:, i, j] = torch.nn.functional.softplus(v) + 1e-3 if i == j else v
            k += 1

    diff = (target - mu).unsqueeze(-1)
    inv = torch.cholesky_solve(diff, L)
    maha = (diff.transpose(-1, -2) @ inv).squeeze(-1).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
    return 0.5 * (maha + logdet).mean()


__all__ = [
    "HybridNet",
    "mse_loss",
    "gaussian_nll_diag",
    "gaussian_nll_full",
    "build_cov_from_params",
]
