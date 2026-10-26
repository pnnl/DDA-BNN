"""Characterize current production likelihood and warmup weighting, not approval."""
import pytest
import random
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from torch.distributions import kl_divergence

from training import config as cfg
from training.model import HybridNet
from training.train import make_svi


@pytest.fixture(autouse=True)
def isolated_pyro_state():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), pyro.get_param_store().scope():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_production_observation_likelihood_is_an_unscaled_batch_sum(monkeypatch):
    monkeypatch.setattr(cfg, "TARGETS", ["y"])
    monkeypatch.setattr(cfg, "UNCERTAINTY_MODE", "diag")
    monkeypatch.setattr(cfg, "LAYER_SPEC", [{"type": "det", "units": 2}])
    net = HybridNet(n_in=1)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
    sums = []
    for batch in (2, 8):
        trace = poutine.trace(net).get_trace(torch.zeros(batch, 1), torch.ones(batch, 1))
        trace.compute_log_prob()
        observation = trace.nodes["obs"]
        assert observation["scale"] == 1.0
        sums.append(float(observation["log_prob_sum"].detach()))
    assert sums[1] == pytest.approx(4 * sums[0])


@pytest.mark.parametrize("epoch", [1, 4])
def test_svi_factory_warmup_changes_kl_without_dataset_scaling(epoch):
    class NormalModel:
        def __call__(self, x, y=None):
            latent = pyro.sample("latent", dist.Normal(0.0, 1.0))
            with pyro.plate("data", len(x)):
                pyro.sample("obs", dist.Normal(latent, 1.0), obs=y)

        def guide(self, x, y=None):
            pyro.sample("latent", dist.Normal(torch.tensor(0.4), torch.tensor(0.7)))

    model = NormalModel()
    svi = make_svi(model, epoch=epoch, warmup=4)
    x, y = torch.zeros(3), torch.ones(3)
    seed = 37
    pyro.set_rng_seed(seed)
    guide_trace = poutine.trace(svi.guide).get_trace(x, y)
    model_trace = poutine.trace(poutine.replay(svi.model, trace=guide_trace)).get_trace(x, y)
    model_trace.compute_log_prob()
    nll = -model_trace.nodes["obs"]["log_prob_sum"]
    kl = kl_divergence(dist.Normal(torch.tensor(0.4), torch.tensor(0.7)), dist.Normal(0.0, 1.0))
    beta = epoch / 4
    pyro.set_rng_seed(seed)
    actual = svi.loss(svi.model, svi.guide, x, y)
    assert actual == pytest.approx(float(nll + beta * kl), rel=1e-6)
