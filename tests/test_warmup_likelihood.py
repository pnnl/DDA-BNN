"""Warmup disables learned observation variance, not only a flag."""
import unittest
from unittest.mock import patch

try:
    import torch
    import pyro
    from training import model as model_mod
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class WarmupLikelihoodTests(unittest.TestCase):
    def test_uncertainty_head_gets_no_gradient_during_warmup(self):
        with patch.object(model_mod.cfg, 'TARGETS', ['A']), \
             patch.object(model_mod.cfg, 'UNCERTAINTY_MODE', 'diag'), \
             patch.object(model_mod.cfg, 'LAYER_SPEC', []):
            model = model_mod.HybridNet(2)
            x = torch.ones(5, 2)
            y = torch.ones(5, 1)
            model.use_predicted_std = False
            loss = -pyro.poutine.trace(model).get_trace(x, y).log_prob_sum()
            loss.backward()
            torch.testing.assert_close(model.head.weight.grad[1:],
                                       torch.zeros_like(model.head.weight.grad[1:]))
            model.head.weight.grad.zero_()
            model.use_predicted_std = True
            loss = -pyro.poutine.trace(model).get_trace(x, y).log_prob_sum()
            loss.backward()
            self.assertGreater(float(model.head.weight.grad[1:].abs().sum()), 0.0)
