"""Bayesian mean-only heads still require a valid observation likelihood."""
import unittest
from unittest.mock import patch

try:
    import torch
    import pyro
    from training import model as train_model
    from release import model as release_model
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class BayesianNoAleatoricTests(unittest.TestCase):
    def test_observation_site_exists_with_none_mode(self):
        for m in (train_model, release_model):
            with patch.object(m.cfg, 'TARGETS', ['A', 'B']), \
                 patch.object(m.cfg, 'UNCERTAINTY_MODE', 'none'), \
                 patch.object(m.cfg, 'LAYER_SPEC', [{'units': 4, 'type': 'bnn', 'act': 'ReLU'}]), \
                 patch.object(m.cfg, 'DEVICE', torch.device('cpu')):
                pyro.clear_param_store()
                net = m.HybridNet(3)
                x, y = torch.zeros(5, 3), torch.ones(5, 2)
                trace = pyro.poutine.trace(net).get_trace(x, y)
                self.assertEqual(trace.nodes['obs']['fn'].event_shape, (2,))
                self.assertTrue(bool(torch.isfinite(trace.log_prob_sum())))
        pyro.clear_param_store()
