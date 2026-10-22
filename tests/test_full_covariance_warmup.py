"""Check the intended full-covariance warmup likelihood contract."""
import unittest
from unittest.mock import patch
try:
    import torch
    import pyro
    from training import model as training_model
    from release import model as release_model
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class FullCovarianceWarmupTests(unittest.TestCase):
    def test_fixed_unit_noise_then_learned_multivariate_noise(self):
        for module in (training_model, release_model):
            with self.subTest(module=module.__name__), \
                 patch.object(module.cfg, 'TARGETS', ['A', 'B']), \
                 patch.object(module.cfg, 'UNCERTAINTY_MODE', 'full'), \
                 patch.object(module.cfg, 'LAYER_SPEC', []):
                net = module.HybridNet(3)
                x, y = torch.ones(5, 3), torch.ones(5, 2)
                net.use_predicted_std = False
                obs = pyro.poutine.trace(net).get_trace(x, y).nodes['obs']['fn']
                torch.testing.assert_close(obs.base_dist.scale, torch.ones_like(y))
                net.use_predicted_std = True
                obs = pyro.poutine.trace(net).get_trace(x, y).nodes['obs']['fn']
                self.assertIsInstance(obs, torch.distributions.MultivariateNormal)
                self.assertTrue(bool((torch.linalg.eigvalsh(obs.covariance_matrix) > 0).all()))
