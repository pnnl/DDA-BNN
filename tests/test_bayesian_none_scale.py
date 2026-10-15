"""The mean-only Bayesian likelihood uses unit noise in current target space."""
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
class FixedObservationScaleTests(unittest.TestCase):
    def test_fixed_scale_independent_of_zscore_setting(self):
        for module in (training_model, release_model):
            for scaled in (False, True):
                with self.subTest(module=module.__name__, scaled=scaled), \
                     patch.object(module.cfg, 'TARGETS', ['A', 'B']), \
                     patch.object(module.cfg, 'UNCERTAINTY_MODE', 'none'), \
                     patch.object(module.cfg, 'USE_ZSCORE_SCALING', scaled), \
                     patch.object(module.cfg, 'LAYER_SPEC', []):
                    model = module.HybridNet(3)
                    x, y = torch.zeros(4, 3), torch.zeros(4, 2)
                    obs = pyro.poutine.trace(model).get_trace(x, y).nodes['obs']['fn']
                    torch.testing.assert_close(obs.base_dist.scale, torch.ones_like(y))
                    self.assertEqual(obs.event_shape, (2,))
