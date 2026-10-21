"""Edge-case contracts for inverse-frequency balancing."""
import unittest
from unittest.mock import patch
import numpy as np
try:
    from training import data_utils
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class BalancedSamplingEdgeCases(unittest.TestCase):
    def weights(self, a):
        with patch.object(data_utils.cfg, 'BALANCE_TARGETS', ['Qext']), \
             patch.object(data_utils.cfg, 'BALANCE_STRATEGY', 'max'), \
             patch.object(data_utils.cfg, 'N_BINS', 4):
            return data_utils._balanced_sample_weights(a, ['Qext'])

    def test_constant_column_produces_uniform_weights(self):
        np.testing.assert_allclose(self.weights(np.ones((5, 1))), np.ones(5))

    def test_empty_training_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.weights(np.empty((0, 1)))

    def test_nonfinite_targets_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'finite'):
                self.weights(np.array([[0.], [invalid]]))

    def test_invalid_target_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '2-D'):
            self.weights(np.array([1., 2.]))
