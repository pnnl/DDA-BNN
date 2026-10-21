"""Balancing must upweight rare bins for all three strategies."""
import unittest
from unittest.mock import patch
import numpy as np

try:
    from training import data_utils
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class BalanceWeightTests(unittest.TestCase):
    def test_rare_category_upweighted_by_each_strategy(self):
        vals = np.array([0.0]*90 + [1.0]*10).reshape(-1, 1)
        for strategy in ('prod', 'mean', 'max'):
            with self.subTest(strategy=strategy), \
                 patch.object(data_utils.cfg, 'BALANCE_TARGETS', ['Qext']), \
                 patch.object(data_utils.cfg, 'BALANCE_STRATEGY', strategy), \
                 patch.object(data_utils.cfg, 'N_BINS', 2):
                weights = data_utils._balanced_sample_weights(vals, ['Qext'])
                self.assertGreater(weights[90], weights[0])
                self.assertAlmostEqual(weights.mean(), 1.0)

    def test_max_bin_boundary_belongs_to_last_bin(self):
        # Unequal values within each half-bin make an extra bin for the
        # maximum distinguishable from the intended rightmost bin. The
        # pre-fix np.digitize() implementation fails this assertion.
        vals = np.array([[0.0], [0.1], [0.9], [1.0]])
        with patch.object(data_utils.cfg, 'BALANCE_TARGETS', ['Qext']), \
             patch.object(data_utils.cfg, 'BALANCE_STRATEGY', 'prod'), \
             patch.object(data_utils.cfg, 'N_BINS', 2):
            weights = data_utils._balanced_sample_weights(vals, ['Qext'])
            np.testing.assert_allclose(weights, np.ones(4))
