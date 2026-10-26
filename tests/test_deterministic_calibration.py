"""Regression tests for deterministic calibration and absent uncertainty."""
import unittest
import numpy as np
import pandas as pd

try:
    from training.temp_tune import load_latent_components
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class DeterministicCalibrationTests(unittest.TestCase):
    def test_deterministic_diagonal_has_zero_epistemic_component(self):
        df = pd.DataFrame({'mu_lat_mean_Qext': [0.0, 1.0],
                           'std_ale_lat_Qext': [0.2, 0.3]})
        mode, mu, epi, ale = load_latent_components(df, ['Qext'])
        self.assertEqual(mode, 'gaussian')
        np.testing.assert_array_equal(epi, np.zeros((2, 1)))
        np.testing.assert_allclose(ale[:, 0], [0.2, 0.3])

    def test_bayesian_none_has_zero_aleatoric_component(self):
        df = pd.DataFrame({'mu_lat_mean_Qext': [0.0, 1.0],
                           'std_epi_lat_Qext': [0.2, 0.3]})
        mode, mu, epi, ale = load_latent_components(df, ['Qext'])
        self.assertEqual(mode, 'gaussian')
        np.testing.assert_array_equal(ale, np.zeros((2, 1)))
        np.testing.assert_allclose(epi[:, 0], [0.2, 0.3])

    def test_no_uncertainty_reports_actionable_error(self):
        df = pd.DataFrame({'mu_lat_mean_Qext': [0.0, 1.0]})
        with self.assertRaisesRegex(ValueError, 'requires a latent uncertainty'):
            load_latent_components(df, ['Qext'])
