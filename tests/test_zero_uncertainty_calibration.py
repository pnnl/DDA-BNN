"""Reject unidentifiable temperatures before replacing calibrated artifacts."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from training import temp_tune as tune


class ZeroUncertaintyCalibrationTests(unittest.TestCase):
    def test_all_zero_uncertainty_is_unidentifiable_even_for_exact_predictions(self):
        for truth in (np.ones((4, 1)), np.zeros((4, 1))):
            with self.subTest(exact=bool((truth == 0).all())), self.assertRaisesRegex(ValueError, 'zero uncertainty'):
                tune.fit_tau_per_target(np.zeros((4, 1)), np.zeros((4, 1)), truth)

    def test_zero_spread_with_nonzero_error_cannot_be_repaired_by_a_temperature(self):
        with self.assertRaisesRegex(ValueError, 'Nonzero residual'):
            tune.fit_tau_per_target(np.zeros((2, 1)), np.array([[0.], [1.]]), np.ones((2, 1)))

    def test_zero_spread_failure_preserves_previous_files(self):
        with TemporaryDirectory() as tmp, patch.object(tune.cfg, 'TARGETS', ['A']):
            run = Path(tmp); (run / 'config_used.yaml').write_text('{}\n')
            torch.save({'tf_info': {'A': 'none'}}, run / 'data_meta.pt')
            pd.DataFrame({'true_A': [1., 2.], 'mu_lat_mean_A': [0., 0.],
                          'std_tot_lat_A': [0., 0.]}).to_csv(run / 'predictions_val.csv', index=False)
            pd.DataFrame({'true_A': [1., 2.], 'mu_lat_mean_A': [0., 0.],
                          'std_ale_lat_A': [1., 1.]}).to_csv(run / 'predictions_test.csv', index=False)
            saved = {'taus.json': '{"A": 2.0}\n', 'predictions_test_cal.csv': 'old results\n'}
            for name, text in saved.items(): (run / name).write_text(text)
            with self.assertRaisesRegex(ValueError, 'zero uncertainty'):
                tune.run_temp_tune(run, K=3, L=3)
            for name, text in saved.items(): self.assertEqual((run / name).read_text(), text)

    def test_aleatoric_only_fit_apply_keeps_epistemic_output_exactly_zero(self):
        with TemporaryDirectory() as tmp, patch.object(tune.cfg, 'TARGETS', ['A']):
            run = Path(tmp); (run / 'config_used.yaml').write_text('{}\n')
            torch.save({'tf_info': {'A': 'none'}}, run / 'data_meta.pt')
            val = pd.DataFrame({'true_A': [0., 1., 2., 3.], 'mu_lat_mean_A': [.1, .8, 2.2, 2.9],
                                'std_tot_lat_A': [.5, .5, .5, .5]})
            val.to_csv(run / 'predictions_val.csv', index=False)
            app = val.drop(columns='std_tot_lat_A').assign(std_ale_lat_A=.5)
            app.to_csv(run / 'predictions_test.csv', index=False)
            tune.run_temp_tune(run, K=20, L=10)
            out = pd.read_csv(run / 'predictions_test_cal.csv')
            np.testing.assert_array_equal(out.std_epi_A, np.zeros(4))
            np.testing.assert_allclose(out.std_tot_A, out.std_ale_A)
