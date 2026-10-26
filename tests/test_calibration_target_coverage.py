"""Calibration must have an uncertainty estimate for every target."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from training import temp_tune


class CalibrationTargetCoverageTests(unittest.TestCase):
    def test_missing_uncertainty_for_one_target_is_rejected(self):
        frame = pd.DataFrame({
            'mu_lat_mean_A': [0.0, 1.0],
            'mu_lat_mean_B': [0.0, 1.0],
            'std_ale_lat_A': [0.2, 0.3],
        })
        with self.assertRaisesRegex(ValueError, 'latent uncertainty.*B'):
            temp_tune.load_latent_components(frame, ['A', 'B'])

    def test_targets_can_have_different_single_components(self):
        frame = pd.DataFrame({
            'mu_lat_mean_A': [0.0, 1.0],
            'mu_lat_mean_B': [0.0, 1.0],
            'std_ale_lat_A': [0.2, 0.3],
            'std_epi_lat_B': [0.4, 0.5],
        })
        mode, _, epi, ale = temp_tune.load_latent_components(frame, ['A', 'B'])
        self.assertEqual(mode, 'gaussian')
        np.testing.assert_array_equal(epi[:, 0], [0.0, 0.0])
        np.testing.assert_array_equal(ale[:, 1], [0.0, 0.0])
        np.testing.assert_allclose(ale[:, 0], [0.2, 0.3])
        np.testing.assert_allclose(epi[:, 1], [0.4, 0.5])

    def test_invalid_apply_data_does_not_overwrite_existing_taus(self):
        with TemporaryDirectory() as tmp, patch.object(temp_tune.cfg, 'TARGETS', ['A', 'B']):
            run = Path(tmp)
            (run / 'config_used.yaml').write_text('{}\n', encoding='utf-8')
            torch.save({'tf_info': {'A': 'none', 'B': 'none'}}, run / 'data_meta.pt')
            original = json.dumps({'A': 2.0, 'B': 3.0}) + '\n'
            (run / 'taus.json').write_text(original, encoding='utf-8')
            val = pd.DataFrame()
            for target in ['A', 'B']:
                val[f'true_{target}'] = [0.1, 0.2]
                val[f'mu_lat_mean_{target}'] = [0.12, 0.18]
                val[f'std_tot_lat_{target}'] = [0.1, 0.1]
            val.to_csv(run / 'predictions_val.csv', index=False)
            app = val.drop(columns=['std_tot_lat_A', 'std_tot_lat_B'])
            app['std_ale_lat_A'] = [0.1, 0.1]
            app.to_csv(run / 'predictions_test.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'latent uncertainty.*B'):
                temp_tune.run_temp_tune(run, K=4, L=4)
            self.assertEqual((run / 'taus.json').read_text(encoding='utf-8'), original)
            self.assertFalse((run / 'predictions_test_cal.csv').exists())
