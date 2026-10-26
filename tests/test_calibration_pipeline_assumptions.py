"""Synthetic end-to-end temperature fitting using single uncertainty component."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json
try:
    import numpy as np
    import pandas as pd
    import torch
    from training import temp_tune
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class SingleComponentCalibrationTests(unittest.TestCase):
    def test_complete_fit_and_apply_for_single_uncertainty_component(self):
        targets = list(temp_tune.cfg.TARGETS)
        for component in ('ale', 'epi'):
            with self.subTest(component=component), TemporaryDirectory() as tmp:
                run = Path(tmp)
                (run/'config_used.yaml').write_text('{}\n', encoding='utf-8')
                torch.save({'tf_info': {t: 'none' for t in targets}}, run/'data_meta.pt')
                val = pd.DataFrame()
                app = pd.DataFrame()
                for t in targets:
                    val[f'true_{t}'] = np.array([0.1, 0.2, 0.3, 0.4])
                    val[f'mu_lat_mean_{t}'] = np.array([0.12, 0.18, 0.32, 0.38])
                    val[f'std_tot_lat_{t}'] = np.full(4, 0.15)
                    app[f'true_{t}'] = np.array([0.1, 0.2, 0.3, 0.4])
                    app[f'mu_lat_mean_{t}'] = np.array([0.12, 0.18, 0.32, 0.38])
                    app[f'std_{component}_lat_{t}'] = np.full(4, 0.15)
                val.to_csv(run/'predictions_val.csv', index=False)
                app.to_csv(run/'predictions_test.csv', index=False)
                taus = temp_tune.run_temp_tune(run, K=8, L=8)
                self.assertEqual(set(taus), set(targets))
                self.assertTrue(all(np.isfinite(v) and v > 0 for v in taus.values()))
                self.assertTrue((run/'predictions_test_cal.csv').is_file())
                self.assertTrue((run/'taus.json').is_file())
                self.assertEqual(set(json.loads((run/'taus.json').read_text())), set(targets))
