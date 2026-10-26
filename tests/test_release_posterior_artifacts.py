"""Bayesian inference must use a complete saved posterior."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import torch
import pyro
import yaml
from sklearn.preprocessing import StandardScaler
from release import inference_api as api


class PosteriorArtifactTests(unittest.TestCase):
    def tearDown(self):
        pyro.clear_param_store()
        api.apply_config_used(api.cfg, api._BASE_CONFIG)

    def make_run(self, run, bayesian=True):
        config = dict(TARGETS=['A'], UNCERTAINTY_MODE='diag', USE_ZSCORE_SCALING=True,
                      LAYER_SPEC=[dict(units=4, type='bnn', act='Tanh')] if bayesian else [])
        (run / 'config_used.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        pyro.clear_param_store()
        with patch.multiple(api.cfg, **config):
            net = api.HybridNet(2)
            if bayesian:
                net.guide(torch.zeros(1, 2))
                pyro.get_param_store().save(run / 'pyro_params.pt')
            torch.save(net.state_dict(), run / 'model.pth')
        torch.save(dict(x_scaler=StandardScaler().fit([[0., 0.], [1., 1.]]),
                        y_scaler=StandardScaler().fit([[0.], [1.]]),
                        tf_info={'A': 'none'}), run / 'data_meta.pt')

    def test_missing_posterior_is_an_error_for_bayesian_models(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp); self.make_run(run)
            (run / 'pyro_params.pt').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'posterior'):
                api.run_inference_latent(run, np.zeros((1, 2)), num_mc=4, taus=None)

    def test_partial_posterior_cannot_initialize_a_replacement_parameter(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp); self.make_run(run)
            state = torch.load(run / 'pyro_params.pt', weights_only=False)
            key = next(k for k in state['params'] if k.endswith('.scale'))
            del state['params'][key]; del state['constraints'][key]
            torch.save(state, run / 'pyro_params.pt')
            with self.assertRaisesRegex(ValueError, 'missing guide parameter'):
                api.run_inference_latent(run, np.zeros((1, 2)), num_mc=4, taus=None)

    def test_complete_posterior_produces_the_requested_sample_count(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp); self.make_run(run)
            result = api.run_inference_latent(run, np.zeros((1, 2)), num_mc=4,
                                              return_samples=True, taus=None)
            self.assertEqual(tuple(result[3].shape), (4, 1, 1))
            self.assertTrue(bool(torch.isfinite(result[0]).all()))

    def test_deterministic_models_do_not_require_posterior_artifacts(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp); self.make_run(run, bayesian=False)
            self.assertFalse((run / 'pyro_params.pt').exists())
            result = api.run_inference_latent(run, np.zeros((1, 2)), taus=None)
            np.testing.assert_array_equal(result[2], np.zeros((1, 1)))
