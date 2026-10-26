"""Only marginal standard deviations are retained by split_output()."""
import unittest
from unittest.mock import patch
try:
    import torch
    from training import model as train_model
    from release import model as release_model
    from training.train import split_output as training_split
    from release.inference_api import split_output as release_split
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class MarginalCovarianceTests(unittest.TestCase):
    def test_marginal_std_agrees_with_covariance_diagonal(self):
        for module, splitter in ((train_model, training_split), (release_model, release_split)):
            with self.subTest(module=module.__name__), \
                 patch.object(module.cfg, 'TARGETS', ['A','B','C']), \
                 patch.object(module.cfg, 'UNCERTAINTY_MODE', 'full'):
                out = torch.ones((3, 9)) * 0.7  # D=3, 3 means + 6 parameters
                params = out[:, 3:]
                sigma = torch.sqrt(torch.exp(module.cfg.LOG_VAR_MIN +
                        (module.cfg.LOG_VAR_MAX - module.cfg.LOG_VAR_MIN) *
                        torch.sigmoid(params[:, :3])) + 1e-6)
                cov = module.build_cov_from_params(sigma, params[:, 3:])
                mean, std = splitter(out)
                torch.testing.assert_close(mean, out[:, :3])
                torch.testing.assert_close(std, torch.diagonal(cov, dim1=-2, dim2=-1).sqrt())
                self.assertTrue(bool((cov[:, 1, 0].abs() > 0).all()))

    def test_zero_correlation_full_and_diag_scales_match(self):
        for module, splitter in ((train_model, training_split), (release_model, release_split)):
            with self.subTest(module=module.__name__), \
                 patch.object(module.cfg, 'TARGETS', ['A', 'B', 'C']), \
                 patch.object(module.cfg, 'LOG_VAR_MIN', -40.0), \
                 patch.object(module.cfg, 'LOG_VAR_MAX', -30.0):
                full_output = torch.zeros((2, 9))  # means, log-vars, then zero rho
                with patch.object(module.cfg, 'UNCERTAINTY_MODE', 'full'):
                    _, full_std = splitter(full_output)
                with patch.object(module.cfg, 'UNCERTAINTY_MODE', 'diag'):
                    _, diag_std = splitter(full_output[:, :6])
                torch.testing.assert_close(full_std, diag_std, rtol=1e-6, atol=1e-9)

    def test_cholesky_factor_bounds_do_not_bound_marginal_std(self):
        for module in (train_model, release_model):
            with self.subTest(module=module.__name__), \
                 patch.object(module.cfg, 'RHO_MAX', 0.4):
                sigma = torch.ones((1, 3))
                rho_raw = torch.full((1, 3), 20.0)
                cov = module.build_cov_from_params(sigma, rho_raw)
                expected = torch.tensor([[1.0, (1.0 + 0.4**2)**0.5,
                                          (1.0 + 2 * 0.4**2)**0.5]])
                torch.testing.assert_close(torch.diagonal(cov, dim1=-2, dim2=-1).sqrt(),
                                           expected, rtol=1e-6, atol=1e-6)
