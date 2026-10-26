"""Full covariance must be positive definite and have one likelihood."""
import unittest
import torch
from unittest.mock import patch

try:
    from training import model as train_model
    from release import model as release_model
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class FullCovarianceTests(unittest.TestCase):
    def test_covariance_positive_definite_for_four_targets(self):
        sig = torch.ones((2, 4))
        raw = torch.full((2, 6), -20.0)
        for m in (train_model, release_model):
            with patch.object(m.cfg, 'RHO_MAX', 0.4):
                cov = m.build_cov_from_params(sig, raw)
                self.assertTrue(bool((torch.linalg.eigvalsh(cov) > 0).all()))

    def test_nll_matches_direct_multivariate_normal_and_has_gradients(self):
        for m in (train_model, release_model):
            with patch.object(m.cfg, 'RHO_MAX', 0.4), \
                 patch.object(m.cfg, 'LOG_VAR_MIN', -10.0), \
                 patch.object(m.cfg, 'LOG_VAR_MAX', -3.0):
                out = torch.randn((5, 9), requires_grad=True)  # D=3: 3 means + 6 params
                truth = torch.randn((5, 3))
                nll = m.gaussian_nll_full(out, truth, 3)
                sigma = torch.sqrt(torch.exp(-10 + 7 * torch.sigmoid(out[:, 3:6])) + 1e-6)
                cov = m.build_cov_from_params(sigma, out[:, 6:])
                want = -torch.distributions.MultivariateNormal(out[:, :3], covariance_matrix=cov).log_prob(truth).mean()
                torch.testing.assert_close(nll, want)
                nll.backward()
                self.assertTrue(bool(torch.isfinite(out.grad).all()))
