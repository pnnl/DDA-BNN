"""Inference and deterministic losses must agree with the observation likelihood."""
from contextlib import ExitStack
import math
import unittest
from unittest.mock import patch

import pyro
import torch

from training import model as training_model
from release import model as release_model
from training.train import split_output as training_split
from training.inference_api import split_output as training_api_split
from release.inference_api import split_output as release_split


class ObservationConsistencyTests(unittest.TestCase):
    def configured_model(self, stack, module, mode, bounds):
        for name, value in {
            'TARGETS': ['A', 'B', 'C'], 'UNCERTAINTY_MODE': mode,
            'LAYER_SPEC': [], 'LOG_VAR_MIN': bounds[0],
            'LOG_VAR_MAX': bounds[1],
        }.items():
            stack.enter_context(patch.object(module.cfg, name, value))
        pyro.clear_param_store()
        net = module.HybridNet(1)
        with torch.no_grad():
            net.head.weight.zero_()
            net.head.bias.fill_(0.7)
        return net

    def test_split_output_matches_observed_standard_deviation(self):
        for module, splitters in (
            (training_model, (training_split, training_api_split)),
            (release_model, (release_split,)),
        ):
            for mode in ('diag', 'full'):
                for bounds in ((-10.0, -3.0), (-40.0, -30.0)):
                    with self.subTest(module=module.__name__, mode=mode, bounds=bounds), ExitStack() as stack:
                        net = self.configured_model(stack, module, mode, bounds)
                        trace = pyro.poutine.trace(net).get_trace(torch.zeros(2, 1), torch.zeros(2, 3))
                        output = trace.nodes['_RETURN']['value']
                        obs = trace.nodes['obs']['fn']
                        for splitter in splitters:
                            mu, std = splitter(output)
                            torch.testing.assert_close(mu, obs.mean)
                            torch.testing.assert_close(std, obs.stddev, rtol=1e-6, atol=0.0)

    def test_diagonal_nll_matches_observation_density_and_gradient(self):
        for module in (training_model, release_model):
            for bounds in ((-10.0, -3.0), (-40.0, -30.0)):
                for residual in (0.0, 0.2):
                    with self.subTest(module=module.__name__, bounds=bounds, residual=residual), ExitStack() as stack:
                        net = self.configured_model(stack, module, 'diag', bounds)
                        # Nonzero residual also exercises the reciprocal-variance
                        # term and nonzero gradients with respect to the mean.
                        truth = torch.full((2, 3), 0.7 + residual)
                        trace = pyro.poutine.trace(net).get_trace(torch.zeros(2, 1), truth)
                        output = trace.nodes['_RETURN']['value']
                        actual = module.gaussian_nll_diag(output, truth, 3)
                        # The helper intentionally omits the Gaussian constant.
                        expected = -trace.nodes['obs']['fn'].log_prob(truth).mean() - 1.5 * math.log(2 * math.pi)
                        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                        actual_grad, = torch.autograd.grad(actual, output, retain_graph=True)
                        expected_grad, = torch.autograd.grad(expected, output)
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
