"""Separate model uncertainty from finite-inner-sample integration noise."""
import unittest
import numpy as np
from release import inference_api as api


class NestedComponentTests(unittest.TestCase):
    def test_independent_gaussian_components_converge_to_known_variances(self):
        rng = np.random.default_rng(192)
        means = rng.normal(0., 2., size=(20000, 1, 1, 1))
        cloud = means + rng.normal(0., 3., size=(20000, 16, 1, 1))
        mean, ale, epi, total, quantiles = api.stats_from_nested_cloud(cloud)
        self.assertAlmostEqual(float(ale[0, 0]), 3., delta=0.03)
        self.assertAlmostEqual(float(epi[0, 0]), 2., delta=0.04)
        np.testing.assert_allclose(total**2, ale**2 + epi**2)
        np.testing.assert_array_equal(mean, cloud.mean(axis=(0, 1)))
        np.testing.assert_array_equal(quantiles['q50'], np.quantile(cloud.reshape(-1, 1, 1), .5, axis=0))

    def test_corrected_total_is_not_the_pooled_cloud_standard_deviation(self):
        cloud = np.array([0.0, 2.0, 4.0, 6.0]).reshape(2, 2, 1, 1)
        mean, ale, epi, total, quantiles = api.stats_from_nested_cloud(cloud)
        np.testing.assert_allclose(mean, [[3.0]])
        np.testing.assert_allclose(ale**2, [[2.0]])
        np.testing.assert_allclose(epi**2, [[3.5]])
        np.testing.assert_allclose(total**2, [[5.5]])
        self.assertNotAlmostEqual(float(total[0, 0]), float(cloud.std()))
        np.testing.assert_array_equal(quantiles['q50'], [[3.0]])

    def test_known_absent_component_stays_exactly_zero(self):
        rng = np.random.default_rng(3)
        cloud = rng.normal(size=(100, 20, 2, 2))
        _, ale, epi, total, _ = api.stats_from_nested_cloud(cloud, epistemic_zero=True)
        np.testing.assert_array_equal(epi, np.zeros((2, 2)))
        np.testing.assert_array_equal(total, ale)
        # A purely epistemic cloud has no inner variability; L=1 is valid.
        _, ale, epi, total, _ = api.stats_from_nested_cloud(cloud[:, :1], aleatoric_zero=True)
        np.testing.assert_array_equal(ale, np.zeros((2, 2)))
        np.testing.assert_array_equal(total, epi)

    def test_one_inner_draw_cannot_identify_aleatoric_variance(self):
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            api.stats_from_nested_cloud(np.ones((3, 1, 1, 1)))

    def test_variance_components_support_per_coordinate_absence(self):
        cloud = np.random.default_rng(1).normal(size=(20, 10, 2, 2))
        mask = np.array([[True, False], [False, True]])
        _, _, epi, _, _ = api.stats_from_nested_cloud(cloud, epistemic_zero=mask)
        np.testing.assert_array_equal(epi[mask], np.zeros(2))
