"""IdentityScaler fitting and finite-value input contracts."""
import unittest
import numpy as np
try:
    from training.data_utils import IdentityScaler
    from training.inference_api import scale_like_training_np as train_scale
    from release.inference_api import scale_like_training_np as release_scale
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

@unittest.skipIf(_IMPORT_ERROR is not None, f"Repository dependencies/config unavailable: {_IMPORT_ERROR}")
class IdentityScalerContractTests(unittest.TestCase):
    def test_fit_sets_dimensions(self):
        scaler = IdentityScaler().fit(np.ones((5, 3)))
        self.assertEqual(scaler.n_features_in_, 3)
        np.testing.assert_equal(scaler.scale_, np.ones(3))
        np.testing.assert_equal(scaler.mean_, np.zeros(3))

    def test_fit_transform_sets_dimensions_and_preserves_data(self):
        scaler = IdentityScaler()
        arr = np.array([[1.0, 2.0]])
        np.testing.assert_equal(scaler.fit_transform(arr), arr)
        self.assertEqual(scaler.n_features_in_, 2)

    def test_only_two_dimensional_inputs_are_valid_for_fit(self):
        with self.assertRaisesRegex(ValueError, '2-D'):
            IdentityScaler().fit(np.ones(3))

    def test_missing_nan_is_zero_but_infinity_rejected(self):
        scaler = IdentityScaler().fit(np.ones((2, 3)))
        arr = np.array([[np.nan, np.nan, np.nan]])
        for fn in (train_scale, release_scale):
            np.testing.assert_array_equal(fn(arr, scaler), np.zeros((1, 3)))
            for bad in (np.inf, -np.inf):
                with self.assertRaisesRegex(ValueError, 'Infinite'):
                    fn(np.array([[bad, 0., 1.]]), scaler)
