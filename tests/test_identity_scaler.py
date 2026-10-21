"""Regression tests for training/inference with scaling disabled."""
import unittest
import numpy as np

try:
    from training.data_utils import IdentityScaler
    from training.inference_api import scale_like_training_np as training_scale
    from release.inference_api import scale_like_training_np as release_scale
except (ImportError, FileNotFoundError) as exc:
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@unittest.skipIf(_IMPORT_ERROR is not None, f"DDA-BNN dependencies/config unavailable: {_IMPORT_ERROR}")
class IdentityScalerTests(unittest.TestCase):
    def test_identity_has_standard_feature_metadata(self):
        scaler = IdentityScaler()
        scaler.scale_ = np.ones(3)
        self.assertEqual(scaler.n_features_in_, 3)
        np.testing.assert_array_equal(scaler.mean_, np.zeros(3))
        X = np.array([[1., np.nan, 3.]])
        for scale in (training_scale, release_scale):
            np.testing.assert_array_equal(scale(X, scaler), [[1., 0., 3.]])

    def test_missing_values_are_zero_in_model_input_space(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit([[2., 8.], [4., 12.]])
        X = np.array([[np.nan, 12.], [4., 8.]])
        for scale in (training_scale, release_scale):
            got = scale(X, scaler)
            self.assertEqual(float(got[0, 0]), 0.0)
            self.assertEqual(float(got[1, 0]), 1.0)

    def test_all_missing_column_is_batch_invariant(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit([[2., 8.], [4., 12.]])
        alone = np.array([[np.nan, 12.]])
        mixed = np.array([[np.nan, 12.], [100., -100.]])
        for scale in (training_scale, release_scale):
            np.testing.assert_allclose(scale(alone, scaler)[0],
                                       scale(mixed, scaler)[0])
            self.assertEqual(float(scale(alone, scaler)[0, 0]), 0.0)
