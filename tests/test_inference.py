"""Training inference regression tests."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import training.config as training_cfg
import training.inference_api as training_api


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "release" / "chosen_model"


class TrainingInferenceTests(unittest.TestCase):
    @staticmethod
    def representative_input() -> np.ndarray:
        return np.array(
            [[
                28.0,
                1.0,
                0.0,
                np.pi * 0.121537 / 0.7,
                1.76891,
                0.816859706,
                0.121279497,
                0.066140142,
            ]],
            dtype=np.float32,
        )

    def tearDown(self) -> None:
        training_api.apply_config_used(
            training_cfg,
            training_api._BASE_CONFIG,
        )

    def test_physical_inference_disables_nested_tau_scaling(self) -> None:
        """Physical inference must apply tau calibration exactly once."""
        with patch.object(
            training_api,
            "run_inference_latent",
            wraps=training_api.run_inference_latent,
        ) as latent_mock:
            training_api.run_inference_phys(
                MODEL_DIR,
                self.representative_input(),
                num_mc=4,
                seed=0,
                L=4,
            )

        latent_mock.assert_called_once()
        self.assertIn("taus", latent_mock.call_args.kwargs)
        self.assertIsNone(latent_mock.call_args.kwargs["taus"])

    def test_physical_none_taus_disables_all_tau_scaling(self) -> None:
        with patch.object(
            training_api,
            "run_inference_latent",
            wraps=training_api.run_inference_latent,
        ) as latent_mock:
            training_api.run_inference_phys(
                MODEL_DIR,
                self.representative_input(),
                num_mc=4,
                seed=0,
                taus=None,
                L=4,
            )

        latent_mock.assert_called_once()
        self.assertIsNone(latent_mock.call_args.kwargs["taus"])

    def test_run_config_resets_between_model_directories(self) -> None:
        baseline_eps = training_api._BASE_CONFIG["TF_EPS"]

        with tempfile.TemporaryDirectory() as temp_dir:
            first_run = Path(temp_dir) / "first"
            second_run = Path(temp_dir) / "second"
            first_run.mkdir()
            second_run.mkdir()

            (first_run / "config_used.yaml").write_text(
                "TF_EPS: 0.125\n",
                encoding="utf-8",
            )

            training_api._apply_run_config(first_run)
            self.assertEqual(training_cfg.TF_EPS, 0.125)
            self.assertEqual(training_cfg.ns.TF_EPS, 0.125)

            training_api._apply_run_config(second_run)
            self.assertEqual(training_cfg.TF_EPS, baseline_eps)
            self.assertEqual(training_cfg.ns.TF_EPS, baseline_eps)


if __name__ == "__main__":
    unittest.main()

