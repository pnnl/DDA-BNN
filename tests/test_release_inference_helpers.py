"""Focused release inference helper regressions."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from release import config as cfg
import release.inference_api as inference_api


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "release" / "chosen_model"


class ReleaseInferenceHelperTests(unittest.TestCase):
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
        inference_api.apply_config_used(cfg, inference_api._BASE_CONFIG)

    def test_apply_config_used_ignores_nonbaseline_and_runtime_state(self) -> None:
        original_default_file = cfg._DEFAULT_FILE
        original_runtime = {
            key: getattr(cfg, key)
            for key in inference_api._RUNTIME_CONFIG_KEYS
        }

        inference_api.apply_config_used(
            cfg,
            {
                "TF_EPS": 0.125,
                "_DEFAULT_FILE": "must-not-be-applied",
                "ROOT_DIR": "/training-host/root",
                "DATA_FILE": "/training-host/data.csv",
                "ARTIFACT_DIR": "/training-host/artifacts",
                "DEVICE": "cuda",
            },
        )

        self.assertEqual(cfg.TF_EPS, 0.125)
        self.assertEqual(cfg.ns.TF_EPS, 0.125)
        self.assertEqual(cfg._DEFAULT_FILE, original_default_file)
        for key, value in original_runtime.items():
            self.assertEqual(getattr(cfg, key), value)
            self.assertEqual(getattr(cfg.ns, key), value)

    def test_run_config_resets_missing_values_to_baseline_and_syncs_namespace(self) -> None:
        baseline_eps = inference_api._BASE_CONFIG["TF_EPS"]

        with tempfile.TemporaryDirectory() as temp_dir:
            first_run = Path(temp_dir) / "first"
            second_run = Path(temp_dir) / "second"
            first_run.mkdir()
            second_run.mkdir()

            (first_run / "config_used.yaml").write_text(
                "TF_EPS: 0.125\n",
                encoding="utf-8",
            )

            inference_api._apply_run_config(first_run)
            self.assertEqual(cfg.TF_EPS, 0.125)
            self.assertEqual(cfg.ns.TF_EPS, 0.125)

            inference_api._apply_run_config(second_run)
            self.assertEqual(cfg.TF_EPS, baseline_eps)
            self.assertEqual(cfg.ns.TF_EPS, baseline_eps)

    def test_physical_inference_disables_nested_tau_scaling(self) -> None:
        """Physical inference must apply tau calibration exactly once."""
        with patch.object(
            inference_api,
            "run_inference_latent",
            wraps=inference_api.run_inference_latent,
        ) as latent_mock:
            inference_api.run_inference_phys(
                MODEL_DIR,
                self.representative_input(),
                num_mc=4,
                seed=0,
                L=4,
            )

        latent_mock.assert_called_once()
        self.assertIn("taus", latent_mock.call_args.kwargs)
        self.assertIsNone(latent_mock.call_args.kwargs["taus"])

    def test_unknown_tau_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'auto'"):
            inference_api._resolve_taus(
                "manual",
                Path("."),
                torch.float32,
            )

    def test_tau_values_must_be_finite_and_strictly_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            inference_api._taus_to_array(
                np.array([1.0, np.nan, 1.0]),
                torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            inference_api._taus_to_array(
                np.array([1.0, 0.0, 1.0]),
                torch.float32,
            )

    def test_tau_mapping_reports_missing_targets(self) -> None:
        incomplete = {
            target: 1.0
            for target in cfg.TARGETS[:-1]
        }
        with self.assertRaisesRegex(ValueError, "missing target"):
            inference_api._taus_to_array(incomplete, torch.float32)


    def test_tau_mapping_rejects_unexpected_targets(self) -> None:
        unexpected = {target: 1.0 for target in cfg.TARGETS}
        unexpected["not-a-target"] = 1.0
        with self.assertRaisesRegex(ValueError, "unexpected target"):
            inference_api._taus_to_array(unexpected, torch.float32)


if __name__ == "__main__":
    unittest.main()

