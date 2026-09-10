"""Focused release inference helper regressions."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from release import config as cfg
import release.inference_api as inference_api


class ReleaseInferenceHelperTests(unittest.TestCase):
    def tearDown(self) -> None:
        inference_api.apply_config_used(cfg, inference_api._BASE_CONFIG)

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

    def test_unknown_tau_string_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'auto'"):
            inference_api._resolve_taus(
                "manual",
                Path("."),
                torch.float32,
            )

    def test_tau_values_must_be_finite_and_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            inference_api._taus_to_array(
                np.array([1.0, np.nan, 1.0]),
                torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "non-negative"):
            inference_api._taus_to_array(
                np.array([1.0, -0.5, 1.0]),
                torch.float32,
            )

    def test_tau_mapping_reports_missing_targets(self) -> None:
        incomplete = {
            target: 1.0
            for target in cfg.TARGETS[:-1]
        }
        with self.assertRaisesRegex(ValueError, "missing target"):
            inference_api._taus_to_array(incomplete, torch.float32)


if __name__ == "__main__":
    unittest.main()

