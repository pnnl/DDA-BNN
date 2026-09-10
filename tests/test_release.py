"""Fast validation and smoke tests for the bundled DDA-BNN release."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nbformat
import numpy as np
import yaml

from release.inference_api import run_inference_latent, run_inference_phys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "release" / "chosen_model"


class RepositoryValidationTests(unittest.TestCase):
    def test_expected_default_configs_exist(self) -> None:
        for package in ("release", "training"):
            config_dir = PROJECT_ROOT / package / "configs"
            names = {path.name for path in config_dir.iterdir()}
            expected = config_dir / "default.yaml"
            with self.subTest(package=package):
                self.assertIn(
                    "default.yaml",
                    names,
                    f"Missing expected config: {expected}",
                )
                self.assertNotIn(
                    "default.YAML",
                    names,
                    "Legacy uppercase config filename must not remain",
                )

    def test_configuration_files_are_valid_yaml(self) -> None:
        config_files = sorted(
            path
            for path in PROJECT_ROOT.rglob("*")
            if path.is_file()
            and "configs" in path.parts
            and path.suffix.lower() in {".yaml", ".yml"}
        )
        self.assertGreater(len(config_files), 0)

        for path in config_files:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                with path.open(encoding="utf-8") as stream:
                    self.assertIsInstance(yaml.safe_load(stream), dict)

    def test_config_imports_are_cwd_independent(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        with tempfile.TemporaryDirectory() as temp_dir:
            for package in ("release", "training"):
                code = f"from {package} import config as cfg; print(cfg.ROOT_DIR)"
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=temp_dir,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                with self.subTest(package=package):
                    self.assertEqual(Path(result.stdout.strip()).resolve(), PROJECT_ROOT)

    def test_release_notebook_is_valid(self) -> None:
        notebook_path = PROJECT_ROOT / "release" / "inference_notebook.ipynb"
        with notebook_path.open(encoding="utf-8") as stream:
            notebook = nbformat.read(stream, as_version=4)
        nbformat.validate(notebook)


class ReleaseInferenceTests(unittest.TestCase):
    @staticmethod
    def representative_input() -> np.ndarray:
        # One representative row in the documented feature order:
        # Npp, V/V0, coating_RI_imag, Xve, core_Df, Qext_HS, SSA_HS, g_HS.
        return np.array(
            [[28.0, 1.0, 0.0, np.pi * 0.121537 / 0.7, 1.76891,
              0.816859706, 0.121279497, 0.066140142]],
            dtype=np.float32,
        )

    @staticmethod
    def load_taus() -> dict[str, float]:
        with (MODEL_DIR / "taus.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def test_bundled_model_runs_inference(self) -> None:
        required_artifacts = {
            "config_used.yaml",
            "data_meta.pt",
            "model.pth",
            "pyro_params.pt",
            "taus.json",
        }
        self.assertTrue(required_artifacts.issubset({path.name for path in MODEL_DIR.iterdir()}))

        mean, std_ale, std_epi, std_tot, quantiles = run_inference_phys(
            MODEL_DIR,
            self.representative_input(),
            num_mc=4,
            seed=0,
            L=4,
        )

        for name, values in {
            "mean": mean,
            "aleatoric standard deviation": std_ale,
            "epistemic standard deviation": std_epi,
            "total standard deviation": std_tot,
            **quantiles,
        }.items():
            with self.subTest(output=name):
                self.assertEqual(values.shape, (1, 3))
                self.assertTrue(np.isfinite(values).all())

        self.assertTrue((std_ale >= 0).all())
        self.assertTrue((std_epi >= 0).all())
        self.assertTrue((std_tot >= 0).all())
        self.assertTrue((mean[:, 0] > 0).all())
        self.assertTrue(((mean[:, 1:] > 0) & (mean[:, 1:] < 1)).all())
        self.assertTrue((quantiles["q05"] <= quantiles["q50"]).all())
        self.assertTrue((quantiles["q50"] <= quantiles["q95"]).all())

    def test_physical_auto_taus_matches_explicit_taus(self) -> None:
        x_raw = self.representative_input()
        taus = self.load_taus()

        explicit = run_inference_phys(
            MODEL_DIR, x_raw, num_mc=4, seed=0, taus=taus, L=4
        )
        automatic = run_inference_phys(
            MODEL_DIR, x_raw, num_mc=4, seed=0, L=4
        )

        for index, label in enumerate(("mean", "std_ale", "std_epi", "std_tot")):
            with self.subTest(output=label):
                np.testing.assert_allclose(
                    automatic[index], explicit[index], rtol=1e-6, atol=1e-7
                )

        self.assertEqual(automatic[4].keys(), explicit[4].keys())
        for key in automatic[4]:
            with self.subTest(quantile=key):
                np.testing.assert_allclose(
                    automatic[4][key], explicit[4][key], rtol=1e-6, atol=1e-7
                )

    def test_latent_auto_taus_matches_explicit_taus(self) -> None:
        x_raw = self.representative_input()
        taus = self.load_taus()

        explicit = run_inference_latent(
            MODEL_DIR, x_raw, num_mc=4, seed=0, taus=taus
        )
        automatic = run_inference_latent(
            MODEL_DIR, x_raw, num_mc=4, seed=0
        )

        for index, label in enumerate(("mean", "std_ale", "std_epi")):
            explicit_values = explicit[index]
            automatic_values = automatic[index]
            with self.subTest(output=label):
                self.assertEqual(explicit_values is None, automatic_values is None)
                if explicit_values is not None:
                    np.testing.assert_allclose(
                        automatic_values.numpy(),
                        explicit_values.numpy(),
                        rtol=1e-6,
                        atol=1e-7,
                    )

if __name__ == "__main__":
    unittest.main()
