"""Regression tests for release configuration path resolution."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from release import config as cfg


class ReleaseConfigPathTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Restore the import-time baseline after tests that intentionally
        # exercise private post-processing helpers.
        cfg._apply(cfg._baseline)
        cfg._post()

    def test_root_dir_expands_user_home(self) -> None:
        old_home = os.environ.get("HOME")
        try:
            with tempfile.TemporaryDirectory() as home:
                os.environ["HOME"] = home
                cfg.ROOT_DIR = "~/dda-bnn-root"
                cfg.DATA_FILE = "data/example.csv"
                cfg.ARTIFACT_DIR = "artifacts"

                cfg._post()

                expected_root = (Path(home) / "dda-bnn-root").resolve()
                self.assertEqual(cfg.ROOT_DIR, expected_root)
                self.assertEqual(cfg.ns.ROOT_DIR, expected_root)
                self.assertEqual(
                    cfg.DATA_FILE,
                    expected_root / "data" / "example.csv",
                )
                self.assertEqual(
                    cfg.ARTIFACT_DIR,
                    expected_root / "artifacts",
                )
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home

    def test_absolute_path_overrides_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            data_file = root / "custom" / "data.csv"
            artifact_dir = root / "custom-artifacts"

            cfg.ROOT_DIR = root
            cfg.DATA_FILE = data_file
            cfg.ARTIFACT_DIR = artifact_dir

            cfg._post()

            self.assertEqual(cfg.ROOT_DIR, root)
            self.assertEqual(cfg.DATA_FILE, data_file)
            self.assertEqual(cfg.ARTIFACT_DIR, artifact_dir)


if __name__ == "__main__":
    unittest.main()
