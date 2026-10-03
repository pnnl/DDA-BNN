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
            self.assertFalse(
                artifact_dir.exists(),
                "release config import/post-processing must not create output directories",
            )


    def test_load_rebases_relative_paths_when_root_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "alternate-root"
            override = Path(temp_dir) / "override.yaml"
            override.write_text(
                f"ROOT_DIR: {root.as_posix()}\n",
                encoding="utf-8",
            )

            cfg.load(override)

            expected_root = root.resolve()
            self.assertEqual(cfg.ROOT_DIR, expected_root)
            self.assertEqual(
                cfg.DATA_FILE,
                expected_root / cfg._baseline["DATA_FILE"],
            )
            self.assertEqual(
                cfg.ARTIFACT_DIR,
                expected_root / cfg._baseline["ARTIFACT_DIR"],
            )


if __name__ == "__main__":
    unittest.main()
