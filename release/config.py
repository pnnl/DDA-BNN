"""Global configuration loaded from ``configs/default.yaml``.

All keys become module-level globals and are mirrored in ``cfg.ns``.

Example::

    from release import config as cfg
    print(cfg.TARGETS)   # ['Qext', 'SSA', 'g']
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import json
import yaml

import torch


def _device(val: str | None) -> torch.device:
    """Convert device string to torch.device, auto-detecting CUDA if needed."""
    if val in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(val)


def _apply(d: dict[str, Any]) -> None:
    """Copy key-value pairs into module globals and ns."""
    g = globals()
    for k, v in d.items():
        g[k] = v
        setattr(ns, k, v)


def _post() -> None:
    """Resolve paths and device after every load/override."""
    this_dir = Path(__file__).resolve().parent
    root = Path(globals()["ROOT_DIR"]).expanduser()
    if not root.is_absolute():
        root = this_dir / root
    root = root.resolve()
    globals()["ROOT_DIR"] = root
    ns.ROOT_DIR = root

    for key in ("DATA_FILE", "ARTIFACT_DIR"):
        path = Path(globals()[key]).expanduser()
        if not path.is_absolute():
            path = root / path
        globals()[key] = path
        setattr(ns, key, path)
    ns.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    globals()["DEVICE"] = ns.DEVICE = _device(globals()["DEVICE"])


def _read_file(path: Path) -> dict[str, Any]:
    """Return dict from JSON or YAML file, chosen by extension."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(path.read_text()) or {}
    if suffix == ".json":
        return json.loads(path.read_text()) or {}
    raise ValueError(f"Unsupported config format: '{path.suffix}'")


def print_cfg():
    """Nicely print current configuration."""
    from pprint import pformat
    print("\nCurrent configuration")
    print("-" * 70)
    g = globals()
    for k in _baseline.keys():
        v = g[k]
        if isinstance(v, (list, dict, tuple, Path)):
            pretty = pformat(v, compact=True, width=80)
            print(f"{k} = {pretty}")
        else:
            print(f"{k} = {v}")
    print("-" * 70 + "\n")


ns = SimpleNamespace()

_DEFAULT_FILE = Path(__file__).resolve().parent / "configs" / "default.yaml"
_baseline = _read_file(_DEFAULT_FILE)

_apply(_baseline)
_post()


def load(path: str | Path) -> None:
    """
    Override the current configuration with another YAML or JSON file.
    Unknown keys raise KeyError.
    """
    path = Path(path).expanduser().resolve()
    data = _read_file(path)

    unknown = data.keys() - _baseline.keys()
    if unknown:
        raise KeyError(f"Unknown config keys: {', '.join(unknown)}")

    _apply(data)
    _post()


__all__ = ["ns", "load"] + list(_baseline.keys())
