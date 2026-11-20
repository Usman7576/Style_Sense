"""Config and path helpers. All paths resolve relative to the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_path(p: str | Path) -> Path:
    """Return an absolute path; relative paths are taken from the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by short name (``"quality"`` -> ``configs/quality.yaml``) or by path."""
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):
        p = Path("configs") / f"{name_or_path}.yaml"
    with open(repo_path(p), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
