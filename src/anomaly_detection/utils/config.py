from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict: {path}")

    return data


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load dataset configuration from a given path.
    Expected usage: load_dataset_config(config_path)
    """
    return load_yaml(Path(config_path))
