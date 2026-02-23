from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict: {path}")

    return data


def load_dataset_config(config_path: str | Path) -> dict[str, Any]:
    return load_config(config_path)


def load_train_config(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    required_sections = ["training", "model"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(
            f"Training config missing required section(s) {missing}: {Path(config_path)}"
        )
    return cfg
