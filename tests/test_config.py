from pathlib import Path

import pytest

from anomaly_detection.utils.config import load_config, load_train_config


def test_load_config_reads_yaml_mapping(tmp_path: Path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")

    cfg = load_config(cfg_path)

    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 2


def test_load_train_config_requires_sections(tmp_path: Path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text("training:\n  seed: 42\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required section"):
        load_train_config(cfg_path)
