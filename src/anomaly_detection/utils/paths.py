from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def configs(self) -> Path:
        return self.root / "configs"


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repository root looking upwards until pyproject.toml is found
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find pyproject.toml")
