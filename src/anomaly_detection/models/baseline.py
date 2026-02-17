from __future__ import annotations

from dataclasses import dataclass

from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IsolationForestConfig:
    n_estimators: int = 200
    contamination: str | float = "auto"
    random_state: int = 42


def build_isolation_forest(cfg: IsolationForestConfig) -> IsolationForest:
    return IsolationForest(
        n_estimators=cfg.n_estimators,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
    )