"""Wrappers around Phase 3 scheduling strategies with unified naming."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Iterator, Tuple

import pandas as pd

try:
    from ..additional_build_strategies import (
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )
except ImportError:  # pragma: no cover - allow execution from repo root
    from additional_build_strategies import (
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )

StrategyFunc = Callable[..., pd.DataFrame]


def _strategy4_wrapper(**kwargs) -> pd.DataFrame:
    frame, *_ = strategy4_cross_project_regression(**kwargs)
    return frame


STRATEGY_REGISTRY: Dict[str, StrategyFunc] = {
    "strategy1_median": strategy1_median_schedule,
    "strategy2_random": strategy2_random_within_median_range,
    "strategy3_line_proportional": strategy3_line_change_proportional,
    "strategy4_regression": _strategy4_wrapper,
}

ALIASES: Dict[str, str] = {
    "median": "strategy1_median",
    "random": "strategy2_random",
    "line": "strategy3_line_proportional",
    "regression": "strategy4_regression",
    "strategy4": "strategy4_regression",
}


def normalize_name(name: str) -> str:
    """Resolve short-hand aliases and validate the strategy identifier."""

    canonical = STRATEGY_REGISTRY.get(name)
    if canonical is not None:
        return name
    alias = ALIASES.get(name)
    if alias:
        return alias
    raise KeyError(f"Unknown strategy name: {name}")


def get_strategy(name: str) -> StrategyFunc:
    """Return the callable implementing the requested strategy."""

    canonical = normalize_name(name)
    return STRATEGY_REGISTRY[canonical]


def run_strategy(name: str, **kwargs) -> pd.DataFrame:
    """Execute a strategy by name, passing keyword arguments through."""

    strategy = get_strategy(name)
    frame = strategy(**kwargs)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Strategy {name!r} must return a pandas.DataFrame.")
    return frame


def iter_strategies() -> Iterator[Tuple[str, StrategyFunc]]:
    """Yield registered strategies in deterministic order."""

    for name in sorted(STRATEGY_REGISTRY.keys()):
        yield name, STRATEGY_REGISTRY[name]


def list_strategies() -> Iterable[str]:
    """Return the available strategy identifiers."""

    return list(sorted(STRATEGY_REGISTRY.keys()))


__all__ = [
    "StrategyFunc",
    "STRATEGY_REGISTRY",
    "normalize_name",
    "get_strategy",
    "run_strategy",
    "iter_strategies",
    "list_strategies",
]
