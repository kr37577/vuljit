from __future__ import annotations

import pandas as pd

from RQ3.core import scheduling


def test_normalize_name_accepts_alias() -> None:
    assert scheduling.normalize_name("median") == "strategy1_median"


def test_run_strategy_uses_registry(monkeypatch) -> None:
    dummy_df = pd.DataFrame({"project": ["alpha"]})
    monkeypatch.setitem(scheduling.STRATEGY_REGISTRY, "strategy1_median", lambda **_: dummy_df)
    result = scheduling.run_strategy("strategy1_median")
    assert result.equals(dummy_df)


def test_iter_strategies_returns_sorted() -> None:
    names = [name for name, _ in scheduling.iter_strategies()]
    assert names == sorted(names)
