from __future__ import annotations

import pandas as pd

from RQ3.core.simulation import SimulationResult, run_minimal_simulation


def test_run_minimal_simulation_aggregates_strategies(monkeypatch) -> None:
    from RQ3.core import simulation

    schedules = {
        "strategy1": pd.DataFrame(
            {
                "project": ["alpha"],
                "merge_date_ts": ["2024-01-01"],
                "scheduled_additional_builds": [2.0],
            }
        ),
        "strategy2": pd.DataFrame(
            {
                "project": ["beta"],
                "merge_date_ts": ["2024-01-02"],
                "scheduled_additional_builds": [3.0],
            }
        ),
    }

    summaries: dict[str, pd.DataFrame] = {
        "strategy1": schedules["strategy1"],
        "strategy2": schedules["strategy2"],
    }

    monkeypatch.setattr(
        simulation,
        "iter_strategies",
        lambda: iter([("strategy1", None), ("strategy2", None)]),
    )
    monkeypatch.setattr(
        simulation,
        "run_strategy",
        lambda name, **_: summaries[name],
    )

    result: SimulationResult = run_minimal_simulation(
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        risk_threshold=0.5,
        return_details=True,
    )

    assert list(result.summary["strategy"]) == ["strategy1", "strategy2"]
    assert result.schedules.keys() == {"strategy1", "strategy2"}


def test_simulation_result_serialisation_helpers() -> None:
    summary = pd.DataFrame({"strategy": ["s1"], "value": [1]})
    result = SimulationResult(summary=summary, schedules={}, metadata={"source": "test"})
    assert result.summary.to_dict("records") == [{"strategy": "s1", "value": 1}]
    assert result.metadata["source"] == "test"
