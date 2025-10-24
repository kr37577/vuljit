from __future__ import annotations

import pandas as pd

from RQ3.core.metrics import (
    aggregate_strategy_metrics,
    prepare_daily_totals,
    prepare_project_metrics,
    safe_ratio,
)


def _baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "baseline_detection_builds": [4.0, 6.0],
            "baseline_detection_days": [2.0, 3.0],
            "builds_per_day": [2.0, 2.0],
        }
    )


def _schedules() -> dict[str, pd.DataFrame]:
    return {
        "strategy1": pd.DataFrame(
            {
                "project": ["alpha", "alpha", "beta"],
                "merge_date_ts": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ],
                "scheduled_additional_builds": [1.0, 2.0, 3.0],
            }
        ),
        "strategy2": pd.DataFrame(
            {
                "project": ["alpha"],
                "merge_date": ["2024-01-04"],
                "scheduled_additional_builds": [1.0],
            }
        ),
    }


def test_prepare_project_metrics_merges_baseline() -> None:
    project_metrics = prepare_project_metrics(_schedules(), _baseline_df())
    assert set(project_metrics["strategy"]) == {"strategy1", "strategy2"}
    alpha_metrics = project_metrics[
        (project_metrics["strategy"] == "strategy1") & (project_metrics["project"] == "alpha")
    ].iloc[0]
    assert alpha_metrics["trigger_count"] == 2
    assert alpha_metrics["scheduled_builds"] == 3.0
    assert alpha_metrics["baseline_detection_builds"] == 4.0


def test_aggregate_strategy_metrics_summarises_values() -> None:
    project_metrics = prepare_project_metrics(_schedules(), _baseline_df())
    aggregate = aggregate_strategy_metrics(project_metrics)
    row = aggregate.loc[aggregate["strategy"] == "strategy1"].iloc[0]
    assert row["projects"] == 2
    assert row["total_scheduled_builds"] == 6.0


def test_prepare_daily_totals_groups_by_day() -> None:
    daily = prepare_daily_totals(_schedules())
    assert set(daily["strategy"]) == {"strategy1", "strategy2"}
    alpha_rows = daily[(daily["strategy"] == "strategy1") & (daily["project"] == "alpha")]
    assert len(alpha_rows) == 2
    assert float(alpha_rows.loc[alpha_rows["date"] == "2024-01-02"]["scheduled_additional_builds"]) == 2.0


def test_safe_ratio_handles_zero_denominator() -> None:
    assert safe_ratio(4, 2) == 2.0
    assert safe_ratio(1, 0) != safe_ratio(1, 0)  # NaN check
