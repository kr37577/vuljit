from __future__ import annotations

import pandas as pd

from RQ3.core.simulation import (
    normalize_to_date,
    prepare_schedule_for_waste_analysis,
    summarize_wasted_builds,
)


def test_prepare_schedule_for_waste_analysis_normalises_dates() -> None:
    df = pd.DataFrame(
        {
            "project": ["alpha"],
            "merge_date_ts": ["2024-01-10T12:34:56Z"],
            "scheduled_additional_builds": [2.0],
        }
    )
    prepared = prepare_schedule_for_waste_analysis(df)
    assert prepared["schedule_date"].iloc[0] == normalize_to_date(pd.Series(["2024-01-10T12:34:56Z"])).iloc[0]
    assert prepared["scheduled_additional_builds"].iloc[0] == 2.0


def test_summarize_wasted_builds_classifies_events() -> None:
    schedules = {
        "strategy": pd.DataFrame(
            {
                "project": ["alpha", "alpha", "alpha"],
                "merge_date_ts": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ],
                "scheduled_additional_builds": [0.5, 1.5, 1.0],
            }
        )
    }
    baseline_df = pd.DataFrame(
        {
            "project": ["alpha"],
            "baseline_detection_builds": [2.0],
            "baseline_detection_days": [2.0],
            "builds_per_day": [1.0],
        }
    )

    summary, events = summarize_wasted_builds(schedules, baseline_df, detection_window_days=0)
    assert summary.iloc[0]["strategy"] == "strategy"
    assert summary.iloc[0]["detections_completed"] == 1
    assert summary.iloc[0]["success_triggers"] == 1
    assert set(events["classification"]) == {"tp", "fp"}
