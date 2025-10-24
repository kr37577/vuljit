from __future__ import annotations

import pandas as pd

from RQ3.core.baseline import DetectionTarget
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
    row = summary.iloc[0]
    assert row["strategy"] == "strategy"
    assert row["detections_completed"] == 1
    assert row["success_triggers"] == 1
    assert row["vulnerabilities_total"] == 1
    assert row["vulnerabilities_detected_additional"] == 1
    assert row["vulnerabilities_detected_baseline"] == 0
    assert row["vulnerabilities_wasted"] == 0
    assert set(events["classification"]) == {"tp", "fp", "baseline_only"}
    assert "baseline_detections_count" in events.columns


def test_summarize_wasted_builds_handles_multiple_vulnerabilities() -> None:
    schedules = {
        "strategy": pd.DataFrame(
            {
                "project": ["alpha", "alpha"],
                "merge_date_ts": ["2024-01-01", "2024-01-02"],
                "scheduled_additional_builds": [1.0, 2.0],
            }
        )
    }
    baseline_df = pd.DataFrame(
        {
            "project": ["alpha"],
            "baseline_detection_builds": [2.0],
            "baseline_detection_days": [2.0],
            "builds_per_day": [0.0],
        }
    )
    detection_targets = {
        "alpha": [
            DetectionTarget("alpha", "v1", 1.0, 1.0, 1.0),
            DetectionTarget("alpha", "v2", 2.0, 1.0, 2.0),
        ]
    }

    summary, events = summarize_wasted_builds(
        schedules,
        baseline_df,
        detection_window_days=0,
        detection_targets=detection_targets,
    )
    row = summary.iloc[0]
    assert row["vulnerabilities_total"] == 2
    assert row["vulnerabilities_detected_additional"] == 2
    assert row["vulnerabilities_detected_baseline"] == 0
    assert row["vulnerabilities_wasted"] == 0
    assert row["success_triggers"] == 2
    first_event = events.iloc[0]
    second_event = events.iloc[1]
    assert first_event["classification"] == "tp"
    assert first_event["vulnerability_ids"] == "v1"
    assert second_event["classification"] == "tp"
    assert second_event["vulnerability_ids"] == "v2"


def test_summarize_wasted_builds_marks_expiration_with_detection_window() -> None:
    schedules = {
        "strategy": pd.DataFrame(
            {
                "project": ["alpha", "alpha"],
                "merge_date_ts": ["2024-01-01", "2024-01-04"],
                "scheduled_additional_builds": [0.0, 0.0],
            }
        )
    }
    baseline_df = pd.DataFrame(
        {
            "project": ["alpha"],
            "baseline_detection_builds": [3.0],
            "baseline_detection_days": [3.0],
            "builds_per_day": [1.0],
        }
    )

    summary_no_window, events_no_window = summarize_wasted_builds(
        schedules,
        baseline_df,
        detection_window_days=0,
        detection_targets={
            "alpha": [
                DetectionTarget("alpha", "v1", 3.0, 1.0, 3.0),
            ]
        },
    )
    row_no_window = summary_no_window.iloc[0]
    assert row_no_window["detections_baseline_only"] == 1
    assert row_no_window["vulnerabilities_detected_baseline"] == 1
    assert row_no_window["expired_triggers"] == 0
    assert events_no_window.iloc[1]["classification"] == "baseline_only"
    assert events_no_window.iloc[1]["baseline_vulnerability_ids"] == "v1"

    summary_with_window, events_with_window = summarize_wasted_builds(
        schedules,
        baseline_df,
        detection_window_days=1,
        detection_targets={
            "alpha": [
                DetectionTarget("alpha", "v1", 3.0, 1.0, 3.0),
            ]
        },
    )
    row_with_window = summary_with_window.iloc[0]
    assert row_with_window["detections_baseline_only"] == 0
    assert row_with_window["vulnerabilities_detected_baseline"] == 0
    assert row_with_window["vulnerabilities_wasted"] == 1
    assert row_with_window["expired_triggers"] == 1
    assert events_with_window.iloc[1]["classification"] == "expired"
    assert events_with_window.iloc[1]["expired"] is True
    assert events_with_window.iloc[1]["baseline_vulnerability_ids"] == ""
