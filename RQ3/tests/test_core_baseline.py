from __future__ import annotations

import pandas as pd

from RQ3.core.baseline import baseline_detection_metrics, build_threshold_map


def test_baseline_detection_metrics_computes_derived_columns() -> None:
    detection_df = pd.DataFrame(
        {
            "project": ["alpha", "alpha", "beta"],
            "detection_time_days": [2.0, 4.0, 3.0],
        }
    )
    build_counts_df = pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "builds_per_day": [1.5, 2.0],
        }
    )

    baseline_df = baseline_detection_metrics(detection_df, build_counts_df)
    alpha_row = baseline_df.loc[baseline_df["project"] == "alpha"].iloc[0]
    assert alpha_row["baseline_detection_days"] == 3.0
    assert alpha_row["builds_per_day"] == 1.5
    assert alpha_row["baseline_detection_builds"] == 4.5


def test_build_threshold_map_prefers_positive_candidates() -> None:
    baseline_df = pd.DataFrame(
        {
            "project": ["alpha", "beta", "gamma"],
            "baseline_detection_builds": [5.0, float("nan"), float("nan")],
            "baseline_detection_days": [2.0, 3.0, float("nan")],
            "builds_per_day": [2.5, 1.0, 0.0],
        }
    )

    thresholds = build_threshold_map(baseline_df)
    assert thresholds["alpha"] == 5.0
    assert thresholds["beta"] == 3.0  # fallback to days * builds_per_day
    assert thresholds["gamma"] == float("inf")
