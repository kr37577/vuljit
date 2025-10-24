from __future__ import annotations

import pandas as pd

from RQ3.core.baseline import (
    DetectionTarget,
    baseline_detection_metrics,
    build_threshold_map,
    compute_detection_targets,
    group_detection_targets,
)


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


def test_compute_detection_targets_creates_vulnerability_entries() -> None:
    detection_df = pd.DataFrame(
        {
            "project": ["alpha", "alpha", "beta", ""],
            "detection_time_days": [2.0, 4.0, 3.0, 5.0],
            "monorail_id": ["100", "", "200", "unused"],
            "introduced_commits": ["abc123", "def456", "", ""],
        }
    )
    build_counts_df = pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "builds_per_day": [1.5, 0.0],
        }
    )

    targets = compute_detection_targets(detection_df, build_counts_df)
    assert len(targets) == 3
    assert targets[0] == DetectionTarget(
        project="alpha",
        vulnerability_id="100",
        detection_time_days=2.0,
        builds_per_day=1.5,
        baseline_detection_builds=3.0,
    )
    # second alpha record falls back to introduced commit string for ID
    assert targets[1].vulnerability_id == "introduced:def456"
    assert targets[1].baseline_detection_builds == 6.0
    # beta record falls back to detection days because build rate is zero
    assert targets[2].baseline_detection_builds == 3.0


def test_group_detection_targets_preserves_insertion_order() -> None:
    targets = [
        DetectionTarget("alpha", "v1", 1.0, 1.0, 1.0),
        DetectionTarget("alpha", "v2", 2.0, 1.0, 2.0),
        DetectionTarget("beta", "v3", 3.0, 2.0, 6.0),
    ]

    grouped = group_detection_targets(targets)
    assert list(grouped.keys()) == ["alpha", "beta"]
    assert [target.vulnerability_id for target in grouped["alpha"]] == ["v1", "v2"]
