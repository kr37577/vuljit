"""Baseline detection utilities shared across RQ3 simulations."""

from __future__ import annotations

import math
from typing import Dict

import pandas as pd

__all__ = [
    "baseline_detection_metrics",
    "build_threshold_map",
]


def baseline_detection_metrics(
    detection_df: pd.DataFrame, build_counts_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-project baseline detection metrics fused with build cadence."""

    baseline_days = (
        detection_df.groupby("project")["detection_time_days"]
        .median()
        .rename("baseline_detection_days")
    )
    merged = baseline_days.to_frame().reset_index()
    merged = merged.merge(build_counts_df, on="project", how="left")
    merged["builds_per_day"] = merged["builds_per_day"].fillna(0.0)
    merged["baseline_detection_builds"] = (
        merged["baseline_detection_days"].fillna(0.0) * merged["builds_per_day"]
    )
    return merged


def build_threshold_map(baseline_df: pd.DataFrame) -> Dict[str, float]:
    """Derive per-project cumulative build thresholds for simulated detections."""

    thresholds: Dict[str, float] = {}
    for _, row in baseline_df.iterrows():
        project = str(row.get("project", "")).strip()
        if not project:
            continue

        baseline_builds = float(row.get("baseline_detection_builds", float("nan")))
        threshold = (
            baseline_builds
            if math.isfinite(baseline_builds) and baseline_builds > 0
            else float("nan")
        )

        if not math.isfinite(threshold) or threshold <= 0:
            baseline_days = float(row.get("baseline_detection_days", float("nan")))
            builds_per_day = float(row.get("builds_per_day", float("nan")))

            candidates = [
                baseline_builds,
                baseline_days * builds_per_day
                if math.isfinite(baseline_days) and math.isfinite(builds_per_day)
                else float("nan"),
                baseline_days,
            ]
            threshold = next(
                (value for value in candidates if math.isfinite(value) and value > 0),
                float("inf"),
            )

        if not math.isfinite(threshold) or threshold <= 0:
            threshold = float("inf")

        thresholds[project] = threshold

    return thresholds

