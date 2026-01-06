"""Baseline detection utilities shared across RQ3 simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

__all__ = [
    "DetectionTarget",
    "baseline_detection_metrics",
    "build_threshold_map",
    "compute_detection_targets",
    "group_detection_targets",
]


@dataclass(frozen=True)
class DetectionTarget:
    """Representation of a single vulnerability detection threshold."""

    project: str
    vulnerability_id: str
    detection_time_days: float
    builds_per_day: float
    baseline_detection_builds: float

    @property
    def has_positive_threshold(self) -> bool:
        return math.isfinite(self.baseline_detection_builds) and self.baseline_detection_builds > 0


def baseline_detection_metrics(
    detection_df: pd.DataFrame, build_counts_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute per-project baseline detection metrics fused with build cadence."""

    build_rate_map = {
        str(row.get("project", "")).strip(): float(row.get("builds_per_day", float("nan")))
        for _, row in build_counts_df.iterrows()
        if str(row.get("project", "")).strip()
    }
    frame = detection_df.copy()
    frame["project"] = frame.get("project", "").astype(str).str.strip()
    frame["detection_time_days"] = pd.to_numeric(frame.get("detection_time_days"), errors="coerce")
    frame = frame.dropna(subset=["project", "detection_time_days"])
    frame["builds_per_day"] = frame["project"].map(build_rate_map).astype(float)
    frame["detection_time_builds"] = (
        frame["detection_time_days"] * frame["builds_per_day"]
    )
    frame.loc[frame["builds_per_day"] <= 0, "detection_time_builds"] = frame["detection_time_days"]

    baseline_builds = (
        frame.groupby("project")["detection_time_builds"]
        .median()
        .rename("baseline_detection_builds")
    )
    merged = baseline_builds.to_frame().reset_index()
    merged = merged.merge(build_counts_df, on="project", how="left")
    merged["builds_per_day"] = merged["builds_per_day"].fillna(0.0)
    merged["baseline_detection_builds"] = merged["baseline_detection_builds"].fillna(0.0)
    merged["baseline_detection_days"] = merged.apply(
        lambda row: row["baseline_detection_builds"] / row["builds_per_day"]
        if row["builds_per_day"] > 0
        else row["baseline_detection_builds"],
        axis=1,
    )
    return merged


def _resolve_vulnerability_id(row: pd.Series, fallback_index: int) -> str:
    """Resolve a stable identifier for vulnerability records."""

    for key in ("monorail_id", "OSV_id", "vulnerability_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not pd.isna(value):
            return str(value)

    introduced = row.get("introduced_commits")
    if isinstance(introduced, str) and introduced.strip():
        return f"introduced:{introduced.strip()}"

    return f"row_{fallback_index}"


def _coerce_detection_days(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or numeric < 0:
        return None
    return numeric


def compute_detection_targets(
    detection_df: pd.DataFrame, build_counts_df: pd.DataFrame
) -> List[DetectionTarget]:
    """Return per-vulnerability detection targets aligned with build cadence."""

    if detection_df.empty:
        return []

    if "project" not in detection_df.columns:
        raise KeyError("detection_df must contain a 'project' column")

    build_rates = {
        str(row["project"]).strip(): float(row["builds_per_day"])
        for _, row in build_counts_df.iterrows()
        if str(row.get("project", "")).strip()
    }

    targets: List[DetectionTarget] = []
    for index, row in detection_df.reset_index(drop=True).iterrows():
        project = str(row.get("project", "")).strip()
        if not project:
            continue

        detection_days = _coerce_detection_days(row.get("detection_time_days"))
        if detection_days is None:
            continue

        vulnerability_id = _resolve_vulnerability_id(row, index)
        builds_per_day = build_rates.get(project, 0.0)

        if math.isfinite(builds_per_day) and builds_per_day > 0:
            baseline_builds = detection_days * builds_per_day
        else:
            baseline_builds = detection_days

        targets.append(
            DetectionTarget(
                project=project,
                vulnerability_id=vulnerability_id,
                detection_time_days=detection_days,
                builds_per_day=builds_per_day,
                baseline_detection_builds=float(baseline_builds),
            )
        )

    return targets


def group_detection_targets(targets: Iterable[DetectionTarget]) -> Dict[str, List[DetectionTarget]]:
    """Group detection targets by project preserving original order."""

    grouped: Dict[str, List[DetectionTarget]] = {}
    for target in targets:
        grouped.setdefault(target.project, []).append(target)
    return grouped


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
