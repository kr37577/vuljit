"""Simulation helpers that orchestrate Phase 3 strategies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

from .metrics import safe_ratio, summarize_schedule
from .scheduling import iter_strategies, run_strategy

__all__ = [
    "SimulationResult",
    "run_minimal_simulation",
    "normalize_to_date",
    "prepare_schedule_for_waste_analysis",
    "summarize_wasted_builds",
]


@dataclass
class SimulationResult:
    """Container for simulation outputs."""

    summary: pd.DataFrame
    schedules: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


def run_minimal_simulation(
    *,
    predictions_root: str,
    risk_column: str,
    label_column: Optional[str],
    risk_threshold: float,
    value_column: str = "scheduled_additional_builds",
    return_details: bool = False,
) -> SimulationResult:
    """Execute all registered strategies with the provided threshold."""

    schedule_map: Dict[str, pd.DataFrame] = {}
    summary_rows = []

    for name, strategy in iter_strategies():
        frame = run_strategy(
            name,
            predictions_root=predictions_root,
            risk_column=risk_column,
            label_column=label_column,
            threshold=risk_threshold,
        )
        if return_details:
            schedule_map[name] = frame
        summary_rows.append(summarize_schedule(name, frame, value_column=value_column))

    summary_df = pd.DataFrame(summary_rows)
    return SimulationResult(summary=summary_df, schedules=schedule_map)


def normalize_to_date(series: pd.Series) -> pd.Series:
    """Normalise timestamps to naive dates for schedule aggregation."""

    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    timestamps = timestamps.dt.normalize()
    return timestamps.dt.tz_convert(None)


def prepare_schedule_for_waste_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare per-strategy schedules for wasted-build evaluation."""

    if df.empty:
        return df
    schedule = df.copy()
    if "merge_date_ts" in schedule.columns:
        date_series = schedule["merge_date_ts"]
    elif "merge_date" in schedule.columns:
        date_series = schedule["merge_date"]
    else:
        schedule["schedule_date"] = pd.NaT
        return schedule

    schedule["schedule_date"] = normalize_to_date(date_series)
    schedule["scheduled_additional_builds"] = pd.to_numeric(
        schedule.get("scheduled_additional_builds"), errors="coerce"
    ).fillna(0.0)
    schedule["project"] = schedule.get("project", "").astype(str).str.strip()
    return schedule.dropna(subset=["project", "schedule_date"])


def summarize_wasted_builds(
    schedules: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    detection_window_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate detection outcomes using build-index thresholds."""

    from .baseline import build_threshold_map

    thresholds = build_threshold_map(baseline_df)
    baseline_rates = {
        str(row.get("project", "")).strip(): float(row.get("builds_per_day", 0.0))
        for _, row in baseline_df.iterrows()
    }
    detection_window = (
        pd.Timedelta(days=detection_window_days) if detection_window_days > 0 else None
    )
    EPS = 1e-9

    summary_records: List[Dict[str, object]] = []
    event_records: List[Dict[str, object]] = []

    for strategy_name, schedule in schedules.items():
        prepared = prepare_schedule_for_waste_analysis(schedule)
        if prepared.empty:
            summary_records.append(
                {
                    "strategy": strategy_name,
                    "detection_window_days": detection_window_days,
                    "triggers_total": 0,
                    "success_triggers": 0,
                    "wasted_triggers": 0,
                    "expired_triggers": 0,
                    "detections_completed": 0,
                    "detections_with_additional": 0,
                    "detections_baseline_only": 0,
                    "success_ratio": float("nan"),
                    "wasted_ratio": float("nan"),
                    "expired_ratio": float("nan"),
                    "additional_builds_total": 0.0,
                    "builds_success": 0.0,
                    "builds_wasted": 0.0,
                    "builds_success_ratio": float("nan"),
                    "builds_wasted_ratio": float("nan"),
                    "projects": 0,
                    "projects_with_success": 0,
                    "projects_with_waste": 0,
                }
            )
            continue

        prepared = prepared.sort_values(["project", "schedule_date"]).reset_index(drop=True)
        prepared["scheduled_additional_builds"] = prepared["scheduled_additional_builds"].astype(float)

        total_additional_builds = float(prepared["scheduled_additional_builds"].sum())
        strategy_event_start = len(event_records)

        detections_completed = 0
        detections_with_additional = 0
        detections_baseline_only = 0

        for project, group in prepared.groupby("project"):
            threshold = thresholds.get(project, float("inf"))
            if not math.isfinite(threshold) or threshold <= 0:
                threshold = float("inf")

            baseline_rate = baseline_rates.get(project, 0.0)
            baseline_progress = 0.0
            baseline_history: Deque[Tuple[pd.Timestamp, float]] = deque()
            detected_state: Optional[str] = None  # None, "baseline", "additional"

            group = group.sort_values("schedule_date").reset_index(drop=True)
            last_date: Optional[pd.Timestamp] = None

            for _, row in group.iterrows():
                schedule_date: pd.Timestamp = row["schedule_date"]
                scheduled_builds = float(row["scheduled_additional_builds"])

                if (
                    baseline_rate > 0
                    and last_date is not None
                    and pd.notna(schedule_date)
                    and pd.notna(last_date)
                ):
                    delta = schedule_date.normalize() - last_date.normalize()
                    if isinstance(delta, pd.Timedelta):
                        delta_days = delta.days
                        for offset in range(1, max(delta_days, 0) + 1):
                            day_stamp = last_date.normalize() + pd.Timedelta(days=offset)
                            baseline_progress += baseline_rate
                            if detection_window is not None:
                                baseline_history.append((day_stamp, baseline_rate))

                if detection_window is not None and pd.notna(schedule_date):
                    cutoff = schedule_date.normalize() - detection_window
                    while baseline_history and baseline_history[0][0] < cutoff:
                        _, contribution = baseline_history.popleft()
                        baseline_progress = max(baseline_progress - contribution, 0.0)

                if (
                    detected_state is None
                    and math.isfinite(threshold)
                    and threshold < float("inf")
                    and baseline_progress + EPS >= threshold
                ):
                    detected_state = "baseline"
                    detections_completed += 1
                    detections_baseline_only += 1

                classification = "pending"
                success = False
                consumed = 0.0
                wasted = 0.0
                evaluation_baseline = baseline_progress
                evaluation_trial = baseline_progress + scheduled_builds

                if detected_state == "baseline":
                    classification = "baseline_only"
                    wasted = scheduled_builds
                elif detected_state == "additional":
                    classification = "fp"
                    wasted = scheduled_builds
                #     classification = "fp_post_detection"
                #     wasted = scheduled_builds
                # elif not math.isfinite(threshold) or threshold == float("inf"):
                #     classification = "fp"
                #     wasted = scheduled_builds
                else:
                    if evaluation_trial + EPS >= threshold:
                        required = max(threshold - baseline_progress, 0.0)
                        consumed = min(max(required, 0.0), scheduled_builds)
                        wasted = max(scheduled_builds - consumed, 0.0)
                        classification = "tp"
                        success = True
                        detections_completed += 1
                        detections_with_additional += 1
                        detected_state = "additional"
                    else:
                        classification = "fp"
                        wasted = scheduled_builds

                event_records.append(
                    {
                        "strategy": strategy_name,
                        "project": project,
                        "schedule_date": schedule_date,
                        "threshold": threshold,
                        "scheduled_builds": scheduled_builds,
                        "consumed_builds": consumed,
                        "wasted_within_event": wasted,
                        "success": success,
                        "expired": False,
                        "detection_id": detections_completed if success else None,
                        "classification": classification,
                        "evaluation_baseline": evaluation_baseline,
                        "evaluation_trial": evaluation_trial,
                    }
                )

                last_date = schedule_date

        strategy_events = event_records[strategy_event_start:]

        baseline_only_triggers = sum(
            1 for event in strategy_events if event["classification"] == "baseline_only"
        )
        in_scope_events = [
            event
            for event in strategy_events
            if event["classification"] not in {"baseline_only"}
        ]

        total_triggers = len(in_scope_events)
        success_triggers = sum(1 for event in in_scope_events if event["classification"] == "tp")
        expired_triggers = sum(1 for event in in_scope_events if event["classification"] == "expired")
        wasted_triggers = total_triggers - success_triggers

        builds_success = float(sum(event["consumed_builds"] for event in strategy_events))
        builds_wasted = float(
            sum(event["wasted_within_event"] for event in strategy_events)
        )

        success_ratio = safe_ratio(success_triggers, total_triggers)
        wasted_ratio = safe_ratio(wasted_triggers, total_triggers)
        expired_ratio = safe_ratio(expired_triggers, total_triggers)
        builds_success_ratio = safe_ratio(builds_success, total_additional_builds)
        builds_wasted_ratio = safe_ratio(builds_wasted, total_additional_builds)

        project_success_projects = {
            event["project"] for event in strategy_events if event["classification"] == "tp"
        }
        project_waste_projects = {
            event["project"]
            for event in strategy_events
            if event["classification"]
            in {"fp", "expired", "baseline_only"}
            # {"fp", "fp_post_detection", "expired", "baseline_only"}
        }

        summary_records.append(
            {
                "strategy": strategy_name,
                "detection_window_days": detection_window_days,
                "triggers_total": total_triggers,
                "success_triggers": success_triggers,
                "wasted_triggers": wasted_triggers,
                "expired_triggers": expired_triggers,
                "baseline_only_triggers": baseline_only_triggers,
                "detections_completed": detections_completed,
                "detections_with_additional": detections_with_additional,
                "detections_baseline_only": detections_baseline_only,
                "success_ratio": success_ratio,
                "wasted_ratio": wasted_ratio,
                "expired_ratio": expired_ratio,
                "additional_builds_total": total_additional_builds,
                "builds_success": builds_success,
                "builds_wasted": builds_wasted,
                "builds_success_ratio": builds_success_ratio,
                "builds_wasted_ratio": builds_wasted_ratio,
                "projects": int(prepared["project"].nunique()),
                "projects_with_success": len(project_success_projects),
                "projects_with_waste": len(project_waste_projects),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    events_df = pd.DataFrame(event_records)
    if events_df.empty:
        events_df = pd.DataFrame(
            columns=[
                "strategy",
                "project",
                "schedule_date",
                "threshold",
                "scheduled_builds",
                "consumed_builds",
                "wasted_within_event",
                "success",
                "expired",
                "detection_id",
                "classification",
                "evaluation_baseline",
                "evaluation_trial",
            ]
        )
    return summary_df, events_df
