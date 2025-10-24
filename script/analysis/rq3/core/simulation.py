"""Simulation helpers that orchestrate Phase 3 strategies."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .baseline import DetectionTarget
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


def _sort_targets(targets: Iterable[DetectionTarget]) -> List[DetectionTarget]:
    """Return detection targets sorted by cumulative build threshold."""

    sortable: List[DetectionTarget] = list(targets)
    sortable.sort(key=lambda target: target.baseline_detection_builds)
    return sortable


def _fallback_targets_from_baseline(baseline_df: pd.DataFrame) -> Dict[str, List[DetectionTarget]]:
    """Create single-entry detection targets from aggregated baseline metrics."""

    targets: Dict[str, List[DetectionTarget]] = {}
    for _, row in baseline_df.iterrows():
        project = str(row.get("project", "")).strip()
        if not project:
            continue
        builds_per_day = float(row.get("builds_per_day", 0.0))
        detection_days = float(row.get("baseline_detection_days", float("nan")))
        baseline_builds_raw = float(row.get("baseline_detection_builds", float("nan")))

        if math.isfinite(baseline_builds_raw) and baseline_builds_raw > 0:
            baseline_builds = baseline_builds_raw
        elif math.isfinite(detection_days) and detection_days > 0:
            if math.isfinite(builds_per_day) and builds_per_day > 0:
                baseline_builds = detection_days * builds_per_day
            else:
                baseline_builds = detection_days
        else:
            baseline_builds = float("inf")

        targets[project] = [
            DetectionTarget(
                project=project,
                vulnerability_id=f"{project}#baseline",
                detection_time_days=detection_days,
                builds_per_day=builds_per_day,
                baseline_detection_builds=baseline_builds if math.isfinite(baseline_builds) else float("inf"),
            )
        ]
    return targets


def summarize_wasted_builds(
    schedules: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    detection_window_days: int,
    detection_targets: Optional[Dict[str, List[DetectionTarget]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate detection outcomes using per-vulnerability thresholds."""

    detection_window = (
        pd.Timedelta(days=detection_window_days) if detection_window_days > 0 else None
    )
    EPS = 1e-9

    baseline_rates = {
        str(row.get("project", "")).strip(): float(row.get("builds_per_day", 0.0))
        for _, row in baseline_df.iterrows()
    }
    if detection_targets is None:
        detection_targets_map = {
            project: _sort_targets(targets)
            for project, targets in _fallback_targets_from_baseline(baseline_df).items()
        }
    else:
        detection_targets_map = {
            project: _sort_targets(targets) for project, targets in detection_targets.items()
        }

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
                    "baseline_only_triggers": 0,
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
                    "vulnerabilities_total": 0,
                    "vulnerabilities_detected_additional": 0,
                    "vulnerabilities_detected_baseline": 0,
                    "vulnerabilities_wasted": 0,
                }
            )
            continue

        prepared = prepared.sort_values(["project", "schedule_date"]).reset_index(drop=True)
        prepared["scheduled_additional_builds"] = prepared["scheduled_additional_builds"].astype(float)

        total_additional_builds = float(prepared["scheduled_additional_builds"].sum())
        strategy_event_start = len(event_records)

        strategy_detections_completed = 0
        strategy_detections_additional = 0
        strategy_detections_baseline = 0
        strategy_vulnerabilities_total = 0
        strategy_vulnerabilities_remaining = 0

        for project, group in prepared.groupby("project"):
            baseline_rate = baseline_rates.get(project, 0.0)
            baseline_progress = 0.0
            baseline_history: Deque[Tuple[pd.Timestamp, float]] = deque()
            project_targets_all = detection_targets_map.get(project, [])
            project_targets: List[DetectionTarget] = list(project_targets_all)
            project_has_targets = bool(project_targets_all)
            project_total_targets = len(project_targets_all)
            project_detection_counter = 0
            last_date: Optional[pd.Timestamp] = None

            group = group.sort_values("schedule_date").reset_index(drop=True)

            for _, row in group.iterrows():
                schedule_date: pd.Timestamp = row["schedule_date"]
                scheduled_builds = float(row["scheduled_additional_builds"])

                baseline_detected_ids: List[str] = []

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

                expired_due_to_window = False
                if detection_window is not None and pd.notna(schedule_date):
                    cutoff = schedule_date.normalize() - detection_window
                    while baseline_history and baseline_history[0][0] < cutoff:
                        _, contribution = baseline_history.popleft()
                        baseline_progress = max(baseline_progress - contribution, 0.0)
                        expired_due_to_window = True

                while project_targets and baseline_progress + EPS >= project_targets[0].baseline_detection_builds:
                    target = project_targets.pop(0)
                    baseline_detected_ids.append(target.vulnerability_id)
                    project_detection_counter += 1
                    strategy_detections_completed += 1
                    strategy_detections_baseline += 1

                current_threshold = (
                    project_targets[0].baseline_detection_builds if project_targets else float("inf")
                )

                classification = "pending"
                success = False
                consumed = 0.0
                wasted = scheduled_builds
                detected_ids: List[str] = []
                evaluation_baseline = baseline_progress
                evaluation_trial = baseline_progress + scheduled_builds

                if not project_targets:
                    if project_has_targets:
                        classification = "baseline_only"
                    else:
                        classification = "fp"
                else:
                    progress_for_event = baseline_progress
                    remaining_builds = scheduled_builds

                    while project_targets and remaining_builds + progress_for_event + EPS >= project_targets[0].baseline_detection_builds:
                        target = project_targets[0]
                        required = max(target.baseline_detection_builds - progress_for_event, 0.0)
                        if required <= EPS:
                            project_targets.pop(0)
                            baseline_detected_ids.append(target.vulnerability_id)
                            project_detection_counter += 1
                            strategy_detections_completed += 1
                            strategy_detections_baseline += 1
                            progress_for_event = max(progress_for_event, target.baseline_detection_builds)
                            continue
                        if required > remaining_builds + EPS:
                            break
                        # Ensure numerical stability for very small requirements
                        actual_required = min(required, remaining_builds)
                        remaining_builds -= actual_required
                        consumed += actual_required
                        progress_for_event = max(progress_for_event, target.baseline_detection_builds)
                        detected_ids.append(target.vulnerability_id)
                        project_targets.pop(0)
                        project_detection_counter += 1
                        strategy_detections_completed += 1
                        strategy_detections_additional += 1
                        success = True

                    wasted = remaining_builds
                    if success:
                        classification = "tp"
                    elif project_has_targets and not project_targets and baseline_detected_ids:
                        classification = "baseline_only"
                    else:
                        classification = "fp"

                event_expired = False
                if (
                    detection_window is not None
                    and classification == "fp"
                    and project_has_targets
                    and expired_due_to_window
                ):
                    classification = "expired"
                    event_expired = True

                event_records.append(
                    {
                        "strategy": strategy_name,
                        "project": project,
                        "schedule_date": schedule_date,
                        "threshold": current_threshold,
                        "scheduled_builds": scheduled_builds,
                        "consumed_builds": consumed,
                        "wasted_within_event": wasted,
                        "success": success,
                        "expired": event_expired,
                        "detection_id": project_detection_counter if success else None,
                        "classification": classification,
                        "evaluation_baseline": evaluation_baseline,
                        "evaluation_trial": evaluation_trial,
                        "vulnerability_ids": ",".join(detected_ids),
                        "baseline_vulnerability_ids": ",".join(baseline_detected_ids),
                        "detections_count": len(detected_ids),
                        "baseline_detections_count": len(baseline_detected_ids),
                    }
                )

                last_date = schedule_date

            strategy_vulnerabilities_total += project_total_targets
            project_remaining_targets = len(project_targets)
            strategy_vulnerabilities_remaining += project_remaining_targets

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
                "detections_completed": strategy_detections_completed,
                "detections_with_additional": strategy_detections_additional,
                "detections_baseline_only": strategy_detections_baseline,
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
                "vulnerabilities_total": strategy_vulnerabilities_total,
                "vulnerabilities_detected_additional": strategy_detections_additional,
                "vulnerabilities_detected_baseline": strategy_detections_baseline,
                "vulnerabilities_wasted": strategy_vulnerabilities_remaining,
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
                "vulnerability_ids",
                "baseline_vulnerability_ids",
                "detections_count",
                "baseline_detections_count",
            ]
        )
    return summary_df, events_df
