#!/usr/bin/env python3
"""Phase 5 simulation orchestrator for RQ3.

This script builds on the Phase 3 strategies and the minimal simulator to
produce per-project and aggregate metrics required in Phase 5 of the
implementation plan.  The workflow is:

1. Run all four scheduling strategies with a fixed risk threshold.
2. Join the schedules with detection-time baselines and build-frequency data.
3. Emit project-level metrics, overall summaries, and daily totals.
4. Generate a basic boxplot that visualises scheduled additional builds per
   project for quick comparisons across strategies.

Outputs are written under ``phase5_outputs/`` and include CSV tables plus a PNG
figure for the additional-build distribution.  The script also estimates wasted
builds (false positive triggers) by tracking cumulative additional builds
against per-project thresholds, emitting aggregate metrics to
``strategy_wasted_builds.csv`` and trigger-level details to
``strategy_wasted_build_events.csv``.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import math
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .phase3_strategies import RISK_COLUMN
    from .phase5_minimal_simulation import (
        load_detection_baseline,
        run_minimal_simulation,
        summarize_schedule_by_project,
    )
except ImportError:  # pragma: no cover - allows CLI execution within package dir
    from phase3_strategies import RISK_COLUMN
    from phase5_minimal_simulation import (
        load_detection_baseline,
        run_minimal_simulation,
        summarize_schedule_by_project,
    )


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "phase5_outputs")
DEFAULT_PREDICTIONS_ROOT = os.path.join(
    os.path.dirname(__file__), "../outputs", "results", "xgboost"
)
DEFAULT_DETECTION_TABLE = os.path.join(
    os.path.dirname(__file__), "../rq3_dataset", "detection_time_results.csv"
)
DEFAULT_BUILD_COUNTS_TABLE = os.path.join(
    os.path.dirname(__file__), "../rq3_dataset", "project_build_counts.csv"
)
DEFAULT_DETECTION_WINDOW_DAYS = 0


def _ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _load_detection_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    project_col = "project" if "project" in df.columns else "package_name"
    df["project"] = df[project_col].astype(str).str.strip()
    df = df[df["project"] != ""]
    if "detection_time_days" in df.columns:
        df["detection_time_days"] = pd.to_numeric(
            df["detection_time_days"], errors="coerce"
        )
    return df.dropna(subset=["project", "detection_time_days"])


def _load_build_counts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["project"] = df.get("project", "").astype(str).str.strip()
    df = df[df["project"] != ""]
    df["builds_per_day"] = pd.to_numeric(df.get("builds_per_day"), errors="coerce")
    return df.dropna(subset=["builds_per_day"])


def _normalize_to_date(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    timestamps = timestamps.dt.normalize()
    return timestamps.dt.tz_convert(None)


def _prepare_schedule_for_waste_analysis(df: pd.DataFrame) -> pd.DataFrame:
    # 各戦略のスケジュールを「プロジェクト×日付×追加ビルド数」の形式に正規化する
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

    schedule["schedule_date"] = _normalize_to_date(date_series)
    schedule["scheduled_additional_builds"] = pd.to_numeric(
        schedule.get("scheduled_additional_builds"), errors="coerce"
    ).fillna(0.0)
    schedule["project"] = schedule.get("project", "").astype(str).str.strip()
    return schedule.dropna(subset=["project", "schedule_date"])


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _baseline_detection_metrics(
    detection_df: pd.DataFrame, build_counts_df: pd.DataFrame
) -> pd.DataFrame:
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


def _build_threshold_map(baseline_df: pd.DataFrame) -> Dict[str, float]:
    """Derive per-project cumulative build thresholds for simulated detections."""

    thresholds: Dict[str, float] = {}
    for _, row in baseline_df.iterrows():
        project = str(row.get("project", "")).strip()
        if not project:
            continue

        baseline_builds = float(row.get("baseline_detection_builds", float("nan")))
        threshold = baseline_builds if math.isfinite(baseline_builds) and baseline_builds > 0 else float("nan")

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


def _prepare_project_metrics(
    schedules: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    project_frames: List[pd.DataFrame] = []
    for strategy_name, schedule in schedules.items():
        summary = summarize_schedule_by_project(strategy_name, schedule)
        if summary.empty:
            project_frames.append(summary)
            continue
        merged = summary.merge(baseline_df, on="project", how="left")
        merged.rename(
            columns={
                "rows": "trigger_count",
                "unique_days": "trigger_days",
                "scheduled_additional_builds": "scheduled_builds",
            },
            inplace=True,
        )
        merged["avg_builds_per_trigger"] = merged.apply(
            lambda row: row["scheduled_builds"] / row["trigger_count"]
            if row["trigger_count"]
            else 0.0,
            axis=1,
        )
        merged["builds_per_day"] = merged["builds_per_day"].fillna(0.0)
        merged["baseline_detection_builds"] = merged["baseline_detection_builds"].fillna(
            0.0
        )
        merged["baseline_detection_days"] = merged["baseline_detection_days"].fillna(0.0)

        def _estimate_residual_days(row: pd.Series) -> float:
            if row["builds_per_day"] <= 0:
                return row["baseline_detection_days"]
            delta_days = row["scheduled_builds"] / row["builds_per_day"]
            return max(row["baseline_detection_days"] - delta_days, 0.0)

        merged["estimated_detection_days"] = merged.apply(_estimate_residual_days, axis=1)
        merged["estimated_detection_builds"] = merged.apply(
            lambda row: max(row["baseline_detection_builds"] - row["scheduled_builds"], 0.0),
            axis=1,
        )
        project_frames.append(merged)

    if not project_frames:
        return pd.DataFrame()
    return pd.concat(project_frames, ignore_index=True, sort=False)


def _aggregate_strategy_metrics(project_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for strategy, group in project_df.groupby("strategy"):
        if group.empty:
            continue
        record = {
            "strategy": strategy,
            "projects": int(group["project"].nunique()),
            "total_scheduled_builds": float(group["scheduled_builds"].sum()),
            "median_scheduled_builds": float(group["scheduled_builds"].median()),
            "mean_scheduled_builds": float(group["scheduled_builds"].mean()),
            "median_estimated_detection_days": float(group["estimated_detection_days"].median()),
            "mean_estimated_detection_days": float(group["estimated_detection_days"].mean()),
            "median_estimated_detection_builds": float(group["estimated_detection_builds"].median()),
            "mean_estimated_detection_builds": float(group["estimated_detection_builds"].mean()),
            "total_estimated_builds_saved": float(
                (group["baseline_detection_builds"] - group["estimated_detection_builds"]).sum()
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def _prepare_daily_totals(schedules: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for strategy_name, schedule in schedules.items():
        if schedule.empty:
            continue
        daily = schedule.copy()
        date_column: Optional[str]
        if "merge_date_ts" in daily.columns:
            date_column = "merge_date_ts"
        elif "merge_date" in daily.columns:
            date_column = "merge_date"
        else:
            date_column = None

        if date_column is None:
            continue

        daily["date"] = pd.to_datetime(daily[date_column], errors="coerce")
        daily = daily.dropna(subset=["date"])
        grouped = (
            daily.groupby(["project", "date"])["scheduled_additional_builds"].sum()
            .reset_index()
        )
        grouped.insert(0, "strategy", strategy_name)
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["strategy", "project", "date", "scheduled_additional_builds"])
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["strategy", "project", "date"], inplace=True)
    return result

# TODO：レビューする，無駄ビルドの計算方法が妥当か
def _summarize_wasted_builds(
    schedules: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
    detection_window_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate detection outcomes using build-index thresholds.

    追加ビルドはイベント発生時にのみ仮想的に累積し、閾値到達の瞬間で検出評価する。
    検出成立後の追加ビルドは ``fp_post_detection`` として扱い、浪費量に反映する。
    """

    thresholds = _build_threshold_map(baseline_df)
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
        prepared = _prepare_schedule_for_waste_analysis(schedule)
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
                    classification = "fp_post_detection"
                    wasted = scheduled_builds
                elif not math.isfinite(threshold) or threshold == float("inf"):
                    classification = "fp"
                    wasted = scheduled_builds
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

        success_ratio = _safe_ratio(success_triggers, total_triggers)
        wasted_ratio = _safe_ratio(wasted_triggers, total_triggers)
        expired_ratio = _safe_ratio(expired_triggers, total_triggers)
        builds_success_ratio = _safe_ratio(builds_success, total_additional_builds)
        builds_wasted_ratio = _safe_ratio(builds_wasted, total_additional_builds)

        project_success_projects = {
            event["project"] for event in strategy_events if event["classification"] == "tp"
        }
        project_waste_projects = {
            event["project"]
            for event in strategy_events
            if event["classification"]
            in {"fp", "fp_post_detection", "expired", "baseline_only"}
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


def _plot_additional_builds_boxplot(project_df: pd.DataFrame, output_dir: str) -> Optional[str]:
    if project_df.empty:
        return None
    grouped: List[List[float]] = []
    labels: List[str] = []
    for name, group in project_df.groupby("strategy"):
        labels.append(name)
        grouped.append(group["scheduled_builds"].tolist())
    if not grouped or not labels:
        return None

    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped, labels=labels, showfliers=False)
    plt.ylabel("Scheduled Additional Builds per Project")
    plt.title("Additional Build Distribution by Strategy")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "additional_builds_boxplot.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 5 simulation orchestrator for RQ3 additional-build strategies."
    )
    parser.add_argument(
        "--predictions-root",
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Directory containing *_daily_aggregated_metrics_with_predictions.csv files.",
    )
    parser.add_argument(
        "--risk-column",
        default=RISK_COLUMN,
        help="Risk score column to use when thresholding predictions.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional precomputed label column to use instead of thresholding.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Risk score threshold used to filter predictions (default: 0.5).",
    )
    parser.add_argument(
        "--detection-table",
        default=DEFAULT_DETECTION_TABLE,
        help="Path to detection_time_results.csv for baseline detection metrics.",
    )
    parser.add_argument(
        "--build-counts",
        default=DEFAULT_BUILD_COUNTS_TABLE,
        help="Path to project_build_counts.csv for per-project build cadence.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where Phase 5 outputs will be written.",
    )
    parser.add_argument(
        "--detection-window-days",
        type=int,
        default=DEFAULT_DETECTION_WINDOW_DAYS,
        help="Detection contribution window in days (0 disables expiration).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress console summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = _ensure_directory(os.path.abspath(args.output_dir))

    detection_df = _load_detection_table(args.detection_table)
    build_counts_df = _load_build_counts(args.build_counts)
    baseline_df = _baseline_detection_metrics(detection_df, build_counts_df)

    # 検出ウィンドウは0日以上に丸め、誤った負値指定を回避する
    detection_window = max(int(args.detection_window_days), 0)

    simulation_result = run_minimal_simulation(
        predictions_root=args.predictions_root,
        risk_column=args.risk_column,
        label_column=args.label_column,
        risk_threshold=args.risk_threshold,
        return_details=True,
    )
    summary_df, schedules = simulation_result

    for key, value in load_detection_baseline(args.detection_table).items():
        summary_df[key] = value

    project_metrics = _prepare_project_metrics(schedules, baseline_df)
    aggregate_metrics = _aggregate_strategy_metrics(project_metrics)
    daily_totals = _prepare_daily_totals(schedules)

    wasted_summary, wasted_events = _summarize_wasted_builds(
        schedules,
        baseline_df,
        detection_window,
    )

    summary_path = os.path.join(output_dir, "strategy_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    project_path = os.path.join(output_dir, "strategy_project_metrics.csv")
    project_metrics.to_csv(project_path, index=False)

    aggregate_path = os.path.join(output_dir, "strategy_overall_metrics.csv")
    aggregate_metrics.to_csv(aggregate_path, index=False)

    daily_path = os.path.join(output_dir, "strategy_daily_totals.csv")
    daily_totals.to_csv(daily_path, index=False)

    wasted_path = os.path.join(output_dir, "strategy_wasted_builds.csv")
    wasted_summary.to_csv(wasted_path, index=False)

    wasted_events_path = os.path.join(output_dir, "strategy_wasted_build_events.csv")
    wasted_events.to_csv(wasted_events_path, index=False)

    plot_path = _plot_additional_builds_boxplot(project_metrics, output_dir)

    if not args.silent:
        print("=== Phase 5 Simulation Summary ===")
        print(summary_df.to_string(index=False))
        print(f"Summary written to: {summary_path}")
        print(f"Project metrics written to: {project_path}")
        print(f"Aggregate metrics written to: {aggregate_path}")
        print(f"Daily totals written to: {daily_path}")
        print(f"Wasted-build metrics written to: {wasted_path}")
        print(f"Wasted-build events written to: {wasted_events_path}")
        if plot_path:
            print(f"Boxplot saved to: {plot_path}")


if __name__ == "__main__":
    main()
