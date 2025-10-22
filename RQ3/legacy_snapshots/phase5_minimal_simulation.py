#!/usr/bin/env python3
"""Minimal Phase 5 simulation runner for RQ3 strategies.

This script is a lightweight wrapper around the Phase 3 strategy helpers. It
accepts a fixed risk-score threshold (instead of deriving it from a precision
analysis) and replays the four additional-build strategies to provide quick
summaries of the scheduled builds.  The goal is to unblock experimentation
while `phase4_analysis.py` is under review.

Outputs: a CSV (and optional console table) with aggregate metrics such as
total additional builds, covered projects, and the number of days where
additional builds are triggered. Optional flags export raw per-day schedules
and project-level aggregates for deeper inspection.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

try:
    from .phase3_strategies import (
        RISK_COLUMN,
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )
except ImportError:  # pragma: no cover - allow direct CLI execution
    from phase3_strategies import (
        RISK_COLUMN,
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )


DEFAULT_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "phase5_minimal_summary.csv")
DEFAULT_PREDICTIONS_ROOT = os.path.join(
    os.path.dirname(__file__), "../outputs", "results", "xgboost"
)


def summarize_schedule(name: str, df: pd.DataFrame, value_column: str = "scheduled_additional_builds") -> Dict[str, object]:
    """Convert a strategy schedule into a single-row metric summary."""

    if df.empty:
        return {
            "strategy": name,
            "rows": 0,
            "unique_projects": 0,
            "unique_days": 0,
            value_column: 0,
        }

    summary: Dict[str, object] = {
        "strategy": name,
        "rows": int(len(df)),
        "unique_projects": int(df["project"].nunique()) if "project" in df.columns else 0,
        "unique_days": int(df["merge_date"].nunique())
        if "merge_date" in df.columns
        else int(df["merge_date_ts"].nunique()),
        value_column: float(df[value_column].fillna(0).sum()) if value_column in df.columns else 0.0,
    }

    if "median_detection_days" in df.columns:
        summary["median_detection_days_mean"] = float(df["median_detection_days"].dropna().mean())
    if "sampled_offset_days" in df.columns:
        summary["sampled_offset_mean"] = float(df["sampled_offset_days"].dropna().mean())
    if "normalized_line_change" in df.columns:
        summary["normalized_line_change_mean"] = float(df["normalized_line_change"].dropna().mean())
    if "predicted_additional_builds" in df.columns:
        summary["predicted_additional_builds_sum"] = float(df["predicted_additional_builds"].fillna(0).sum())

    return summary


def summarize_schedule_by_project(
    name: str,
    df: pd.DataFrame,
    value_column: str = "scheduled_additional_builds",
) -> pd.DataFrame:
    """Aggregate a strategy schedule per project."""

    columns = ["strategy", "project", "rows", "unique_days", value_column]
    if df.empty or "project" not in df.columns:
        return pd.DataFrame(columns=columns)

    day_column: Optional[str]
    if "merge_date" in df.columns:
        day_column = "merge_date"
    elif "merge_date_ts" in df.columns:
        day_column = "merge_date_ts"
    else:
        day_column = None

    grouped = df.groupby("project", dropna=False)
    result = grouped.size().rename("rows").to_frame()

    if day_column is not None:
        unique_days = grouped[day_column].nunique(dropna=True).rename("unique_days")
        result = result.join(unique_days, how="left")
    else:
        result["unique_days"] = 0

    if value_column in df.columns:
        totals = grouped[value_column].sum(min_count=1).rename(value_column)
        result = result.join(totals, how="left")
    else:
        result[value_column] = 0.0

    if "median_detection_days" in df.columns:
        result["median_detection_days_mean"] = grouped["median_detection_days"].mean()
    if "sampled_offset_days" in df.columns:
        result["sampled_offset_mean"] = grouped["sampled_offset_days"].mean()
    if "normalized_line_change" in df.columns:
        result["normalized_line_change_mean"] = grouped["normalized_line_change"].mean()
    if "predicted_additional_builds" in df.columns:
        total_predicted = grouped["predicted_additional_builds"].sum(min_count=1)
        result["predicted_additional_builds_sum"] = total_predicted

    result = result.reset_index()
    result.insert(0, "strategy", name)

    result["rows"] = result["rows"].astype(int)
    result["unique_days"] = result["unique_days"].fillna(0).astype(int)
    result[value_column] = result[value_column].fillna(0).astype(float)
    for col in (
        "median_detection_days_mean",
        "sampled_offset_mean",
        "normalized_line_change_mean",
        "predicted_additional_builds_sum",
    ):
        if col in result.columns:
            result[col] = result[col].astype(float)

    ordered_cols = columns + [col for col in result.columns if col not in columns]
    return result[ordered_cols]


def load_detection_baseline(path: str) -> Dict[str, float]:
    """Provide simple baseline statistics from detection_time_results.csv."""

    df = pd.read_csv(path)
    df = df.dropna(subset=["detection_time_days"])
    if df.empty:
        return {
            "baseline_records": 0,
            "baseline_detection_days_mean": float("nan"),
        }
    return {
        "baseline_records": int(len(df)),
        "baseline_detection_days_mean": float(df["detection_time_days"].astype(float).mean()),
    }


def run_minimal_simulation(
    predictions_root: str,
    risk_column: str,
    label_column: Optional[str],
    risk_threshold: float,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute all four strategies with the provided threshold.

    When ``return_details`` is ``True`` a tuple of ``(summary_df, schedules)`` is
    returned so callers can inspect the raw per-day schedules per strategy.
    ``schedules`` maps strategy names to their respective DataFrames.
    """

    strat1 = strategy1_median_schedule(
        predictions_root=predictions_root,
        risk_column=risk_column,
        label_column=label_column,
        threshold=risk_threshold,
    )
    strat2 = strategy2_random_within_median_range(
        predictions_root=predictions_root,
        risk_column=risk_column,
        label_column=label_column,
        threshold=risk_threshold,
    )
    strat3 = strategy3_line_change_proportional(
        predictions_root=predictions_root,
        risk_column=risk_column,
        label_column=label_column,
        threshold=risk_threshold,
    )
    strat4, _, _ = strategy4_cross_project_regression(
        predictions_root=predictions_root,
        risk_column=risk_column,
        label_column=label_column,
        threshold=risk_threshold,
    )

    schedules: Dict[str, pd.DataFrame] = {
        "strategy1_median": strat1,
        "strategy2_random": strat2,
        "strategy3_line_proportional": strat3,
        "strategy4_regression": strat4,
    }

    summaries: List[Dict[str, object]] = [
        summarize_schedule(name, df) for name, df in schedules.items()
    ]

    summary_df = pd.DataFrame(summaries)
    if return_details:
        return summary_df, schedules
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal simulation runner for RQ3 additional-build strategies."
    )
    parser.add_argument(
        "--predictions-root",
        default=DEFAULT_PREDICTIONS_ROOT,
        help="Root directory containing *_daily_aggregated_metrics_with_predictions.csv files.",
    )
    parser.add_argument(
        "--risk-column",
        default=RISK_COLUMN,
        help="Prediction column containing risk scores (default: RISK_COLUMN).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional boolean prediction column. If omitted, the risk column is thresholded.",
    )
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.5,
        help="Risk score threshold used to trigger additional builds (default: 0.5).",
    )
    parser.add_argument(
        "--detection-table",
        default=os.path.join(os.path.dirname(__file__), "../rq3_dataset", "detection_time_results.csv"),
        help="Path to detection_time_results.csv for baseline statistics.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="CSV file where the summary metrics will be written.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="If set, do not print the summary table to stdout.",
    )
    parser.add_argument(
        "--schedules-dir",
        default=None,
        help="Directory where raw per-day schedules will be exported as CSV files.",
    )
    parser.add_argument(
        "--project-summary-output",
        default=None,
        help="Optional CSV path for per-project aggregates derived from each strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    need_details = bool(args.schedules_dir or args.project_summary_output)

    simulation_result = run_minimal_simulation(
        predictions_root=args.predictions_root,
        risk_column=args.risk_column,
        label_column=args.label_column,
        risk_threshold=args.risk_threshold,
        return_details=need_details,
    )

    if need_details:
        summary_df, schedules = simulation_result
    else:
        summary_df = simulation_result
        schedules = {}

    baseline = load_detection_baseline(args.detection_table)
    for key, value in baseline.items():
        summary_df[key] = value

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    if args.schedules_dir:
        schedules_dir = os.path.abspath(args.schedules_dir)
        os.makedirs(schedules_dir, exist_ok=True)
        for name, frame in schedules.items():
            schedule_path = os.path.join(schedules_dir, f"{name}_schedule.csv")
            frame.to_csv(schedule_path, index=False)

    if args.project_summary_output:
        project_frames: List[pd.DataFrame] = []
        for name, frame in schedules.items():
            project_frames.append(summarize_schedule_by_project(name, frame))
        project_summary_df = (
            pd.concat(project_frames, ignore_index=True)
            if project_frames
            else summarize_schedule_by_project("strategy", pd.DataFrame())
        )
        project_output = os.path.abspath(args.project_summary_output)
        os.makedirs(os.path.dirname(project_output) or ".", exist_ok=True)
        project_summary_df.to_csv(project_output, index=False)

    if not args.silent:
        print("=== Minimal Simulation Summary ===")
        print(summary_df.to_string(index=False))
        print(f"Results saved to: {output_path}")
        if args.schedules_dir:
            print(f"Schedule CSVs saved under: {os.path.abspath(args.schedules_dir)}")
        if args.project_summary_output:
            print(f"Per-project summary saved to: {project_output}")


if __name__ == "__main__":
    main()
