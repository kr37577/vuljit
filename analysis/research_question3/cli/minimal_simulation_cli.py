"""CLI entry point for the minimal additional-build simulation."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import pandas as pd

from ..core import ensure_directory, ensure_parent_directory, resolve_default
from ..core.metrics import summarize_schedule_by_project
from ..core.simulation import SimulationResult, run_minimal_simulation
from ..additional_build_strategies import RISK_COLUMN

__all__ = [
    "load_detection_baseline",
    "parse_args",
    "main",
]


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
        "baseline_detection_days_mean": float(
            df["detection_time_days"].astype(float).mean()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal simulation runner for RQ3 additional-build strategies."
    )
    def _default_proj_mode(env_key: str) -> str:
        value = os.environ.get(env_key, "per_project")
        normalized = value.strip().lower()
        return normalized if normalized in {"per_project", "cross_project"} else "per_project"

    default_strategy_modes = {
        "strategy1": _default_proj_mode("RQ3_STRATEGY1_MODE"),
        "strategy2": _default_proj_mode("RQ3_STRATEGY2_MODE"),
        "strategy3": _default_proj_mode("RQ3_STRATEGY3_MODE"),
    }
    default_strategy4_mode = os.environ.get("RQ3_STRATEGY4_MODE", "multi").strip().lower()
    if default_strategy4_mode not in {"multi", "simple"}:
        default_strategy4_mode = "multi"
    parser.add_argument(
        "--predictions-root",
        default=resolve_default("phase5_minimal.predictions_root"),
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
        "--strategy1-mode",
        choices=("per_project", "cross_project"),
        default=default_strategy_modes["strategy1"],
        help="Scheduling mode for Strategy1 (default: env RQ3_STRATEGY1_MODE or 'per_project').",
    )
    parser.add_argument(
        "--strategy2-mode",
        choices=("per_project", "cross_project"),
        default=default_strategy_modes["strategy2"],
        help="Scheduling mode for Strategy2 (default: env RQ3_STRATEGY2_MODE or 'per_project').",
    )
    parser.add_argument(
        "--strategy3-mode",
        choices=("per_project", "cross_project"),
        default=default_strategy_modes["strategy3"],
        help="Scheduling mode for Strategy3 (default: env RQ3_STRATEGY3_MODE or 'per_project').",
    )
    parser.add_argument(
        "--strategy3-global-budget",
        type=float,
        default=None,
        help="Override global additional-build budget used by Strategy3 cross-project mode.",
    )
    parser.add_argument(
        "--strategy4-mode",
        choices=("multi", "simple"),
        default=default_strategy4_mode,
        help="Regression mode for Strategy4 (default: env RQ3_STRATEGY4_MODE or 'multi').",
    )
    parser.add_argument(
        "--strategy4-feature",
        dest="strategy4_features",
        action="append",
        default=None,
        help="Append a feature column for Strategy4 regression (repeatable).",
    )
    parser.add_argument(
        "--detection-table",
        default=resolve_default("phase5.detection_table"),
        help="Path to detection_time_results.csv for baseline statistics.",
    )
    parser.add_argument(
        "--output",
        default=resolve_default("phase5_minimal.output"),
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

    strategy_overrides: Dict[str, Dict[str, object]] = {}
    if args.strategy1_mode:
        strategy_overrides.setdefault("strategy1_median", {})["mode"] = args.strategy1_mode
    if args.strategy2_mode:
        strategy_overrides.setdefault("strategy2_random", {})["mode"] = args.strategy2_mode
    if args.strategy3_mode:
        strategy_overrides.setdefault("strategy3_line_proportional", {})["mode"] = args.strategy3_mode
    if args.strategy3_global_budget is not None:
        strategy_overrides.setdefault("strategy3_line_proportional", {})["global_budget"] = float(args.strategy3_global_budget)

    strategy4_features = None
    if args.strategy4_features:
        strategy4_features = [feature.strip() for feature in args.strategy4_features if feature and feature.strip()]
        if not strategy4_features:
            strategy4_features = None
    strategy4_kwargs: Dict[str, object] = {}
    if args.strategy4_mode:
        strategy4_kwargs["mode"] = args.strategy4_mode
    if strategy4_features:
        strategy4_kwargs["feature_cols"] = tuple(strategy4_features)
    if strategy4_kwargs:
        strategy_overrides["strategy4_regression"] = strategy4_kwargs

    simulation_result: SimulationResult = run_minimal_simulation(
        predictions_root=args.predictions_root,
        risk_column=args.risk_column,
        label_column=args.label_column,
        risk_threshold=args.risk_threshold,
        strategy_overrides=strategy_overrides or None,
        return_details=need_details,
    )

    summary_df = simulation_result.summary.copy()
    schedules = simulation_result.schedules if need_details else {}

    baseline = load_detection_baseline(args.detection_table)
    for key, value in baseline.items():
        summary_df[key] = value

    output_path = ensure_parent_directory(args.output)
    summary_df.to_csv(output_path, index=False)

    schedules_dir: Optional[str] = None
    if args.schedules_dir:
        schedules_dir = ensure_directory(args.schedules_dir)
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
        project_output = ensure_parent_directory(args.project_summary_output)
        project_summary_df.to_csv(project_output, index=False)
    else:
        project_output = None

    if not args.silent:
        print("=== Minimal Additional-Build Simulation Summary ===")
        print(summary_df.to_string(index=False))
        print(f"Results saved to: {output_path}")
        if schedules_dir:
            print(f"Schedule CSVs saved under: {schedules_dir}")
        if project_output:
            print(f"Per-project summary saved to: {project_output}")
