"""CLI entry point for the additional-build simulation orchestrator."""

from __future__ import annotations

import argparse
import os
from typing import Optional

from ..core import baseline_detection_metrics, ensure_directory, load_build_counts, load_detection_table, resolve_default
from ..core.metrics import aggregate_strategy_metrics, prepare_daily_totals, prepare_project_metrics
from ..core.plotting import plot_additional_builds_boxplot
from ..core.simulation import SimulationResult, run_minimal_simulation, summarize_wasted_builds
from ..additional_build_strategies import RISK_COLUMN
from .minimal_simulation_cli import load_detection_baseline

__all__ = ["parse_args", "main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate additional-build simulations for RQ3 strategies.")
    parser.add_argument(
        "--predictions-root",
        default=resolve_default("phase5.predictions_root"),
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
        default=resolve_default("phase5.detection_table"),
        help="Path to detection_time_results.csv for baseline detection metrics.",
    )
    parser.add_argument(
        "--build-counts",
        default=resolve_default("phase5.build_counts"),
        help="Path to project_build_counts.csv for per-project build cadence.",
    )
    parser.add_argument(
        "--output-dir",
        default=resolve_default("phase5.output_dir"),
        help="Directory where additional-build simulation outputs will be written.",
    )
    parser.add_argument(
        "--detection-window-days",
        type=int,
        default=resolve_default("phase5.detection_window_days"),
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
    output_dir = ensure_directory(args.output_dir)

    detection_df = load_detection_table(args.detection_table)
    build_counts_df = load_build_counts(args.build_counts)
    baseline_df = baseline_detection_metrics(detection_df, build_counts_df)

    detection_window = max(int(args.detection_window_days), 0)

    simulation_result: SimulationResult = run_minimal_simulation(
        predictions_root=args.predictions_root,
        risk_column=args.risk_column,
        label_column=args.label_column,
        risk_threshold=args.risk_threshold,
        return_details=True,
    )
    summary_df = simulation_result.summary.copy()
    schedules = simulation_result.schedules

    for key, value in load_detection_baseline(args.detection_table).items():
        summary_df[key] = value

    project_metrics = prepare_project_metrics(schedules, baseline_df)
    aggregate_metrics = aggregate_strategy_metrics(project_metrics)
    daily_totals = prepare_daily_totals(schedules)

    wasted_summary, wasted_events = summarize_wasted_builds(
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

    plot_path: Optional[str] = plot_additional_builds_boxplot(project_metrics, output_dir)

    if not args.silent:
        print("=== Additional-Build Simulation Summary ===")
        print(summary_df.to_string(index=False))
        print(f"Summary written to: {summary_path}")
        print(f"Project metrics written to: {project_path}")
        print(f"Aggregate metrics written to: {aggregate_path}")
        print(f"Daily totals written to: {daily_path}")
        print(f"Wasted-build metrics written to: {wasted_path}")
        print(f"Wasted-build events written to: {wasted_events_path}")
        if plot_path:
            print(f"Boxplot saved to: {plot_path}")
