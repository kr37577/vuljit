#!/usr/bin/env python3
"""Precision/threshold analysis for RQ3 additional-build strategies.

This script derives precision/recall curves for the JIT prediction outputs,
maps desired precision targets to risk score thresholds, and evaluates the
additional-build strategies under those thresholds to quantify added build load
and proxy detection speed measures.  The outputs are written to
`phase4_outputs/` in CSV/Markdown formats for inspection in later phases.
"""

from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .core import (
        ensure_directory,
        resolve_default,
        collect_predictions,
    )
except ImportError:  # pragma: no cover - allow direct CLI execution
    from core import (
        ensure_directory,
        resolve_default,
        collect_predictions,
    )

try:
    from .additional_build_strategies import (
        RISK_COLUMN,
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )
except ImportError:  # pragma: no cover - allow direct CLI execution
    from additional_build_strategies import (  # type: ignore
        RISK_COLUMN,
        strategy1_median_schedule,
        strategy2_random_within_median_range,
        strategy3_line_change_proportional,
        strategy4_cross_project_regression,
    )


DEFAULT_RISK_COLUMN = RISK_COLUMN
DEFAULT_LABEL_COLUMN = None
DEFAULT_PRECISION_TARGETS = (0.05, 0.07, 0.08, 0.09, 0.1)


@dataclass
class PrecisionRecallResult:
    threshold: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


def compute_precision_recall(df: pd.DataFrame, risk_column: str) -> List[PrecisionRecallResult]:
    scores = df[risk_column].to_numpy(dtype=float)
    labels = df["is_vcc"].to_numpy(dtype=bool)
    order = np.argsort(-scores)
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    tp = 0
    fp = 0
    fn_total = int(labels_sorted.sum())
    results: List[PrecisionRecallResult] = []
    visited_thresholds = set()

    for idx, (score, label) in enumerate(zip(scores_sorted, labels_sorted), start=1):
        if label:
            tp += 1
        else:
            fp += 1
        fn = fn_total - tp
        threshold = float(score)
        if threshold in visited_thresholds:
            continue
        visited_thresholds.add(threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        results.append(PrecisionRecallResult(threshold, precision, recall, tp, fp, fn))

    # Add baseline point at threshold > max score (recall 0).
    results.append(PrecisionRecallResult(
        threshold=float(scores_sorted.max()) + 1.0 if scores_sorted.size else 1.0,
        precision=1.0,
        recall=0.0,
        tp=0,
        fp=0,
        fn=fn_total,
    ))

    results.sort(key=lambda r: r.threshold, reverse=True)
    return results


def find_threshold_for_precision(
    pr_curve: Sequence[PrecisionRecallResult], target_precision: float
) -> Optional[PrecisionRecallResult]:
    for point in pr_curve:
        if point.precision >= target_precision:
            return point
    return None


def _summarize_strategy_frame(df: pd.DataFrame, value_column: str = "scheduled_additional_builds") -> Dict[str, float]:
    if df.empty:
        return {
            "rows": 0,
            "unique_projects": 0,
            "unique_days": 0,
            value_column: 0.0,
        }
    metrics: Dict[str, float] = {
        "rows": float(len(df)),
        "unique_projects": float(df["project"].nunique()),
        "unique_days": float(df["merge_date"].nunique() if "merge_date" in df.columns else df["merge_date_ts"].nunique()),
        value_column: float(df[value_column].fillna(0).sum()),
    }
    if "median_detection_days" in df.columns:
        metrics["median_detection_days_mean"] = float(df["median_detection_days"].dropna().mean())
    if "sampled_offset_days" in df.columns:
        metrics["sampled_offset_mean"] = float(df["sampled_offset_days"].dropna().mean())
    if "normalized_line_change" in df.columns:
        metrics["normalized_line_change_mean"] = float(df["normalized_line_change"].dropna().mean())
    if "predicted_additional_builds" in df.columns:
        metrics["predicted_additional_builds_sum"] = float(df["predicted_additional_builds"].fillna(0).sum())
    return metrics


def evaluate_strategies(
    thresholds: Dict[float, PrecisionRecallResult],
    predictions_root: str,
    risk_column: str,
    label_column: Optional[str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for precision_target, pr_point in thresholds.items():
        threshold = pr_point.threshold
        strat1 = strategy1_median_schedule(
            predictions_root=predictions_root,
            risk_column=risk_column,
            label_column=label_column,
            threshold=threshold,
        )
        strat2 = strategy2_random_within_median_range(
            predictions_root=predictions_root,
            risk_column=risk_column,
            label_column=label_column,
            threshold=threshold,
        )
        strat3 = strategy3_line_change_proportional(
            predictions_root=predictions_root,
            risk_column=risk_column,
            label_column=label_column,
            threshold=threshold,
        )
        strat4, _, _ = strategy4_cross_project_regression(
            predictions_root=predictions_root,
            risk_column=risk_column,
            label_column=label_column,
            threshold=threshold,
        )

        summaries = {
            "strategy1": _summarize_strategy_frame(strat1),
            "strategy2": _summarize_strategy_frame(strat2),
            "strategy3": _summarize_strategy_frame(strat3),
            "strategy4": _summarize_strategy_frame(strat4),
        }
        for strategy_name, summary in summaries.items():
            record: Dict[str, object] = {
                "precision_target": precision_target,
                "risk_threshold": threshold,
                "strategy": strategy_name,
                "precision": pr_point.precision,
                "recall": pr_point.recall,
                "tp": pr_point.tp,
                "fp": pr_point.fp,
                "fn": pr_point.fn,
            }
            record.update(summary)
            records.append(record)
    return pd.DataFrame(records)


def summarize_low_precision(
    pr_curve: Sequence[PrecisionRecallResult],
    strategy_summary: pd.DataFrame,
    low_precision: float,
) -> pd.DataFrame:
    point = find_threshold_for_precision(pr_curve, low_precision)
    if point is None:
        return pd.DataFrame()
    subset = strategy_summary[strategy_summary["precision_target"] == low_precision]
    subset = subset.copy()
    if subset.empty:
        return pd.DataFrame()
    subset["false_positive_rate"] = 1.0 - point.precision
    return subset


def write_precision_recall_outputs(
    output_dir: str,
    pr_curve: Sequence[PrecisionRecallResult],
    per_project_curves: Dict[str, Sequence[PrecisionRecallResult]],
) -> None:
    curve_df = pd.DataFrame(
        {
            "threshold": [p.threshold for p in pr_curve],
            "precision": [p.precision for p in pr_curve],
            "recall": [p.recall for p in pr_curve],
            "tp": [p.tp for p in pr_curve],
            "fp": [p.fp for p in pr_curve],
            "fn": [p.fn for p in pr_curve],
        }
    )
    curve_df.to_csv(os.path.join(output_dir, "precision_recall_aggregate.csv"), index=False)

    rows: List[Dict[str, object]] = []
    for project, curve in per_project_curves.items():
        for point in curve:
            rows.append(
                {
                    "project": project,
                    "threshold": point.threshold,
                    "precision": point.precision,
                    "recall": point.recall,
                    "tp": point.tp,
                    "fp": point.fp,
                    "fn": point.fn,
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "precision_recall_per_project.csv"), index=False
        )


def write_threshold_mapping(
    output_dir: str, thresholds: Dict[float, PrecisionRecallResult]
) -> None:
    rows = []
    for target, point in thresholds.items():
        rows.append(
            {
                "precision_target": target,
                "threshold": point.threshold,
                "recall": point.recall,
                "tp": point.tp,
                "fp": point.fp,
                "fn": point.fn,
            }
        )
    pd.DataFrame(rows).sort_values("precision_target").to_csv(
        os.path.join(output_dir, "precision_thresholds.csv"), index=False
    )


def write_low_precision_report(
    output_dir: str,
    low_precision_df: pd.DataFrame,
    precision_value: float,
) -> None:
    if low_precision_df.empty:
        return
    csv_path = os.path.join(output_dir, f"low_precision_{precision_value:.2f}_summary.csv")
    low_precision_df.to_csv(csv_path, index=False)

    md_path = os.path.join(output_dir, f"low_precision_{precision_value:.2f}_summary.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(f"# Low Precision Risk Summary (precision ≥ {precision_value:.2f})\n\n")
        fh.write(
            "This table captures the expected additional builds when operating at a low precision" "\n"
        )
        fh.write(
            "target. The `false_positive_rate` column approximates how many triggered alerts may be" "\n"
        )
        fh.write(
            "incorrect given the current predictions.\n\n"
        )
        fh.write(low_precision_df.to_markdown(index=False))
        fh.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 precision/threshold analysis for RQ3 strategies."
    )
    parser.add_argument(
        "--predictions-root",
        default=resolve_default("phase4.predictions_root"),
        help="Directory containing *_daily_aggregated_metrics_with_predictions.csv files.",
    )
    parser.add_argument(
        "--risk-column",
        default=DEFAULT_RISK_COLUMN,
        help="Risk score column to use for precision/recall computation.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help="Optional column containing precomputed predictions (boolean).",
    )
    parser.add_argument(
        "--precision-targets",
        default=",".join(str(x) for x in DEFAULT_PRECISION_TARGETS),
        help="Comma-separated precision targets to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=resolve_default("phase4.output_dir"),
        help="Directory where analysis outputs will be written.",
    )
    parser.add_argument(
        "--low-precision",
        type=float,
        default=0.5,
        help="Low precision level for risk summary (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    precision_targets = []
    for chunk in str(args.precision_targets).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            precision_targets.append(float(chunk))
        except ValueError:
            continue
    if not precision_targets:
        precision_targets = list(DEFAULT_PRECISION_TARGETS)

    predictions_df = collect_predictions(
        args.predictions_root, args.risk_column, args.label_column
    )

    aggregate_pr = compute_precision_recall(predictions_df, args.risk_column)
    per_project_curves: Dict[str, List[PrecisionRecallResult]] = {}
    for project, group in predictions_df.groupby("project"):
        per_project_curves[project] = compute_precision_recall(group, args.risk_column)

    write_precision_recall_outputs(output_dir, aggregate_pr, per_project_curves)

    threshold_map: Dict[float, PrecisionRecallResult] = {}
    for target in precision_targets:
        point = find_threshold_for_precision(aggregate_pr, target)
        if point is not None:
            threshold_map[target] = point
    if not threshold_map:
        raise RuntimeError("Unable to map any precision targets to thresholds.")

    write_threshold_mapping(output_dir, threshold_map)

    strategy_summary = evaluate_strategies(
        threshold_map,
        args.predictions_root,
        args.risk_column,
        args.label_column,
    )
    strategy_summary.to_csv(
        os.path.join(output_dir, "strategy_precision_sweep.csv"), index=False
    )

    low_precision_df = summarize_low_precision(
        aggregate_pr,
        strategy_summary,
        args.low_precision,
    )
    if not low_precision_df.empty:
        write_low_precision_report(output_dir, low_precision_df, args.low_precision)


if __name__ == "__main__":
    main()
