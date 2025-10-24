#!/usr/bin/env python3
"""Generate publication-ready tables from strategy_wasted_builds.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "simulation_outputs" / "strategy_wasted_builds.csv"

STRATEGY_LABELS = {
    "strategy1_median": "S1 Median",
    "strategy2_random": "S2 IQR Random",
    "strategy3_line_proportional": "S3 Line-Proportional",
    "strategy4_regression": "S4 Regression",
}


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["strategy"].isin(STRATEGY_LABELS)].copy()
    if df.empty:
        raise ValueError("No recognised strategies found in the metrics file.")
    return df


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    baseline_builds = float(df.loc[df["strategy"] == "strategy1_median", "additional_builds_total"].iloc[0])

    def _pct(series: pd.Series) -> pd.Series:
        return series.astype(float) * 100

    summary = pd.DataFrame(
        {
            "Strategy": df["strategy"].map(STRATEGY_LABELS),
            "Success rate (%)": _pct(df["success_ratio"]),
            "Success triggers": df["success_triggers"].astype(int).astype(str)
            + " / "
            + df["triggers_total"].astype(int).astype(str),
            "Additional builds (k)": df["additional_builds_total"] / 1_000.0,
            "Success build share (%)": _pct(df["builds_success_ratio"]),
            "Wasted build share (%)": _pct(df["builds_wasted_ratio"]),
            "Build reduction vs S1 (%)": 100.0 * (1.0 - df["additional_builds_total"] / baseline_builds),
        }
    )

    summary = summary.sort_values("Strategy").reset_index(drop=True)
    numeric_cols = [
        "Success rate (%)",
        "Additional builds (k)",
        "Success build share (%)",
        "Wasted build share (%)",
        "Build reduction vs S1 (%)",
    ]
    summary[numeric_cols] = summary[numeric_cols].round(2)
    return summary


def format_output(table: pd.DataFrame) -> tuple[str, str]:
    headers = list(table.columns)
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    markdown = "\n".join(lines)
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lrrrrrr",
        float_format="%.2f",
    )
    return markdown, latex


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready tables highlighting strategy efficiency.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to strategy_wasted_builds.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex", "both"),
        default="both",
        help="Output format to print (default: both).",
    )
    args = parser.parse_args()

    metrics = load_metrics(args.input)
    summary = prepare_summary(metrics)
    markdown, latex = format_output(summary)

    if args.format in ("markdown", "both"):
        print("# Strategy Efficiency Summary (Markdown)")
        print(markdown)
        print()
    if args.format in ("latex", "both"):
        print("% Strategy Efficiency Summary (LaTeX)")
        print(latex)


if __name__ == "__main__":
    main()
