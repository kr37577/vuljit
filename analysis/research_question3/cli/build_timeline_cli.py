"""CLI entry point for generating project build timelines."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[1]
    parent_dir = project_root.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    try:
        from analysis.research_question3.core import (
            build_timeline,
            ensure_directory,
            normalise_path,
            parse_build_counts_csv,
            resolve_default,
            scan_daily_records,
            summarise_project_timeline,
            write_timeline_csv,
        )
    except ImportError:
        from RQ3.core import (  # type: ignore[attr-defined]
            build_timeline,
            ensure_directory,
            normalise_path,
            parse_build_counts_csv,
            resolve_default,
            scan_daily_records,
            summarise_project_timeline,
            write_timeline_csv,
        )
else:
    from ..core import (
        build_timeline,
        ensure_directory,
        normalise_path,
        parse_build_counts_csv,
        resolve_default,
        scan_daily_records,
        summarise_project_timeline,
        write_timeline_csv,
    )

__all__ = ["parse_args", "main"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert daily project metrics into build timelines with sequential indices."
    )
    parser.add_argument(
        "--data-dir",
        default=resolve_default("timeline.data_dir"),
        help="Directory containing <project>/<project>_daily_aggregated_metrics.csv files (default: datasets/derived_artifacts).",
    )
    parser.add_argument(
        "--build-counts",
        default=resolve_default("timeline.build_counts"),
        help="Path to project_build_counts.csv (default: datasets/raw/rq3_dataset/project_build_counts.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=resolve_default("timeline.output_dir"),
        help="Destination directory for generated timeline CSV files.",
    )
    parser.add_argument(
        "--default-builds-per-day",
        type=int,
        default=resolve_default("timeline.default_builds_per_day"),
        help="Fallback builds_per_day value when a project is missing in the build counts table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_counts = parse_build_counts_csv(args.build_counts)
    default_builds = int(args.default_builds_per_day)

    data_dir = normalise_path(args.data_dir)
    output_dir = ensure_directory(args.output_dir)

    csv_pattern = os.path.join(data_dir, "*", "*_daily_aggregated_metrics.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise SystemExit(f"No daily aggregated metrics found under {data_dir!r}.")

    summary_rows: List[Dict[str, object]] = []
    for csv_path in csv_files:
        project = os.path.basename(os.path.dirname(csv_path))
        records = scan_daily_records(csv_path)
        if not records:
            continue
        builds_per_day = build_counts.get(project, default_builds)
        timeline_rows = build_timeline(project, records, builds_per_day)
        if not timeline_rows:
            continue
        project_output = os.path.join(output_dir, f"{project}_build_timeline.csv")
        write_timeline_csv(project_output, timeline_rows)
        summary_rows.append(
            summarise_project_timeline(project, timeline_rows, builds_per_day)
        )

    if summary_rows:
        summary_path = os.path.join(output_dir, "projects_summary.csv")
        write_timeline_csv(summary_path, summary_rows)
