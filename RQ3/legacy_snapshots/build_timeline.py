#!/usr/bin/env python3
"""Generate per-project build timelines with sequential build indices and fuzzing run counts.

The script consumes the daily aggregated project metrics (RQ1/RQ2 outputs) and the
`project_build_counts.csv` table that lists how many OSS-Fuzz builds are scheduled per day
for each project. For every project we emit a CSV that aligns calendar days (merge_date)
with build indices and cumulative fuzzing execution counts, providing a unified timeline
for downstream RQ3 simulations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional


@dataclass
class DailyRecord:
    merge_date: datetime
    raw_merge_date: str
    daily_commit_count: Optional[int]


@dataclass
class TimelineRow:
    project: str
    merge_date: str
    day_index: int
    builds_per_day: int
    build_index_start: int
    build_index_end: int
    cumulative_builds: int
    fuzzing_runs_daily: float
    fuzzing_runs_cumulative: float
    daily_commit_count: Optional[int]


def _parse_build_counts(path: str, default_builds: int) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            project = (row.get("project") or "").strip()
            if not project:
                continue
            value_raw = (row.get("builds_per_day") or "").strip()
            try:
                value = int(float(value_raw)) if value_raw else default_builds
            except ValueError:
                value = default_builds
            mapping[project] = max(value, 0)
    return mapping


def _scan_daily_records(csv_path: str) -> List[DailyRecord]:
    records: List[DailyRecord] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "merge_date" not in reader.fieldnames:
            return records
        for row in reader:
            date_str = (row.get("merge_date") or "").strip()
            if not date_str:
                continue
            date_obj: Optional[datetime] = None
            for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
                try:
                    if fmt is None:
                        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    else:
                        date_obj = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            if date_obj is None:
                continue
            daily_commit_count: Optional[int] = None
            if "daily_commit_count" in reader.fieldnames:
                raw_daily_commits = (row.get("daily_commit_count") or "").strip()
                if raw_daily_commits:
                    try:
                        daily_commit_count = int(float(raw_daily_commits))
                    except ValueError:
                        pass
            records.append(
                DailyRecord(
                    merge_date=date_obj,
                    raw_merge_date=date_str,
                    daily_commit_count=daily_commit_count,
                )
            )
    records.sort(key=lambda rec: rec.merge_date)
    return records


def _compute_timeline(
    project: str,
    records: Iterable[DailyRecord],
    builds_per_day: int,
    fuzz_multiplier: float,
) -> List[TimelineRow]:
    timeline: List[TimelineRow] = []
    if builds_per_day < 0:
        builds_per_day = 0
    cumulative_builds = 0
    cumulative_fuzz = 0.0
    for day_index, record in enumerate(records, start=1):
        daily_builds = builds_per_day
        build_index_start = cumulative_builds + 1 if daily_builds > 0 else cumulative_builds
        cumulative_builds += daily_builds
        build_index_end = cumulative_builds
        fuzz_daily = daily_builds * fuzz_multiplier
        cumulative_fuzz += fuzz_daily
        timeline.append(
            TimelineRow(
                project=project,
                merge_date=record.raw_merge_date,
                day_index=day_index,
                builds_per_day=daily_builds,
                build_index_start=build_index_start,
                build_index_end=build_index_end,
                cumulative_builds=cumulative_builds,
                fuzzing_runs_daily=fuzz_daily,
                fuzzing_runs_cumulative=cumulative_fuzz,
                daily_commit_count=record.daily_commit_count,
            )
        )
    return timeline


def _write_timeline_csv(path: str, rows: Iterable[TimelineRow]) -> None:
    fieldnames = [
        "project",
        "merge_date",
        "day_index",
        "builds_per_day",
        "build_index_start",
        "build_index_end",
        "cumulative_builds",
        "fuzzing_runs_daily",
        "fuzzing_runs_cumulative",
        "daily_commit_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "project": row.project,
                    "merge_date": row.merge_date,
                    "day_index": row.day_index,
                    "builds_per_day": row.builds_per_day,
                    "build_index_start": row.build_index_start,
                    "build_index_end": row.build_index_end,
                    "cumulative_builds": row.cumulative_builds,
                    "fuzzing_runs_daily": f"{row.fuzzing_runs_daily:.6f}",
                    "fuzzing_runs_cumulative": f"{row.fuzzing_runs_cumulative:.6f}",
                    "daily_commit_count": (
                        row.daily_commit_count if row.daily_commit_count is not None else ""
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert per-project daily metrics into build timelines with sequential indices."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Directory containing <project>/<project>_daily_aggregated_metrics.csv files (default: ../data).",
    )
    parser.add_argument(
        "--build-counts",
        default=os.path.join(os.path.dirname(__file__), "..", "rq3_dataset", "project_build_counts.csv"),
        help="Path to project_build_counts.csv (default: ../rq3_dataset/project_build_counts.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "phase2_outputs", "build_timelines"),
        help="Destination directory for generated timeline CSV files.",
    )
    parser.add_argument(
        "--default-builds-per-day",
        type=int,
        default=1,
        help="Fallback builds_per_day value when a project is missing in the build counts table.",
    )
    parser.add_argument(
        "--fuzzing-multiplier",
        type=float,
        default=1.0,
        help="Estimated number of fuzzing executions per build (default: 1.0).",
    )
    args = parser.parse_args()

    build_counts = _parse_build_counts(args.build_counts, args.default_builds_per_day)

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    csv_pattern = os.path.join(data_dir, "*", "*_daily_aggregated_metrics.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise SystemExit(f"No daily aggregated metrics found under {data_dir!r}.")

    summary_rows: List[TimelineRow] = []
    for csv_path in csv_files:
        project = os.path.basename(os.path.dirname(csv_path))
        records = _scan_daily_records(csv_path)
        if not records:
            continue
        builds_per_day = build_counts.get(project, args.default_builds_per_day)
        timeline = _compute_timeline(project, records, builds_per_day, args.fuzzing_multiplier)
        if not timeline:
            continue
        project_output = os.path.join(output_dir, f"{project}_build_timeline.csv")
        _write_timeline_csv(project_output, timeline)
        summary_rows.append(
            TimelineRow(
                project=project,
                merge_date=timeline[-1].merge_date,
                day_index=timeline[-1].day_index,
                builds_per_day=builds_per_day,
                build_index_start=timeline[0].build_index_start,
                build_index_end=timeline[-1].build_index_end,
                cumulative_builds=timeline[-1].cumulative_builds,
                fuzzing_runs_daily=timeline[-1].fuzzing_runs_daily,
                fuzzing_runs_cumulative=timeline[-1].fuzzing_runs_cumulative,
                daily_commit_count=None,
            )
        )

    if summary_rows:
        summary_path = os.path.join(output_dir, "projects_summary.csv")
        _write_timeline_csv(summary_path, summary_rows)


if __name__ == "__main__":
    main()
