"""Timeline utilities for transforming daily metrics into build schedules."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence

PathLike = str | bytes | Path

FIELDNAMES = [
    "project",
    "merge_date",
    "day_index",
    "builds_per_day",
    "build_index_start",
    "build_index_end",
    "cumulative_builds",
    "daily_commit_count",
]


def scan_daily_records(csv_path: PathLike) -> List[Dict[str, object]]:
    """Read a daily metrics CSV and normalise rows for timeline computation."""

    records: List[Dict[str, object]] = []
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
                        daily_commit_count = None
            records.append(
                {
                    "merge_date": date_obj,
                    "raw_merge_date": date_str,
                    "daily_commit_count": daily_commit_count,
                }
            )
    records.sort(key=lambda rec: rec["merge_date"])
    return records


def build_timeline(
    project: str,
    records: Sequence[Dict[str, object]],
    builds_per_day: int,
) -> List[Dict[str, object]]:
    """Convert daily records into a sequential build timeline for a project."""

    timeline: List[Dict[str, object]] = []
    builds_per_day = max(builds_per_day, 0)
    cumulative_builds = 0

    for day_index, record in enumerate(records, start=1):
        daily_builds = builds_per_day
        build_index_start = cumulative_builds + 1 if daily_builds > 0 else cumulative_builds
        cumulative_builds += daily_builds
        build_index_end = cumulative_builds
        timeline.append(
            {
                "project": project,
                "merge_date": record["raw_merge_date"],
                "day_index": day_index,
                "builds_per_day": daily_builds,
                "build_index_start": build_index_start,
                "build_index_end": build_index_end,
                "cumulative_builds": cumulative_builds,
                "daily_commit_count": record.get("daily_commit_count"),
            }
        )

    return timeline


def summarise_project_timeline(
    project: str,
    timeline: Sequence[Dict[str, object]],
    builds_per_day: int,
) -> Dict[str, object]:
    """Create a lightweight summary row for the latest entry of a timeline."""

    if not timeline:
        return {
            "project": project,
            "merge_date": "",
            "day_index": 0,
            "builds_per_day": builds_per_day,
            "build_index_start": 0,
            "build_index_end": 0,
            "cumulative_builds": 0,
            "daily_commit_count": None,
        }

    first_entry = timeline[0]
    last_entry = timeline[-1]
    return {
        "project": project,
        "merge_date": last_entry["merge_date"],
        "day_index": last_entry["day_index"],
        "builds_per_day": builds_per_day,
        "build_index_start": first_entry["build_index_start"],
        "build_index_end": last_entry["build_index_end"],
        "cumulative_builds": last_entry["cumulative_builds"],
        "daily_commit_count": None,
    }


def write_timeline_csv(path: PathLike, rows: Iterable[Dict[str, object]]) -> None:
    """Write timeline rows to ``path`` using the canonical field order."""

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            output: MutableMapping[str, object] = dict(row)
            writer.writerow(output)
