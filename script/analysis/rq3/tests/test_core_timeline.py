from __future__ import annotations

import csv
from pathlib import Path

from RQ3.core.timeline import (
    FIELDNAMES,
    build_timeline,
    scan_daily_records,
    summarise_project_timeline,
    write_timeline_csv,
)


def test_scan_daily_records_parses_dates(tmp_path) -> None:
    csv_path = tmp_path / "daily.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["merge_date", "daily_commit_count"])
        writer.writeheader()
        writer.writerow({"merge_date": "2024-01-02", "daily_commit_count": "3"})
    records = scan_daily_records(csv_path)
    assert len(records) == 1
    assert records[0]["daily_commit_count"] == 3


def test_build_timeline_generates_cumulative_values() -> None:
    records = [
        {"raw_merge_date": "2024-01-01", "daily_commit_count": 1},
        {"raw_merge_date": "2024-01-02", "daily_commit_count": 2},
    ]
    timeline = build_timeline("alpha", records, builds_per_day=2)
    assert timeline[-1]["cumulative_builds"] == 4


def test_summarise_project_timeline_handles_empty() -> None:
    summary = summarise_project_timeline("project", [], builds_per_day=1)
    assert summary["project"] == "project"
    assert summary["cumulative_builds"] == 0


def test_write_timeline_csv_persists_rows(tmp_path) -> None:
    rows = [
        {
            "project": "alpha",
            "merge_date": "2024-01-01",
            "day_index": 1,
            "builds_per_day": 2,
            "build_index_start": 1,
            "build_index_end": 2,
            "cumulative_builds": 2,
            "daily_commit_count": 3,
        }
    ]
    path = tmp_path / "timeline.csv"
    write_timeline_csv(path, rows)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == FIELDNAMES
        loaded = next(reader)
        assert loaded["project"] == "alpha"
