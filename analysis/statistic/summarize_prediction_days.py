#!/usr/bin/env python3
"""Summarize daily VCC presence in prediction CSVs.

This script walks the modeling outputs stored under
`datasets/model_outputs/random_forest/<project>/` and, for every
`*_daily_aggregated_metrics_with_predictions.csv`, counts how many
calendar days are tracked in the file and how many of those days
contain an actual vulnerability-inducing commit (`is_vcc == 1`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ProjectStats:
    project: str
    file: Path
    total_days: int
    days_with_vcc: int
    days_without_vcc: int

    @property
    def vcc_ratio(self) -> float:
        return (self.days_with_vcc / self.total_days) if self.total_days else 0.0


def _default_root() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent
    return repo_root / "datasets" / "model_outputs" / "random_forest"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count total/VCC days per project from prediction CSVs."
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=_default_root(),
        help="Root directory that contains per-project prediction CSVs.",
    )
    parser.add_argument(
        "--pattern",
        default="*_daily_aggregated_metrics_with_predictions.csv",
        help="Glob pattern (relative to each project directory) used to match CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to store aggregated stats (JSON by default, CSV if suffix == .csv).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress warnings for malformed files.",
    )
    return parser.parse_args(argv)


def _parse_merge_date(raw: str) -> Optional[date]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    # Try ISO date first (YYYY-MM-DD)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value[:10], fmt).date()
        except ValueError:
            continue
    # Fallback: use fromisoformat (handles timezone suffixes)
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _parse_is_vcc(raw: str) -> int:
    if raw is None:
        return 0
    value = str(raw).strip()
    if not value:
        return 0
    truthy = {"1", "true", "True", "TRUE"}
    falsy = {"0", "false", "False", "FALSE"}
    if value in truthy:
        return 1
    if value in falsy:
        return 0
    try:
        return 1 if float(value) >= 1 else 0
    except ValueError:
        return 0


def _load_rows(csv_path: Path) -> Dict[date, int]:
    """Return mapping date -> is_vcc for the given CSV."""
    by_day: Dict[date, int] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or {"merge_date", "is_vcc"} - set(reader.fieldnames):
            raise ValueError("Required columns missing")
        for row in reader:
            merge_date = _parse_merge_date(row.get("merge_date", ""))
            if merge_date is None:
                continue
            is_vcc = _parse_is_vcc(row.get("is_vcc"))
            by_day[merge_date] = is_vcc  # keep last occurrence per day
    return by_day


def _analyze_file(path: Path, quiet: bool = False) -> Optional[ProjectStats]:
    try:
        day_map = _load_rows(path)
    except Exception as exc:  # noqa: BLE001
        if not quiet:
            print(f"[warn] Failed to parse {path}: {exc}", file=sys.stderr)
        return None

    if not day_map:
        if not quiet:
            print(f"[warn] No valid rows found in {path}", file=sys.stderr)
        return None

    total_days = len(day_map)
    days_with_vcc = sum(day_map.values())
    project_name = path.parent.name
    return ProjectStats(
        project=project_name,
        file=path,
        total_days=total_days,
        days_with_vcc=days_with_vcc,
        days_without_vcc=total_days - days_with_vcc,
    )


def _collect_prediction_files(root: Path, pattern: str) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {root}")
    files: List[Path] = []
    for project_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files.extend(sorted(project_dir.glob(pattern)))
    return files


def _print_table(rows: List[ProjectStats], summary: Dict[str, int]) -> None:
    if not rows:
        print("No prediction CSVs produced usable statistics.")
        return
    header = f"{'Project':20} {'TotalDays':>10} {'VCCDays':>10} {'NonVCC':>10} {'VCC%':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.project:20} {r.total_days:10d} {r.days_with_vcc:10d} "
            f"{r.days_without_vcc:10d} {r.vcc_ratio*100:7.2f}%"
        )
    print("-" * len(header))
    total = summary["total_days"]
    with_vcc = summary["days_with_vcc"]
    without_vcc = summary["days_without_vcc"]
    ratio = (with_vcc / total * 100.0) if total else 0.0
    print(f"{'TOTAL':20} {total:10d} {with_vcc:10d} {without_vcc:10d} {ratio:7.2f}%")


def _save_output(path: Path, rows: List[ProjectStats], summary: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["project", "file", "total_days", "days_with_vcc", "days_without_vcc", "vcc_ratio"]
            )
            for r in rows:
                writer.writerow(
                    [r.project, str(r.file), r.total_days, r.days_with_vcc, r.days_without_vcc, r.vcc_ratio]
                )
        return
    payload = {
        "projects": [asdict(r) | {"vcc_ratio": r.vcc_ratio} for r in rows],
        "summary": summary,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _aggregate(rows: Iterable[ProjectStats]) -> Dict[str, int]:
    rows = list(rows)
    return {
        "projects": len(rows),
        "total_days": sum(r.total_days for r in rows),
        "days_with_vcc": sum(r.days_with_vcc for r in rows),
        "days_without_vcc": sum(r.days_without_vcc for r in rows),
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        files = _collect_prediction_files(args.pred_root, args.pattern)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not files:
        print(f"No prediction CSVs found under {args.pred_root} using pattern '{args.pattern}'.")
        return 1

    rows: List[ProjectStats] = []
    for file_path in files:
        stats = _analyze_file(file_path, quiet=args.quiet)
        if stats:
            rows.append(stats)
    rows.sort(key=lambda r: r.project.lower())
    summary = _aggregate(rows)
    _print_table(rows, summary)

    if args.output:
        _save_output(args.output, rows, summary)
        print(f"\nSaved statistics to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
