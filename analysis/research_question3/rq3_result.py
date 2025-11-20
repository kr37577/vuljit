#!/usr/bin/env python3
"""Generate aggregate RQ3 results table.

Reads ``strategy_wasted_builds.csv`` from the combined simulation outputs and
emits ``rq3_result.csv`` with the requested columns:
- Strategy
- Detection vulnerability rate (%)
- Detection vulnerability trigger
- Triggers total
- Additional builds total
- Builds per detection
"""

from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "derived_artifacts" / "rq3" / "simulation_outputs_strategy_combined"
INPUT_CSV = DATA_DIR / "strategy_wasted_builds.csv"
OUTPUT_CSV = DATA_DIR / "rq3_result.csv"


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    rows = []
    with INPUT_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            detections = int(float(row["detections_with_additional"]))
            additional_builds = float(row["additional_builds_total"])
            builds_per_detection = additional_builds / detections if detections else 0.0
            rows.append(
                {
                    "Strategy": row["strategy"],
                    "Detection vulnerability rate (%)": float(row["detection_vulnerability_rate"]) * 100,
                    "Detection vulnerability trigger": int(float(row["detection_vulnerability_trigger"])),
                    "Triggers total": int(float(row["triggers_total"])),
                    "Additional builds total": additional_builds,
                    "Builds per detection": builds_per_detection,
                }
            )

    fieldnames = [
        "Strategy",
        "Detection vulnerability rate (%)",
        "Detection vulnerability trigger",
        "Triggers total",
        "Additional builds total",
        "Builds per detection",
    ]
    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
