#!/usr/bin/env python3
"""Generate publication-ready RQ3 efficiency table.

Reads ``strategy_wasted_builds.csv`` and emits ``rq3_result.csv`` with columns:
- Strategy (labelled S1–S5)
- Total Cost (Builds) -> additional_builds_total
- Found vulnerabilities -> vulnerabilities_detected_additional
- Effective Trigger Rate (%) -> detection_vulnerability_rate (as percent)
- Cost per vulnerability -> Total Cost / Found vulnerabilities
"""

from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "derived_artifacts" / "rq3" / "simulation_outputs_strategy_combined"
INPUT_CSV = DATA_DIR / "strategy_wasted_builds.csv"
OUTPUT_CSV = DATA_DIR / "rq3_result.csv"

STRATEGY_LABELS = {
    "strategy1_median": "S1 Median",
    "strategy2_random": "S2 Random",
    "strategy3_line_proportional": "S3 Line-Proportional",
    "strategy4_regression_simple": "S4 Simple Regression",
    "strategy5_regression_multi": "S5 Multi Regression",
}

OUTPUT_FIELDS = [
    "Strategy",
    "Total Cost (Builds)",
    "Found vulnerabilities",
    "Effective Trigger Rate (%)",
    "Cost per vulnerability",
]


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

    rows = []
    with INPUT_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            strategy_key = row.get("strategy", "").strip()
            if strategy_key not in STRATEGY_LABELS:
                continue

            total_cost = float(row["additional_builds_total"])
            found_vulns = int(float(row["vulnerabilities_detected_additional"]))
            raw_rate = float(
                row.get("detection_vulnerability_rate", row.get("success_ratio", 0.0))
            )
            trigger_rate = raw_rate * 100.0 if raw_rate <= 1.0 else raw_rate
            cost_per_vuln = total_cost / found_vulns if found_vulns else 0.0

            rows.append(
                {
                    "Strategy": STRATEGY_LABELS[strategy_key],
                    "Total Cost (Builds)": total_cost,
                    "Found vulnerabilities": found_vulns,
                    "Effective Trigger Rate (%)": trigger_rate,
                    "Cost per vulnerability": cost_per_vuln,
                }
            )

    # Preserve the intended S1–S5 order.
    rows.sort(key=lambda record: list(STRATEGY_LABELS.values()).index(record["Strategy"]))

    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
