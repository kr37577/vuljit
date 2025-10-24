from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def phase5_dataset(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    predictions_root = root / "predictions"
    timeline_dir = root / "build_timelines"
    data_dir = root / "data"
    build_counts_path = root / "project_build_counts.csv"
    detection_table = root / "detection_time_results.csv"
    output_dir = root / "outputs"
    phase4_output = root / "phase4_outputs"

    predictions_root.mkdir(parents=True, exist_ok=True)
    timeline_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase4_output.mkdir(parents=True, exist_ok=True)
    (data_dir / "alpha").mkdir(parents=True, exist_ok=True)

    detection_df = pd.DataFrame(
        {
            "project": ["alpha"],
            "commit_date": ["2024-01-01"],
            "reported_date": ["2024-01-03"],
            "detection_time_days": [2.0],
        }
    )
    detection_df.to_csv(detection_table, index=False)

    build_counts_df = pd.DataFrame({"project": ["alpha"], "builds_per_day": [1.0]})
    build_counts_df.to_csv(build_counts_path, index=False)

    timeline_df = pd.DataFrame(
        {
            "project": ["alpha", "alpha", "alpha"],
            "merge_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "day_index": [1, 2, 3],
            "builds_per_day": [1, 1, 1],
            "build_index_start": [1, 2, 3],
            "build_index_end": [1, 2, 3],
            "cumulative_builds": [1, 2, 3],
            "daily_commit_count": [5, 4, 3],
        }
    )
    timeline_df.to_csv(timeline_dir / "alpha_build_timeline.csv", index=False)

    metrics_df = pd.DataFrame(
        {
            "merge_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "lines_added": [10, 20, 15],
            "lines_deleted": [5, 5, 3],
            "files_changed": [2, 3, 1],
            "daily_commit_count": [5, 4, 3],
        }
    )
    metrics_df.to_csv(data_dir / "alpha" / "alpha_daily_aggregated_metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "merge_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "is_vcc": [True, False, False],
            "predicted_risk_VCCFinder_Coverage": [0.9, 0.2, 0.1],
            "daily_commit_count": [5, 4, 3],
            "files_changed": [2, 3, 1],
            "lines_added": [10, 20, 15],
            "lines_deleted": [5, 5, 3],
            "builds_per_day": [1, 1, 1],
        }
    )
    predictions_df.to_csv(
        predictions_root / "alpha_daily_aggregated_metrics_with_predictions.csv", index=False
    )

    # Set environment overrides so defaults point to the test dataset.
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_DETECTION_TABLE", str(detection_table))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_BUILD_COUNTS", str(build_counts_path))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_PREDICTIONS_ROOT", str(predictions_root))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_MINIMAL_PREDICTIONS_ROOT", str(predictions_root))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE5_MINIMAL_OUTPUT", str(output_dir / "minimal.csv"))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE4_OUTPUT_DIR", str(phase4_output))
    monkeypatch.setenv("RQ3_DEFAULT_PHASE4_PREDICTIONS_ROOT", str(predictions_root))
    monkeypatch.setenv("RQ3_DEFAULT_TIMELINE_OUTPUT_DIR", str(timeline_dir))
    monkeypatch.setenv("RQ3_DEFAULT_TIMELINE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("RQ3_DEFAULT_TIMELINE_BUILD_COUNTS", str(build_counts_path))

    env = os.environ.copy()
    return {
        "detection_table": detection_table,
        "build_counts": build_counts_path,
        "predictions_root": predictions_root,
        "timeline_dir": timeline_dir,
        "data_dir": data_dir,
        "output_dir": output_dir,
        "phase4_output": phase4_output,
        "env": env,
    }
