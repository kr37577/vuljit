from __future__ import annotations

from pathlib import Path

import pandas as pd

from RQ3.core.predictions import (
    collect_predictions,
    iter_prediction_files,
    load_project_predictions,
    PREDICTION_SUFFIX,
)


def _write_prediction_csv(path: Path, project: str) -> None:
    df = pd.DataFrame(
        {
            "merge_date": ["2024-01-01", "2024-01-02"],
            "is_vcc": [True, False],
            "predicted_risk_score": [0.8, 0.1],
        }
    )
    df.to_csv(path / f"{project}{PREDICTION_SUFFIX}", index=False)


def test_iter_prediction_files_finds_nested_csv(tmp_path) -> None:
    nested = tmp_path / "alpha"
    nested.mkdir()
    _write_prediction_csv(nested, "alpha")
    files = list(iter_prediction_files(tmp_path))
    assert len(files) == 1
    assert files[0].endswith(PREDICTION_SUFFIX)


def test_load_project_predictions_filters_columns(tmp_path) -> None:
    _write_prediction_csv(tmp_path, "project")
    df = load_project_predictions(
        tmp_path / f"project{PREDICTION_SUFFIX}",
        risk_column="predicted_risk_score",
        label_column=None,
    )
    assert df is not None
    assert set(df.columns) == {"merge_date", "is_vcc", "predicted_risk_score", "project"}
    assert df["project"].iloc[0] == "project"


def test_collect_predictions_concatenates_frames(tmp_path) -> None:
    _write_prediction_csv(tmp_path, "one")
    _write_prediction_csv(tmp_path, "two")
    collected = collect_predictions(tmp_path, risk_column="predicted_risk_score", label_column=None)
    assert collected["project"].nunique() == 2
