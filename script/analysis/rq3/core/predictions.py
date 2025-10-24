"""Helpers for enumerating and loading prediction CSV files used in RQ3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

from .io import normalise_path


PREDICTION_SUFFIX = "_daily_aggregated_metrics_with_predictions.csv"


def iter_prediction_files(root: str | os.PathLike[str]) -> Iterator[str]:
    """Yield prediction file paths under ``root`` with the expected suffix."""

    root_path = Path(root)
    if not root_path.is_dir():
        return iter(())
    for dirpath, _, filenames in os.walk(root_path):
        for name in filenames:
            if name.endswith(PREDICTION_SUFFIX):
                yield normalise_path(Path(dirpath) / name)


def load_project_predictions(
    path: str | os.PathLike[str],
    risk_column: str,
    label_column: Optional[str],
) -> Optional[pd.DataFrame]:
    """Load a single project's prediction CSV, validating required columns."""

    df = pd.read_csv(path)
    required = {"merge_date", "is_vcc", risk_column}
    if not required.issubset(df.columns):
        return None
    df = df[list(required)].copy()
    df["project"] = (
        Path(path).name.split(PREDICTION_SUFFIX)[0]
    )
    df[risk_column] = pd.to_numeric(df[risk_column], errors="coerce")
    df = df.dropna(subset=[risk_column])
    if df.empty:
        return None
    if label_column and label_column in df.columns:
        df[label_column] = df[label_column].astype(bool)
    df["is_vcc"] = df["is_vcc"].astype(bool)
    return df


def collect_predictions(
    predictions_root: str | os.PathLike[str],
    risk_column: str,
    label_column: Optional[str],
) -> pd.DataFrame:
    """Aggregate all project-level prediction CSVs into a single DataFrame."""

    frames: List[pd.DataFrame] = []
    for path in iter_prediction_files(predictions_root):
        frame = load_project_predictions(path, risk_column, label_column)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"No prediction files found under {normalise_path(predictions_root)!r}."
        )
    return pd.concat(frames, ignore_index=True)

