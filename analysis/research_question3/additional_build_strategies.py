#!/usr/bin/env python3
"""Additional-build scheduling strategies for RQ3.

This module implements four strategies that consume the build timeline outputs
and the detection time table extracted from OSS-Fuzz metadata. Each strategy
emits a per-project schedule that can be consumed by the additional-build
simulation framework.

Inputs (default locations resolve via :mod:`RQ3.core.io.DEFAULTS`):
- ``datasets/derived_artifacts/rq3/timeline_outputs/build_timelines/*.csv`` for
  daily build timelines.
- ``datasets/raw/rq3_dataset/detection_time_results.csv`` for commit/detection
  intervals.
- ``datasets/raw/rq3_dataset/project_build_counts.csv`` for baseline build
  frequency.
- ``datasets/derived_artifacts/<project>/<project>_daily_aggregated_metrics.csv``
  for daily code churn metrics (used by strategies 3 and 4).
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .core.io import load_detection_table, load_build_counts, normalise_path, resolve_default
except ImportError:  # pragma: no cover
    from core.io import load_detection_table, load_build_counts, normalise_path, resolve_default


from scripts.modeling import settings as prediction_settings

RISK_COLUMN = "predicted_risk_VCCFinder_Coverage"

__all__ = [
    "strategy1_median_schedule",
    "strategy2_random_within_median_range",
    "strategy3_line_change_proportional",
    "strategy4_cross_project_regression",
]

LOGGER = logging.getLogger(__name__)

_STRATEGY4_BASE_FEATURES = ("builds_per_day", "label_flag")
_STRATEGY4_FALLBACK_FEATURES = (
    "daily_commit_count",
    "files_changed",
    "lines_added",
    "lines_deleted",
)
_STRATEGY4_MODEL_VERSION = "linear_regression_v1"
SIMPLE_REGRESSION_FEATURES = ("line_change_total",)
_VALID_STRATEGY4_MODES = {"multi", "simple"}
_PROJECT_STRATEGY_MODES = {"per_project", "cross_project"}


def _normalize_project_mode(value: Optional[str], default: str = "per_project") -> str:
    normalized = (value or default).strip().lower()
    if normalized not in _PROJECT_STRATEGY_MODES:
        raise ValueError(f"Unsupported strategy mode: {value!r} (expected one of {_PROJECT_STRATEGY_MODES})")
    return normalized


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    """Return items with duplicates removed while preserving order."""

    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        if not item or not isinstance(item, str):
            continue
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _default_strategy4_features() -> List[str]:
    """Resolve default feature columns for Strategy4 from prediction settings."""

    feature_candidates: List[str] = []
    if prediction_settings is None:
        LOGGER.debug("Strategy4 debug: prediction_settings is None")
    else:
        for attr in (
            # "KAMEI_FEATURES",
            "VCCFINDER_FEATURES",
            # "PROJECT_CALCULATION_FEATURES",
            "PROJECT_TOTAL_PERCENT_FEATURES",
        ):
            values = getattr(prediction_settings, attr, None)
            LOGGER.debug("Strategy4 debug: %s -> %r", attr, values)
            if isinstance(values, (list, tuple)):
                feature_candidates.extend(str(v) for v in values)
    # if not feature_candidates:
    #     feature_candidates.extend(_STRATEGY4_FALLBACK_FEATURES)
    # for base in _STRATEGY4_BASE_FEATURES:
    #     feature_candidates.append(base)
    return _dedupe_preserve_order(feature_candidates)


def _feature_signature(features: Sequence[str]) -> str:
    """Return a stable hash signature for the feature ordering."""

    joined = "\n".join(features)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _compute_error_statistics(errors: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, and MAPE for provided error and actual arrays."""

    if errors.size == 0:
        nan = float("nan")
        return {"mae": nan, "rmse": nan, "mape": nan}

    abs_errors = np.abs(errors)
    mae = float(abs_errors.mean()) if abs_errors.size else float("nan")
    rmse = float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan")
    non_zero_mask = np.abs(actual) > 1e-12
    if non_zero_mask.any():
        mape = float(np.mean(abs_errors[non_zero_mask] / np.abs(actual[non_zero_mask])))
    else:
        mape = float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape}


def _summarise_fold_performance(
    frame: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    fold_column: str = "walkforward_fold",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Aggregate per-fold regression performance and overall statistics."""

    if frame.empty or target_column not in frame.columns or prediction_column not in frame.columns:
        return {}, {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}

    relevant = frame[[fold_column, target_column, prediction_column]].copy()
    relevant = relevant.dropna(subset=[target_column, prediction_column])
    if relevant.empty:
        return {}, {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0}

    fold_metrics: Dict[str, Dict[str, float]] = {}
    for fold_id, fold_df in relevant.groupby(fold_column):
        actual = fold_df[target_column].to_numpy(dtype=float)
        predicted = fold_df[prediction_column].to_numpy(dtype=float)
        errors = predicted - actual
        stats = _compute_error_statistics(errors, actual)
        stats["count"] = float(len(fold_df))
        fold_metrics[str(fold_id)] = stats

    overall_actual = relevant[target_column].to_numpy(dtype=float)
    overall_predicted = relevant[prediction_column].to_numpy(dtype=float)
    overall_stats = _compute_error_statistics(overall_predicted - overall_actual, overall_actual)
    overall_stats["count"] = float(len(relevant))
    return fold_metrics, overall_stats


def _serialise_timestamp(value: Any) -> Optional[str]:
    """Convert value to ISO8601 UTC string if possible."""

    ts = _coerce_utc_timestamp(value)
    if ts is None:
        return None
    return ts.isoformat()


def _collect_fold_details(frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Collect train/validation window boundaries per fold."""

    if frame.empty or "walkforward_fold" not in frame.columns:
        return {}

    details: Dict[str, Dict[str, Any]] = {}
    for fold_id, subset in frame.groupby("walkforward_fold"):
        info: Dict[str, Any] = {}
        if "train_window_start" in subset.columns:
            info["train_window_start"] = _serialise_timestamp(subset["train_window_start"].dropna().min())
            info["train_window_end"] = _serialise_timestamp(subset["train_window_end"].dropna().max())
        if "validation_window_start" in subset.columns:
            info["validation_window_start"] = _serialise_timestamp(subset["validation_window_start"].dropna().min())
            info["validation_window_end"] = _serialise_timestamp(subset["validation_window_end"].dropna().max())
        info["rows"] = int(len(subset))
        details[str(fold_id)] = info
    return details


@dataclass
class Alignment:
    """Timeline alignment metadata for a scheduled additional build."""

    merge_date: Optional[pd.Timestamp]
    day_index: Optional[int]
    status: str  # "within_range", "before_start", "after_end", "missing"


@dataclass
class RegressionModel:
    """Simple linear regression model learned in Strategy 4."""

    intercept: float
    coefficients: Dict[str, float]
    feature_order: List[str]

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = features[self.feature_order].to_numpy(dtype=float)
        return self.intercept + values.dot(np.array([self.coefficients[f] for f in self.feature_order]))


def _ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _resolve_path(default_key: str, value: Optional[str]) -> str:
    return normalise_path(value or resolve_default(default_key))


def _load_detection_table(path: Optional[str] = None) -> pd.DataFrame:
    df = load_detection_table(_resolve_path("phase5.detection_table", path))

    def _coalesce_datetime(columns: Sequence[str]) -> pd.Series:
        series = pd.Series(pd.NaT, index=df.index)
        for column in columns:
            if column in df.columns:
                converted = _ensure_datetime(df[column])
                series = series.fillna(converted)
        return series

    reported_columns = (
        "reported_date_utc",
    )
    commit_columns = (
        "commit_date_utc",
    )
    df["reported_date"] = _coalesce_datetime(reported_columns)
    df["commit_date"] = _coalesce_datetime(commit_columns)
    return df


def _load_build_timelines(directory: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    timelines: Dict[str, pd.DataFrame] = {}
    root = Path(_resolve_path("timeline.output_dir", directory))
    if not root.is_dir():
        return timelines
    for csv_path in root.glob("*_build_timeline.csv"):
        project = csv_path.name.replace("_build_timeline.csv", "")
        df = pd.read_csv(csv_path)
        if "merge_date" not in df.columns:
            continue
        df["merge_date_ts"] = pd.to_datetime(df["merge_date"], utc=True, errors="coerce")
        df = df.sort_values("merge_date_ts").reset_index(drop=True)
        timelines[project] = df
    return timelines


def _align_to_timeline(timeline: pd.DataFrame, target_date: pd.Timestamp) -> Alignment:
    if timeline.empty or pd.isna(target_date):
        return Alignment(merge_date=None, day_index=None, status="missing")
    if "merge_date_ts" not in timeline.columns:
        raise ValueError("Timeline must include a 'merge_date_ts' column.")
    target_date = target_date.normalize()
    first_date = timeline["merge_date_ts"].iloc[0]
    last_date = timeline["merge_date_ts"].iloc[-1]
    if target_date < first_date:
        return Alignment(merge_date=first_date, day_index=int(timeline["day_index"].iloc[0]), status="before_start")
    if target_date > last_date:
        return Alignment(merge_date=last_date, day_index=int(timeline["day_index"].iloc[-1]), status="after_end")
    mask = timeline["merge_date_ts"] >= target_date
    subset = timeline.loc[mask]
    if subset.empty:
        return Alignment(merge_date=last_date, day_index=int(timeline["day_index"].iloc[-1]), status="after_end")
    row = subset.iloc[0]
    return Alignment(merge_date=row["merge_date_ts"], day_index=int(row["day_index"]), status="within_range")


def _load_project_metrics(project: str, data_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    csv_path = Path(_resolve_path("timeline.data_dir", data_dir)) / project / f"{project}_daily_aggregated_metrics.csv"
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    date_column = None
    for candidate in ("merge_date", "label_date"):
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column is None:
        return None
    if date_column != "merge_date":
        df = df.rename(columns={date_column: "merge_date"})
    df["merge_date_ts"] = pd.to_datetime(df["merge_date"], utc=True, errors="coerce")
    return df


def _prepare_line_change_metrics(project: str, data_dir: str) -> Optional[pd.DataFrame]:
    df = _load_project_metrics(project, data_dir)
    if df is None:
        return None
    for col in ("lines_added", "lines_deleted"):
        if col not in df.columns:
            df[col] = 0.0
    df["line_change_total"] = df["lines_added"].fillna(0) + df["lines_deleted"].fillna(0)
    df = df.sort_values("merge_date_ts")
    return df


def _load_build_counts(path: Optional[str] = None) -> pd.DataFrame:
    return load_build_counts(_resolve_path("phase5.build_counts", path))


def _prediction_csv_path(project: str, predictions_root: Optional[str]) -> Optional[Path]:
    root = Path(_resolve_path("phase5.predictions_root", predictions_root))
    if not root.exists():
        return None
    candidate = root / project / f"{project}_daily_aggregated_metrics_with_predictions.csv"
    if candidate.is_file():
        return candidate
    alternate = root / f"{project}_daily_aggregated_metrics_with_predictions.csv"
    if alternate.is_file():
        return alternate
    pattern = f"{project}_daily_aggregated_metrics_with_predictions"
    suffix = ".csv"
    search_roots = []
    project_dir = root / project
    if project_dir.is_dir():
        search_roots.append(project_dir)
    search_roots.append(root)
    matches: List[Path] = []
    for base in search_roots:
        if base.is_dir():
            matches.extend(base.rglob(f"{pattern}*{suffix}"))
    if matches:
        matches = sorted(matches, key=lambda p: (len(p.relative_to(root).parts), str(p)))
        return matches[0]
    return None


def _load_prediction_frame(
    project: str,
    predictions_root: str,
) -> Optional[pd.DataFrame]:
    path = _prediction_csv_path(project, predictions_root)
    if path is None:
        return None
    df = pd.read_csv(path)
    date_column = None
    for candidate in ("merge_date", "label_date", "date"):
        if candidate in df.columns:
            date_column = candidate
            break
    if date_column is None:
        return None
    if date_column != "merge_date":
        df = df.rename(columns={date_column: "merge_date"})
    df.columns = [str(col).strip() for col in df.columns]
    df = df.copy()
    df["merge_date_ts"] = pd.to_datetime(df["merge_date"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["merge_date_ts"])
    return df


_WALKFORWARD_CACHE: Dict[Tuple[str, str, int, bool], Dict[str, Any]] = {}
if prediction_settings is not None:
    _DEFAULT_WALKFORWARD_SPLITS = int(getattr(prediction_settings, "N_SPLITS_TIMESERIES", 10))
    _DEFAULT_USE_RECENT_FOR_TRAINING = bool(
        getattr(prediction_settings, "USE_ONLY_RECENT_FOR_TRAINING", False)
    )
else:
    _DEFAULT_WALKFORWARD_SPLITS = 10
    _DEFAULT_USE_RECENT_FOR_TRAINING = False


def _resolve_walkforward_config(
    walkforward_splits: Optional[int],
    use_recent_training: Optional[bool],
) -> Tuple[int, bool]:
    """Resolve walkforward configuration with environment fallbacks."""

    splits = walkforward_splits
    if splits is None:
        env_value = os.getenv("RQ3_WALKFORWARD_SPLITS")
        if env_value:
            try:
                splits = int(env_value)
            except ValueError:
                splits = None
    if splits is None or splits <= 1:
        splits = _DEFAULT_WALKFORWARD_SPLITS

    recent = use_recent_training
    if recent is None:
        env_recent = os.getenv("RQ3_WALKFORWARD_USE_RECENT")
        if env_recent:
            recent = env_recent.strip().lower() in {"1", "true", "yes", "on"}
    if recent is None:
        recent = _DEFAULT_USE_RECENT_FOR_TRAINING

    return splits, recent


def _compute_walkforward_chunks(n_rows: int, n_splits: int) -> List[List[int]]:
    """Split ``n_rows`` items into ``n_splits`` sequential chunks."""

    if n_rows <= 0 or n_splits <= 1:
        return []
    chunk_size = max(1, math.ceil(n_rows / n_splits))
    indices = list(range(n_rows))
    return [indices[i : i + chunk_size] for i in range(0, n_rows, chunk_size)]


def _build_walkforward_metadata(
    predictions: pd.DataFrame,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
) -> Dict[str, Any]:
    """Derive walkforward fold metadata from a prediction frame."""

    splits, recent = _resolve_walkforward_config(walkforward_splits, use_recent_training)
    if predictions.empty:
        return {
            "config": {"splits": splits, "use_recent_training": recent},
            "folds": {},
            "assignments": pd.DataFrame(columns=["merge_date_ts", "walkforward_fold", "train_window_start", "train_window_end"]),
        }

    ordered = predictions.sort_values("merge_date_ts").reset_index(drop=False)
    chunks = _compute_walkforward_chunks(len(ordered), splits)
    if not chunks or len(chunks) <= 1:
        return {
            "config": {"splits": splits, "use_recent_training": recent},
            "folds": {},
            "assignments": pd.DataFrame(columns=["merge_date_ts", "walkforward_fold", "train_window_start", "train_window_end"]),
        }

    assignments: List[Dict[str, Any]] = []
    folds: Dict[str, Dict[str, Any]] = {}
    for fold_number in range(1, len(chunks)):
        fold_id = f"fold-{fold_number}"
        train_indices = chunks[fold_number - 1] if recent else [idx for chunk in chunks[:fold_number] for idx in chunk]
        test_indices = chunks[fold_number]

        if not test_indices:
            continue

        train_dates = ordered.loc[train_indices, "merge_date_ts"] if train_indices else pd.Series(dtype="datetime64[ns, UTC]")
        test_dates = ordered.loc[test_indices, "merge_date_ts"]
        train_start = train_dates.min() if not train_dates.empty else None
        train_end = train_dates.max() if not train_dates.empty else None
        test_start = test_dates.min()
        test_end = test_dates.max()

        folds[fold_id] = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_indices": [int(ordered.loc[idx, "index"]) for idx in train_indices],
            "test_indices": [int(ordered.loc[idx, "index"]) for idx in test_indices],
        }

        for local_idx in test_indices:
            assignments.append(
                {
                    "merge_date_ts": ordered.loc[local_idx, "merge_date_ts"],
                    "walkforward_fold": fold_id,
                    "train_window_start": train_start,
                    "train_window_end": train_end,
                }
            )

    assignment_df = pd.DataFrame(assignments).sort_values("merge_date_ts").reset_index(drop=True)
    return {
        "config": {"splits": splits, "use_recent_training": recent},
        "folds": folds,
        "assignments": assignment_df,
    }


def _get_project_walkforward_metadata(
    project: str,
    predictions_root: str,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return cached walkforward metadata for ``project``."""

    splits, recent = _resolve_walkforward_config(walkforward_splits, use_recent_training)
    key = (project, predictions_root, splits, recent)
    if key in _WALKFORWARD_CACHE:
        return _WALKFORWARD_CACHE[key]

    frame = _load_prediction_frame(project, predictions_root)
    if frame is None:
        metadata = {
            "config": {"splits": splits, "use_recent_training": recent},
            "folds": {},
            "assignments": pd.DataFrame(columns=["merge_date_ts", "walkforward_fold", "train_window_start", "train_window_end"]),
        }
        _WALKFORWARD_CACHE[key] = metadata
        return metadata

    metadata = _build_walkforward_metadata(frame, splits, recent)
    _WALKFORWARD_CACHE[key] = metadata
    return metadata


def _coerce_utc_timestamp(value: Any) -> Optional[pd.Timestamp]:
    """Convert ``value`` to a UTC timestamp or ``None`` if invalid."""

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _summarise_duration_series(values: pd.Series) -> Dict[str, float]:
    """Return median and interquartile statistics for a numeric series."""

    cleaned = pd.to_numeric(values, errors="coerce").astype(float)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.empty:
        nan = float("nan")
        return {"median": nan, "q1": nan, "q3": nan}

    median = float(np.nanmedian(cleaned))
    q1 = float(cleaned.quantile(0.25))
    q3 = float(cleaned.quantile(0.75))
    if not math.isfinite(q1):
        q1 = float(np.nanmin(cleaned))
    if not math.isfinite(q3):
        q3 = float(np.nanmax(cleaned))
    if q3 < q1:
        q3 = q1
    return {"median": median, "q1": q1, "q3": q3}


def _resolve_median_with_fallback(
    project_stats: Dict[str, Dict[str, float]],
    global_stats: Dict[str, float],
    fold_identifier: Optional[str],
    *,
    allow_project_fallback: bool = False,
    allow_global_fallback: bool = False,
    cross_project_stats: Optional[Dict[str, Dict[str, float]]] = None,
    project_key: Optional[str] = None,
    prefer_lopo_only: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Resolve median detection builds with optional preference for LOPO statistics."""

    def _extract(stats_dict: Optional[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(stats_dict, dict):
            return None, None
        median_raw = stats_dict.get("median")
        if median_raw is None:
            return None, None
        median_val = float(median_raw)
        if not math.isfinite(median_val):
            return None, None
        median_val = max(median_val, 0.0)
        count_raw = stats_dict.get("count")
        count_val = float(count_raw) if count_raw is not None and math.isfinite(float(count_raw)) else float("nan")
        return median_val, count_val

    if fold_identifier and not prefer_lopo_only:
        fold_stats = project_stats.get(fold_identifier)
        median_val, count_val = _extract(fold_stats)
        if median_val is not None:
            return median_val, count_val, "fold"
    if allow_project_fallback and not prefer_lopo_only:
        project_stats_dict = project_stats.get("__overall__")
        project_median, project_count = _extract(project_stats_dict)
        if project_median is not None:
            return project_median, project_count, "project"

    if allow_global_fallback:
        if project_key and cross_project_stats:
            lopo_stats = cross_project_stats.get(project_key)
            lopo_median, lopo_count = _extract(lopo_stats)
            if lopo_median is not None:
                return lopo_median, lopo_count, "global_lopo"

        global_median, global_count = _extract(global_stats)
        if global_median is not None:
            return global_median, global_count, "global"

    return None, None, None


def _compute_lopo_project_budgets(project_totals: Dict[str, float]) -> Dict[str, float]:
    """Return per-project caps derived from the mean demand of all other projects."""

    budgets: Dict[str, float] = {}
    projects = list(project_totals.keys())
    if not projects:
        return budgets
    for project in projects:
        others = [
            float(project_totals.get(other, 0.0))
            for other in projects
            if other != project and math.isfinite(project_totals.get(other, float("nan")))
        ]
        others = [value for value in others if value > 0]
        if others:
            cap = float(sum(others) / len(others))
        else:
            cap = float(project_totals.get(project, 0.0) or 0.0)
        budgets[project] = max(cap, 0.0)
    return budgets


def _allocate_project_budget(contexts: List[Dict[str, Any]], project_budget: Optional[float]) -> None:
    """Assign integer budgets to each context subject to a project-level cap."""

    if not contexts:
        return

    if project_budget is None or not math.isfinite(project_budget):
        for ctx in contexts:
            ctx["allocated_budget"] = max(int(ctx.get("requested_budget", 0)), 0)
        return

    budget_int = int(max(round(project_budget), 0))
    if budget_int <= 0:
        for ctx in contexts:
            ctx["allocated_budget"] = 0
        return

    requested_sum = sum(max(int(ctx.get("requested_budget", 0)), 0) for ctx in contexts)
    if requested_sum <= 0:
        for ctx in contexts:
            ctx["allocated_budget"] = 0
        return
    if requested_sum <= budget_int:
        for ctx in contexts:
            ctx["allocated_budget"] = max(int(ctx.get("requested_budget", 0)), 0)
        return

    ratio = budget_int / requested_sum
    provisional: List[Tuple[Dict[str, Any], float]] = []
    allocated_total = 0
    for ctx in contexts:
        requested = max(int(ctx.get("requested_budget", 0)), 0)
        scaled = requested * ratio
        base_value = int(math.floor(scaled))
        base_value = min(base_value, requested)
        frac = float(scaled - base_value)
        ctx["allocated_budget"] = base_value
        provisional.append((ctx, frac))
        allocated_total += base_value

    remaining = budget_int - allocated_total
    if remaining <= 0:
        return
    adjustment_order = sorted(
        provisional,
        key=lambda item: (
            -item[1],
            -item[0].get("requested_budget", 0),
            item[0].get("order", 0),
        ),
    )
    for ctx, _ in adjustment_order:
        if remaining <= 0:
            break
        requested = max(int(ctx.get("requested_budget", 0)), 0)
        if ctx["allocated_budget"] >= requested:
            continue
        ctx["allocated_budget"] += 1
        remaining -= 1


def _compute_project_fold_statistics(
    detection_df: pd.DataFrame,
    fold_metadata: Optional[Dict[str, Any]] = None,
    *,
    predictions_root: Optional[str] = None,
    projects: Optional[Sequence[str]] = None,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    compute_lopo: bool = True,
    build_counts_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute per-project statistics for each walkforward fold."""

    if detection_df is None or detection_df.empty:
        return {"__global__": {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}}

    frame = detection_df.copy()
    if "project" not in frame.columns:
        if "package_name" in frame.columns:
            frame["project"] = frame["package_name"]
        else:
            frame["project"] = ""
    frame["project"] = frame["project"].astype(str).str.strip()
    frame = frame[frame["project"] != ""]
    frame["detection_time_days"] = pd.to_numeric(frame.get("detection_time_days"), errors="coerce")
    frame = frame.dropna(subset=["detection_time_days"])
    frame = frame[frame["detection_time_days"] >= 0]

    if build_counts_df is None:
        try:
            build_counts_df = _load_build_counts()
        except Exception:  # pragma: no cover - fallback for missing defaults
            build_counts_df = pd.DataFrame(columns=["project", "builds_per_day"])

    build_counts_map = {}
    if build_counts_df is not None and not build_counts_df.empty:
        build_counts_map = {
            str(row.get("project", "")).strip(): float(row.get("builds_per_day", float("nan")))
            for _, row in build_counts_df.iterrows()
            if str(row.get("project", "")).strip()
        }

    frame["builds_per_day"] = frame["project"].map(build_counts_map).astype(float)
    builds_per_day = pd.to_numeric(frame["builds_per_day"], errors="coerce")
    detection_days = frame["detection_time_days"].astype(float)
    scaled_builds = detection_days * builds_per_day
    frame["detection_time_builds"] = scaled_builds.where(builds_per_day > 0, detection_days)
    frame = frame.dropna(subset=["detection_time_builds"])
    frame = frame[frame["detection_time_builds"] >= 0]

    if frame.empty:
        return {"__global__": {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}}

    projects_in_data = set(frame["project"].unique())
    if projects is not None:
        selected_projects = set(projects)
        projects_in_data &= selected_projects
    else:
        selected_projects = projects_in_data.copy()

    if fold_metadata:
        metadata_projects = set(fold_metadata.keys())
        selected_projects |= metadata_projects

    selected_projects = sorted(selected_projects)

    global_stats = _summarise_duration_series(frame["detection_time_builds"])
    global_stats["count"] = float(len(frame))

    dated = frame.copy()
    dated["commit_ts"] = _ensure_datetime(dated.get("commit_date")).dt.normalize()
    dated = dated.dropna(subset=["commit_ts"])

    results: Dict[str, Dict[str, Dict[str, float]]] = {"__global__": global_stats}
    project_column = frame["project"].to_numpy()

    lopo_stats: Dict[str, Dict[str, float]] = {}
    if compute_lopo and len(project_column) > 0:
        unique_projects_for_lopo = selected_projects or sorted(projects_in_data)
        for project in unique_projects_for_lopo:
            if project in lopo_stats:
                continue
            mask = project_column != project
            others = frame.loc[mask, "detection_time_builds"]
            stats = _summarise_duration_series(others)
            stats["count"] = float(len(others))
            lopo_stats[project] = stats
        results["__global_exclusive__"] = lopo_stats
    resolved_root = _resolve_path("phase5.predictions_root", predictions_root) if predictions_root is not None else _resolve_path("phase5.predictions_root", None)

    metadata_cache: Dict[str, Dict[str, Any]] = {}
    if fold_metadata:
        metadata_cache.update(fold_metadata)

    for project in selected_projects:
        project_rows = frame.loc[frame["project"] == project, :]
        project_stats: Dict[str, Dict[str, float]] = {"__overall__": _summarise_duration_series(project_rows["detection_time_builds"])}
        project_stats["__overall__"]["count"] = float(len(project_rows))

        if project not in metadata_cache:
            metadata_cache[project] = _get_project_walkforward_metadata(
                project,
                resolved_root,
                walkforward_splits=walkforward_splits,
                use_recent_training=use_recent_training,
            )

        metadata_entry = metadata_cache.get(project) or {}
        fold_map = metadata_entry.get("folds") if isinstance(metadata_entry, dict) and "folds" in metadata_entry else metadata_entry
        if not isinstance(fold_map, dict):
            fold_map = {}

        project_dated = dated.loc[dated["project"] == project, :]
        for fold_id, fold_info in fold_map.items():
            train_start = _coerce_utc_timestamp(fold_info.get("train_start") if isinstance(fold_info, dict) else None)
            train_end = _coerce_utc_timestamp(fold_info.get("train_end") if isinstance(fold_info, dict) else None)
            if train_end is None:
                fold_series = pd.Series(dtype=float)
            else:
                if train_start is not None:
                    mask = (project_dated["commit_ts"] >= train_start) & (project_dated["commit_ts"] <= train_end)
                else:
                    mask = project_dated["commit_ts"] <= train_end
                fold_series = project_dated.loc[mask, "detection_time_builds"]
            project_stats[fold_id] = _summarise_duration_series(fold_series)
            project_stats[fold_id]["count"] = float(len(fold_series))

        results[project] = project_stats

    return results


def _stable_fold_rng(project: str, fold: Optional[str], base_seed: int) -> np.random.Generator:
    """Return a deterministic RNG keyed by project, fold, and the base seed."""

    fold_token = fold if fold else "__no_fold__"
    token = f"{project}::{fold_token}::{base_seed}"
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
    seed_int = int.from_bytes(digest, "little")
    return np.random.Generator(np.random.PCG64(seed_int))


def _resolve_label_series(
    predictions: pd.DataFrame,
    risk_column: Optional[str],
    label_column: Optional[str],
    threshold: float,
) -> Tuple[Optional[pd.Series], Optional[str], Optional[float], bool]:
    label_name: Optional[str] = None

    if label_column and label_column in predictions.columns:
        label_name = label_column

    if label_name is None and risk_column:
        derived = risk_column.replace("predicted_risk_", "predicted_label_", 1)
        if derived in predictions.columns:
            label_name = derived

    if label_name is None:
        label_candidates = [col for col in predictions.columns if col.startswith("predicted_label")]
        if label_candidates:
            if risk_column:
                suffix = risk_column.replace("predicted_risk_", "")
                for candidate in label_candidates:
                    if candidate.endswith(suffix):
                        label_name = candidate
                        break
            if label_name is None:
                label_name = label_candidates[0]

    if label_name is not None:
        raw = predictions[label_name]
        if raw.dtype == bool:
            series = raw.fillna(False)
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if numeric.notna().any():
                series = numeric.fillna(0) >= 0.5
            else:
                lowered = raw.astype(str).str.lower()
                series = lowered.isin({"true", "1", "t", "yes"})
        return series.astype(bool), label_name, None, False

    if risk_column and risk_column in predictions.columns:
        numeric = pd.to_numeric(predictions[risk_column], errors="coerce").fillna(0.0)
        series = numeric >= float(threshold)
        label_descriptor = f"{risk_column}>= {threshold:.3f}"
        return series.astype(bool), label_descriptor, float(threshold), True

    return None, None, None, False


def _prepare_labelled_timeline(
    project: str,
    timeline: pd.DataFrame,
    predictions_root: Optional[str],
    risk_column: str,
    label_column: Optional[str],
    threshold: float,
    extra_columns: Optional[Sequence[str]] = None,
    walkforward_assignments: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[float], bool]:
    """Join build timeline with prediction-derived JIT labels."""

    predictions = _load_prediction_frame(project, predictions_root)
    if predictions is None:
        return None, None, None, False

    label_series, label_name, threshold_used, derived_from_threshold = _resolve_label_series(
        predictions, risk_column, label_column, threshold
    )
    if label_series is None or label_series.empty:
        return None, None, None, False

    predictions = predictions.copy()
    predictions["_strategy_label"] = label_series
    if "line_change_total" not in predictions.columns:
        if {"lines_added", "lines_deleted"}.issubset(predictions.columns):
            predictions["line_change_total"] = (
                pd.to_numeric(predictions["lines_added"], errors="coerce").fillna(0)
                + pd.to_numeric(predictions["lines_deleted"], errors="coerce").fillna(0)
            )
        else:
            predictions["line_change_total"] = np.nan
    merge_cols = ["merge_date_ts", "_strategy_label"]
    risk_present = risk_column in predictions.columns
    if risk_present:
        merge_cols.append(risk_column)
    if extra_columns:
        for col in extra_columns:
            if col in predictions.columns and col not in merge_cols:
                merge_cols.append(col)
    merged = pd.merge(timeline, predictions[merge_cols], on="merge_date_ts", how="left")
    label_series = merged["_strategy_label"]
    if label_series.dtype == object:
        label_series = label_series.infer_objects(copy=False)
    merged["_strategy_label"] = label_series.fillna(False).astype(bool)
    if walkforward_assignments is not None and not walkforward_assignments.empty:
        assignment_cols = [col for col in ("merge_date_ts", "walkforward_fold", "train_window_start", "train_window_end") if col in walkforward_assignments.columns]
        if assignment_cols and "merge_date_ts" in assignment_cols:
            assignment_frame = (
                walkforward_assignments[assignment_cols]
                .drop_duplicates(subset=["merge_date_ts"])
            )
            merged = pd.merge(merged, assignment_frame, on="merge_date_ts", how="left")
    if risk_present:
        merged[risk_column] = pd.to_numeric(merged[risk_column], errors="coerce")
    else:
        merged[risk_column] = np.nan
    return merged, label_name, threshold_used, derived_from_threshold

def strategy1_median_schedule(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    predictions_root: Optional[str] = None,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    *,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    mode: str = "per_project",
) -> pd.DataFrame:
    """Trigger median-sized additional builds using walkforward-aware medians.

    When the JIT model marks a day as vulnerable (`predicted_label_*` or risk
    >= ``threshold``), we allocate `ceil(median_detection_builds)` extra builds.
    The detection-build median is resolved with the following precedence to
    remain aligned with walkforward training windows:

    1. Median for the walkforward fold that contains the labelled day.
    2. Project-wide median across all detection samples.
    3. Global median across every project (fallback).
    """

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()
    predictions_root = _resolve_path("phase5.predictions_root", predictions_root)
    mode_normalized = _normalize_project_mode(mode)
    use_fold = mode_normalized == "per_project"

    project_timelines = {project: timeline for project, timeline in (timelines or {}).items() if not timeline.empty}
    if not project_timelines:
        return pd.DataFrame()

    metadata_map: Dict[str, Dict[str, Any]] = {}
    if use_fold:
        for project in project_timelines:
            metadata_map[project] = _get_project_walkforward_metadata(
                project,
                predictions_root,
                walkforward_splits=walkforward_splits,
                use_recent_training=use_recent_training,
            )

    stats = _compute_project_fold_statistics(
        detection_df,
        fold_metadata=metadata_map if use_fold else None,
        predictions_root=predictions_root,
        projects=list(project_timelines.keys()),
        walkforward_splits=walkforward_splits,
        use_recent_training=use_recent_training,
        compute_lopo=mode_normalized == "cross_project",
    )
    global_stats = stats.get("__global__", {}) if isinstance(stats.get("__global__"), dict) else {}
    lopo_stats = stats.get("__global_exclusive__", {}) if isinstance(stats.get("__global_exclusive__"), dict) else {}
    prefer_lopo_only = mode_normalized == "cross_project"

    def _resolve_median(
        project_stats: Dict[str, Dict[str, float]],
        fold: Optional[str],
        project_key: Optional[str],
        *,
        allow_project_fallback: bool,
        allow_global_fallback: bool,
    ) -> Tuple[Optional[float], Optional[str]]:
        # Prefer fold medians and fall back only when the training window lacks samples;
        # NaNs propagate until a stable (>=0) fallback is available to avoid over-scheduling.
        def _coerce_median(stats_dict: Optional[Dict[str, float]]) -> Optional[float]:
            if not isinstance(stats_dict, dict):
                return None
            candidate = stats_dict.get("median")
            if candidate is None:
                return None
            value = float(candidate)
            if not math.isfinite(value):
                return None
            return max(value, 0.0)

        if not prefer_lopo_only:
            fold_stats = project_stats.get(fold) if fold else None
            fold_median_days = _coerce_median(fold_stats)
            if fold_median_days is not None:
                return fold_median_days, "fold"

        if allow_project_fallback and not prefer_lopo_only:
            project_stats_dict = project_stats.get("__overall__")
            project_median_days = _coerce_median(project_stats_dict)
            if project_median_days is not None:
                return project_median_days, "project"

        if allow_global_fallback:
            if project_key and project_key in lopo_stats:
                lopo_median_days = _coerce_median(lopo_stats.get(project_key))
                if lopo_median_days is not None:
                    return lopo_median_days, "global_lopo"

        if allow_global_fallback:
            if project_key and project_key in lopo_stats:
                lopo_median_days = _coerce_median(lopo_stats.get(project_key))
                if lopo_median_days is not None:
                    return lopo_median_days, "global_lopo"

            global_median_days = _coerce_median(global_stats)
            if global_median_days is not None:
                return global_median_days, "global"

        return None, None

    records: List[Dict[str, object]] = []
    for project, timeline in project_timelines.items():
        metadata = metadata_map.get(project, {}) if use_fold else {}
        assignments = metadata.get("assignments") if isinstance(metadata, dict) else None

        merged, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
            walkforward_assignments=assignments if isinstance(assignments, pd.DataFrame) else None if use_fold else None,
        )
        if merged is None:
            continue
        merged = merged.copy()
        if use_fold:
            if "walkforward_fold" not in merged.columns:
                continue
            merged = merged[merged["walkforward_fold"].notna()].copy()
            if merged.empty:
                continue
        positive_mask = merged["_strategy_label"].fillna(False)
        positive = merged.loc[positive_mask].copy()
        if positive.empty:
            continue

        project_stats = stats.get(project, {}) if isinstance(stats.get(project), dict) else {}
        allow_project_fallback = True
        allow_global_fallback = True

        for _, row in positive.iterrows():
            fold_id = row.get("walkforward_fold") if use_fold else None
            if pd.isna(fold_id):
                fold_id = None
            fold_identifier = None if fold_id is None else str(fold_id)

            resolved_median_builds, median_source = _resolve_median(
                project_stats,
                fold_identifier,
                project,
                allow_project_fallback=allow_project_fallback,
                allow_global_fallback=allow_global_fallback,
            )
            if resolved_median_builds is None:
                continue

            builds_per_day = float(row.get("builds_per_day", 0.0))
            if not math.isfinite(builds_per_day) or builds_per_day <= 0:
                continue
            scheduled = int(math.ceil(resolved_median_builds))
            if scheduled <= 0:
                continue

            train_start = row.get("train_window_start")
            if pd.isna(train_start):
                train_start = None
            train_end = row.get("train_window_end")
            if pd.isna(train_end):
                train_end = None

            record: Dict[str, object] = {
                "project": project,
                "strategy": "median_label_trigger",
                "merge_date": row.get("merge_date_ts"),
                "day_index": int(row.get("day_index", 0)) if not pd.isna(row.get("day_index")) else None,
                "builds_per_day": builds_per_day,
                "median_detection_builds": resolved_median_builds,
                "scheduled_additional_builds": scheduled,
                "label_source": label_name,
                "walkforward_fold": fold_identifier if use_fold else None,
                "train_window_start": train_start if use_fold else None,
                "train_window_end": train_end if use_fold else None,
                "median_source": median_source,
                "strategy_mode": mode_normalized,
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            records.append(record)

    return pd.DataFrame(records)

def strategy2_random_within_median_range(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    predictions_root: Optional[str] = None,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    random_seed: int = 42,
    *,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    mode: str = "per_project",
) -> pd.DataFrame:
    """Schedule additional builds using fold-aware interquartile sampling of build counts."""

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()
    predictions_root = _resolve_path("phase5.predictions_root", predictions_root)
    mode_normalized = _normalize_project_mode(mode)
    use_fold = mode_normalized == "per_project"

    project_timelines = {project: timeline for project, timeline in (timelines or {}).items() if not timeline.empty}
    if not project_timelines:
        return pd.DataFrame()

    metadata_map: Dict[str, Dict[str, Any]] = {}
    if use_fold:
        for project in project_timelines:
            metadata_map[project] = _get_project_walkforward_metadata(
                project,
                predictions_root,
                walkforward_splits=walkforward_splits,
                use_recent_training=use_recent_training,
            )

    stats = _compute_project_fold_statistics(
        detection_df,
        fold_metadata=metadata_map if use_fold else None,
        predictions_root=predictions_root,
        projects=list(project_timelines.keys()),
        walkforward_splits=walkforward_splits,
        use_recent_training=use_recent_training,
        compute_lopo=mode_normalized == "cross_project",
    )
    global_stats = stats.get("__global__", {}) if isinstance(stats.get("__global__"), dict) else {}
    lopo_stats = stats.get("__global_exclusive__", {}) if isinstance(stats.get("__global_exclusive__"), dict) else {}
    prefer_lopo_only = mode_normalized == "cross_project"

    def _resolve_quartiles(
        project_stats: Dict[str, Dict[str, float]],
        fold: Optional[str],
        project_key: Optional[str],
        *,
        allow_project_fallback: bool,
        allow_global_fallback: bool,
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        # Resolve interquartile range in fold -> project -> global order, skipping folds with
        # sparse training data rather than synthesising ranges from incomplete statistics.
        def _coerce_quartiles(stats_dict: Optional[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
            if not isinstance(stats_dict, dict):
                return None, None
            q1_candidate = stats_dict.get("q1")
            q3_candidate = stats_dict.get("q3")
            if q1_candidate is None or q3_candidate is None:
                return None, None
            q1_value = float(q1_candidate)
            q3_value = float(q3_candidate)
            if not (math.isfinite(q1_value) and math.isfinite(q3_value)):
                return None, None
            q1_value = max(q1_value, 0.0)
            q3_value = max(q3_value, q1_value)
            return q1_value, q3_value

        if not prefer_lopo_only:
            fold_stats = project_stats.get(fold) if fold else None
            fold_q1_days, fold_q3_days = _coerce_quartiles(fold_stats)
            if fold_q1_days is not None and fold_q3_days is not None:
                return fold_q1_days, fold_q3_days, "fold"

        if allow_project_fallback and not prefer_lopo_only:
            project_stats_dict = project_stats.get("__overall__")
            project_q1_days, project_q3_days = _coerce_quartiles(project_stats_dict)
            if project_q1_days is not None and project_q3_days is not None:
                return project_q1_days, project_q3_days, "project"

        if allow_global_fallback:
            if project_key and project_key in lopo_stats:
                lopo_q1_days, lopo_q3_days = _coerce_quartiles(lopo_stats.get(project_key))
                if lopo_q1_days is not None and lopo_q3_days is not None:
                    return lopo_q1_days, lopo_q3_days, "global_lopo"

            global_q1_days, global_q3_days = _coerce_quartiles(global_stats)
            if global_q1_days is not None and global_q3_days is not None:
                return global_q1_days, global_q3_days, "global"

        return None, None, None

    rng_cache: Dict[Tuple[str, Optional[str]], np.random.Generator] = {}

    def _next_rng(project: str, fold: Optional[str]) -> np.random.Generator:
        key = (project, fold)
        if key not in rng_cache:
            rng_cache[key] = _stable_fold_rng(project, fold, random_seed)
        return rng_cache[key]

    records: List[Dict[str, object]] = []
    for project, timeline in project_timelines.items():
        metadata = metadata_map.get(project, {}) if use_fold else {}
        assignments = metadata.get("assignments") if isinstance(metadata, dict) else None

        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
            walkforward_assignments=assignments if isinstance(assignments, pd.DataFrame) else None if use_fold else None,
        )
        if labelled is None:
            continue

        labelled = labelled.copy()
        if use_fold:
            if "walkforward_fold" not in labelled.columns:
                continue
            labelled = labelled[labelled["walkforward_fold"].notna()].copy()
            if labelled.empty:
                continue
        positive = labelled["_strategy_label"].fillna(False)
        if not positive.any():
            continue

        positive_rows = labelled.loc[positive].copy()
        if positive_rows.empty:
            continue

        project_stats = stats.get(project, {}) if isinstance(stats.get(project), dict) else {}
        allow_project_fallback = True
        allow_global_fallback = True

        for _, row in positive_rows.iterrows():
            fold_id = row.get("walkforward_fold") if use_fold else None
            if pd.isna(fold_id):
                fold_id = None
            fold_identifier: Optional[str] = None if fold_id is None else str(fold_id)

            q1_builds, q3_builds, quartile_source = _resolve_quartiles(
                project_stats,
                fold_identifier,
                project,
                allow_project_fallback=allow_project_fallback,
                allow_global_fallback=allow_global_fallback,
            )
            if q1_builds is None or q3_builds is None:
                continue

            rng = _next_rng(project, fold_identifier)
            sampled_offset = float(q1_builds) if q3_builds == q1_builds else float(rng.uniform(q1_builds, q3_builds))

            builds_per_day = float(row.get("builds_per_day", 0.0))
            scheduled_builds = int(math.ceil(sampled_offset))
            if scheduled_builds <= 0:
                continue

            train_start = row.get("train_window_start")
            if pd.isna(train_start):
                train_start = None
            train_end = row.get("train_window_end")
            if pd.isna(train_end):
                train_end = None

            record: Dict[str, object] = {
                "project": project,
                "strategy": "median_iqr_random",
                "merge_date": row.get("merge_date_ts"),
                "day_index": int(row.get("day_index", 0)) if not pd.isna(row.get("day_index")) else None,
                "builds_per_day": builds_per_day,
                "offset_builds_q1": q1_builds,
                "offset_builds_q3": q3_builds,
                "sampled_offset_builds": sampled_offset,
                "scheduled_additional_builds": scheduled_builds,
                "label_source": label_name,
                "walkforward_fold": fold_identifier if use_fold else None,
                "train_window_start": train_start if use_fold else None,
                "train_window_end": train_end if use_fold else None,
                "quartile_source": quartile_source,
                "strategy_mode": mode_normalized,
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            records.append(record)

    return pd.DataFrame(records)


def strategy3_line_change_proportional(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    data_dir: Optional[str] = None,
    predictions_root: Optional[str] = None,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    scaling_factor: float = 1.0,
    clip_max: float = None,
    rounding_mode: str = "ceil",
    *,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    mode: str = "per_project",
    global_budget: Optional[float] = None,
) -> pd.DataFrame:
    """Adjust additional build frequency proportional to daily line churn, gated by JIT labels."""

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()
    data_dir = _resolve_path("timeline.data_dir", data_dir)
    predictions_root = _resolve_path("phase5.predictions_root", predictions_root)
    mode_normalized = _normalize_project_mode(mode)
    use_fold = mode_normalized == "per_project"

    rounding_mode_normalized = (rounding_mode or "ceil").strip().lower()
    if rounding_mode_normalized not in {"ceil", "floor", "round"}:
        raise ValueError(f"Unsupported rounding_mode: {rounding_mode!r}")

    project_payloads: Dict[str, Dict[str, Any]] = {}
    project_timelines = {project: timeline for project, timeline in (timelines or {}).items() if not timeline.empty}
    if not project_timelines:
        return pd.DataFrame()

    metadata_map: Dict[str, Dict[str, Any]] = {}
    if use_fold:
        for project in project_timelines:
            metadata_map[project] = _get_project_walkforward_metadata(
                project,
                predictions_root,
                walkforward_splits=walkforward_splits,
                use_recent_training=use_recent_training,
            )

    stats = _compute_project_fold_statistics(
        detection_df,
        fold_metadata=metadata_map if use_fold else None,
        predictions_root=predictions_root,
        projects=list(project_timelines.keys()),
        walkforward_splits=walkforward_splits,
        use_recent_training=use_recent_training,
        compute_lopo=mode_normalized == "cross_project",
    )
    global_stats = stats.get("__global__", {}) if isinstance(stats.get("__global__"), dict) else {}
    lopo_stats = stats.get("__global_exclusive__", {}) if isinstance(stats.get("__global_exclusive__"), dict) else {}
    prefer_lopo_only = mode_normalized == "cross_project"

    for project, timeline in project_timelines.items():
        if timeline.empty:
            continue
        metrics_df = _prepare_line_change_metrics(project, data_dir)
        if metrics_df is None:
            continue
        metadata = metadata_map.get(project, {}) if use_fold else {}
        assignments = metadata.get("assignments") if isinstance(metadata, dict) else None
        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
            walkforward_assignments=assignments if isinstance(assignments, pd.DataFrame) else None if use_fold else None,
        )
        if labelled is None:
            continue
        merged = pd.merge(
            labelled,
            metrics_df[["merge_date_ts", "line_change_total", "daily_commit_count"]],
            on="merge_date_ts",
            how="left",
        ).reset_index(drop=True)
        if not use_fold and "walkforward_fold" not in merged.columns:
            merged["walkforward_fold"] = None
        sort_cols = ["merge_date_ts"]
        if "day_index" in merged.columns:
            sort_cols.append("day_index")
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
        positive_mask = merged["_strategy_label"].fillna(False)
        if not positive_mask.any():
            continue
        positive = merged.loc[positive_mask].copy()
        project_stats = stats.get(project, {}) if isinstance(stats.get(project), dict) else {}

        line_change_series = merged["line_change_total"].fillna(0).astype(float)
        positive_line_series = line_change_series.where(positive_mask, np.nan)
        d_median_past_series = positive_line_series.expanding(min_periods=1).median().shift(1)

        commit_count_columns = [col for col in merged.columns if col.startswith("daily_commit_count")]
        if commit_count_columns:
            commit_count_series = merged[commit_count_columns[0]].astype(float)
            for col in commit_count_columns[1:]:
                commit_count_series = commit_count_series.combine_first(merged[col].astype(float))
        else:
            commit_count_series = pd.Series(np.nan, index=merged.index, dtype=float)

        allocation_map: Dict[int, Dict[str, object]] = {}
        if not positive.empty:
            if use_fold:
                fold_grouped = positive.groupby("walkforward_fold", dropna=False)
            else:
                fold_grouped = [(None, positive)]
            allow_project_fallback = True
            allow_global_fallback = True
            for fold_value, fold_rows in fold_grouped:
                fold_identifier = None if (pd.isna(fold_value)) else str(fold_value)
                median_builds, sample_count, source = _resolve_median_with_fallback(
                    project_stats,
                    global_stats,
                    fold_identifier,
                    allow_project_fallback=allow_project_fallback,
                    allow_global_fallback=allow_global_fallback,
                    cross_project_stats=lopo_stats,
                    project_key=project,
                    prefer_lopo_only=prefer_lopo_only,
                )
                if median_builds is None:
                    continue
                scaled_median_builds = max(scaling_factor, 0.0) * float(median_builds)
                if not math.isfinite(scaled_median_builds) or scaled_median_builds <= 0:
                    continue
                positive_count = int(len(fold_rows))
                sample_count_val = float(sample_count) if sample_count is not None and math.isfinite(sample_count) else float("nan")
                fold_budget_continuous = float(scaled_median_builds)
                fold_budget_target = max(int(round(scaled_median_builds)), 0)

                indices = [int(idx) for idx in fold_rows.index]
                raw_weights: Dict[int, float] = {}
                for idx in indices:
                    line_value = max(float(line_change_series.loc[idx]), 0.0)
                    baseline_value = float(d_median_past_series.loc[idx]) if idx in d_median_past_series.index else float("nan")
                    baseline_valid = math.isfinite(baseline_value) and baseline_value > 0
                    if not baseline_valid:
                        weight = 1.0
                        baseline_zero_fallback = True
                    else:
                        weight = math.log1p(line_value) / math.log1p(baseline_value)
                        if clip_max is not None and clip_max > 0:
                            weight = min(weight, clip_max)
                        baseline_zero_fallback = False
                    raw_weights[idx] = max(weight, 0.0)

                rounded_map: Dict[int, int] = {}
                expected_map: Dict[int, float] = {}
                for idx in indices:
                    expected_value = float(scaled_median_builds * raw_weights.get(idx, 0.0))
                    expected_map[idx] = expected_value
                    if rounding_mode_normalized == "ceil":
                        rounded_val = int(math.ceil(expected_value - 1e-9))
                    elif rounding_mode_normalized == "floor":
                        rounded_val = int(math.floor(expected_value + 1e-9))
                    else:  # "round"
                        rounded_val = int(round(expected_value))
                    rounded_map[idx] = max(rounded_val, 0)

                for idx in indices:
                    baseline_value = float(d_median_past_series.loc[idx]) if idx in d_median_past_series.index else float("nan")
                    baseline_valid = math.isfinite(baseline_value) and baseline_value > 0
                    allocation_map[idx] = {
                        "fold_budget": fold_budget_target,
                        "fold_budget_continuous": fold_budget_continuous,
                        "fold_budget_source": source,
                        "fold_positive_days": positive_count,
                        "fold_sample_count": sample_count_val,
                        "fold_identifier": fold_identifier,
                        "fold_median_detection_builds": float(median_builds),
                        "line_churn_baseline": baseline_value if baseline_valid else np.nan,
                        "baseline_zero_fallback": not baseline_valid,
                        "line_weight_raw": raw_weights.get(idx, 0.0),
                        "line_weight_share": raw_weights.get(idx, 0.0),
                        "expected_raw": expected_map.get(idx, 0.0),
                        "rounded_value": rounded_map.get(idx, 0),
                        "final_scheduled": rounded_map.get(idx, 0),
                        "rounding_mode_used": rounding_mode_normalized,
                        "fold_overflow_used": False,
                        "strategy_mode": mode_normalized,
                        "project_budget_requested": float("nan"),
                        "project_budget_cap": float("nan"),
                        "project_budget_source": "median_builds",
                    }

        if not allocation_map:
            continue

        project_payloads[project] = {
            "positive_rows": positive,
            "line_change_series": line_change_series,
            "commit_count_series": commit_count_series,
            "allocation_map": allocation_map,
            "label_name": label_name,
            "threshold_used": threshold_used,
            "derived_from_threshold": derived_from_threshold,
        }

    if not project_payloads:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for project, payload in project_payloads.items():
        positive_rows = payload["positive_rows"]
        line_change_series = payload["line_change_series"]
        commit_count_series = payload["commit_count_series"]
        label_name = payload["label_name"]
        threshold_used = payload["threshold_used"]
        derived_from_threshold = payload["derived_from_threshold"]
        allocation_map = payload["allocation_map"]

        for idx, row in positive_rows.iterrows():
            idx_int = int(idx)
            allocation = allocation_map.get(idx_int)
            if allocation is None:
                continue
            scheduled = int(max(allocation.get("final_scheduled", 0), 0))
            if scheduled <= 0:
                continue
            line_change_value = float(line_change_series.loc[idx]) if idx in line_change_series.index else float("nan")
            commit_value = float(commit_count_series.loc[idx]) if idx in commit_count_series.index else float("nan")
            record: Dict[str, object] = {
                "project": project,
                "strategy": "line_change_proportional",
                "merge_date": row.merge_date_ts,
                "day_index": int(row.day_index) if not pd.isna(row.day_index) else None,
                "builds_per_day": float(row.builds_per_day) if not pd.isna(row.builds_per_day) else np.nan,
                "line_change_total": line_change_value,
                "normalized_line_change": float(allocation.get("line_weight_raw", float("nan"))),
                "line_churn_baseline": float(allocation.get("line_churn_baseline", float("nan"))),
                "line_weight_share": float(allocation.get("line_weight_share", float("nan"))),
                "line_weight_raw": float(allocation.get("line_weight_raw", float("nan"))),
                "baseline_zero_fallback": bool(allocation.get("baseline_zero_fallback", False)),
                "expected_additional_builds_raw": float(allocation.get("expected_raw", 0.0)),
                "expected_additional_builds": float(allocation.get("expected_raw", 0.0)),
                "rounded_additional_builds": int(allocation.get("rounded_value", 0)),
                "rounding_mode_used": allocation.get("rounding_mode_used", rounding_mode_normalized),
                "fold_overflow_used": bool(allocation.get("fold_overflow_used", False)),
                "fold_budget": int(allocation.get("fold_budget", 0)),
                "fold_budget_continuous": float(allocation.get("fold_budget_continuous", float("nan"))),
                "fold_budget_source": allocation.get("fold_budget_source"),
                "fold_positive_days": int(allocation.get("fold_positive_days", 0)),
                "fold_sample_count": float(allocation.get("fold_sample_count", float("nan"))),
                "fold_median_detection_builds": float(allocation.get("fold_median_detection_builds", float("nan"))),
                "scheduled_additional_builds": scheduled,
                "daily_commit_count": commit_value if not pd.isna(commit_value) else np.nan,
                "strategy_mode": mode_normalized,
                "label_source": label_name,
                "walkforward_fold": row.get("walkforward_fold") if use_fold else None,
                "train_window_start": row.get("train_window_start") if use_fold else None,
                "train_window_end": row.get("train_window_end") if use_fold else None,
                "project_budget_requested": allocation.get("project_budget_requested"),
                "project_budget_cap": allocation.get("project_budget_cap"),
                "project_budget_source": allocation.get("project_budget_source"),
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            rows.append(record)

    return pd.DataFrame(rows)


def _build_regression_dataset(
    detection_df: pd.DataFrame,
    timelines: Dict[str, pd.DataFrame],
    build_counts_df: pd.DataFrame,
    predictions_root: Optional[str],
    feature_cols: Sequence[str],
    risk_column: str,
    label_column: Optional[str],
    threshold: float,
    *,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    required_feature_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    feature_cols = tuple(feature_cols)
    required_feature_columns = tuple(required_feature_columns or ())
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Strategy4 dataset build: features=%s (count=%d)",
            feature_cols,
            len(feature_cols),
        )
    predictions_root = _resolve_path("phase5.predictions_root", predictions_root)
    build_counts_map = dict(zip(build_counts_df["project"], build_counts_df["builds_per_day"]))
    detection_lookup: Dict[str, Dict[pd.Timestamp, List[float]]] = {}
    for _, record in detection_df.iterrows():
        project = (record.get("project") or "").strip()
        commit_ts = record.get("commit_date_utc")
        detection_time = record.get("detection_time_days")
        if not project or pd.isna(commit_ts) or pd.isna(detection_time):
            continue
        commit_norm = pd.to_datetime(commit_ts).normalize()
        detection_lookup.setdefault(project, {}).setdefault(commit_norm, []).append(float(detection_time))

    rows: List[Dict[str, object]] = []
    for project, timeline in timelines.items():
        if timeline.empty:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Strategy4 dataset: skip %s (empty timeline)", project)
            continue
        extra_columns = [col for col in feature_cols if col not in _STRATEGY4_BASE_FEATURES]
        metadata = _get_project_walkforward_metadata(
            project,
            predictions_root,
            walkforward_splits=walkforward_splits,
            use_recent_training=use_recent_training,
        )
        assignments = metadata.get("assignments") if isinstance(metadata, dict) else None
        folds_meta: Dict[str, Dict[str, Any]] = metadata.get("folds", {}) if isinstance(metadata, dict) else {}
        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
            extra_columns=extra_columns,
            walkforward_assignments=assignments,
        )
        if labelled is None:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Strategy4 dataset: skip %s (no labelled timeline)", project)
            continue
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Strategy4 dataset: %s labelled rows=%s", project, labelled.shape)
        labelled = labelled.copy()
        labelled["label_flag"] = labelled["_strategy_label"].astype(bool)
        labelled = labelled[labelled["label_flag"]]
        if labelled.empty:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Strategy4 dataset: skip %s (no positive labels)", project)
            continue
        if "walkforward_fold" not in labelled.columns:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("Strategy4 dataset: skip %s (missing walkforward_fold column)", project)
            continue
        before_fold_filter = len(labelled)
        labelled = labelled[labelled["walkforward_fold"].notna()]
        if labelled.empty:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Strategy4 dataset: skip %s (fold assignments missing; before=%d)",
                    project,
                    before_fold_filter,
                )
            continue
        labelled["walkforward_fold"] = labelled["walkforward_fold"].astype(str)
        train_start_map: Dict[str, Optional[pd.Timestamp]] = {}
        train_end_map: Dict[str, Optional[pd.Timestamp]] = {}
        val_start_map: Dict[str, Optional[pd.Timestamp]] = {}
        val_end_map: Dict[str, Optional[pd.Timestamp]] = {}
        for fold_id, fold_info in folds_meta.items():
            train_start_map[fold_id] = _coerce_utc_timestamp(fold_info.get("train_start"))
            train_end_map[fold_id] = _coerce_utc_timestamp(fold_info.get("train_end"))
            val_start_map[fold_id] = _coerce_utc_timestamp(fold_info.get("test_start"))
            val_end_map[fold_id] = _coerce_utc_timestamp(fold_info.get("test_end"))
        labelled["train_window_start"] = labelled["walkforward_fold"].map(train_start_map)
        labelled["train_window_end"] = labelled["walkforward_fold"].map(train_end_map)
        labelled["validation_window_start"] = labelled["walkforward_fold"].map(val_start_map)
        labelled["validation_window_end"] = labelled["walkforward_fold"].map(val_end_map)
        labelled = labelled[labelled["validation_window_start"].notna()]
        if labelled.empty:
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug(
                    "Strategy4 dataset: skip %s (validation window missing after mapping)",
                    project,
                )
            continue

        if required_feature_columns:
            missing_required = [col for col in required_feature_columns if col not in labelled.columns]
            if missing_required:
                missing_joined = ", ".join(sorted(set(missing_required)))
                raise ValueError(
                    f"Required Strategy4 features ({missing_joined}) are missing for project {project}. "
                    "Ensure the aggregated metrics/prediction CSVs include these columns before running simple mode."
                )

        detection_map = detection_lookup.get(project, {})
        for _, row in labelled.iterrows():
            merge_date = row.get("merge_date_ts")
            if pd.isna(merge_date):
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(
                        "Strategy4 dataset: project %s row skipped (missing merge_date_ts)",
                        project,
                    )
                continue
            builds_per_day = float(row.get("builds_per_day", 0.0))
            if builds_per_day <= 0 and project in build_counts_map:
                builds_per_day = float(build_counts_map.get(project, 0.0))
            observed_additional_builds = 0.0
            for det_days in detection_map.get(pd.to_datetime(merge_date).normalize(), []):
                if builds_per_day > 0:
                    observed_additional_builds += det_days * builds_per_day
            schedule_row: Dict[str, object] = {
                "project": project,
                "merge_date": merge_date,
                "day_index": int(row.get("day_index")) if not pd.isna(row.get("day_index")) else None,
                "observed_additional_builds": int(math.ceil(observed_additional_builds)) if observed_additional_builds > 0 else 0,
                "builds_per_day": builds_per_day,
                "label_flag": bool(row.get("label_flag", False)),
                "label_source": label_name,
                "walkforward_fold": row.get("walkforward_fold"),
                "train_window_start": row.get("train_window_start"),
                "train_window_end": row.get("train_window_end"),
                "validation_window_start": row.get("validation_window_start"),
                "validation_window_end": row.get("validation_window_end"),
            }
            if derived_from_threshold:
                schedule_row["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                schedule_row[risk_column] = float(row.get(risk_column))

            for col in feature_cols:
                raw_value = row.get(col, np.nan)
                if pd.isna(raw_value):
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug(
                            "Strategy4 dataset: project %s date %s missing feature %s (default to 0.0)",
                            project,
                            merge_date,
                            col,
                        )
                    value = 0.0
                else:
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        value = 0.0
                schedule_row[col] = value

            rows.append(schedule_row)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Strategy4 dataset: no rows collected across projects.")
        return dataset
    required_columns = ["observed_additional_builds"] + list(feature_cols)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "Strategy4 dataset: collected rows=%d; ensuring required columns=%s",
            len(dataset),
            required_columns,
        )
    dataset = dataset.dropna(subset=required_columns).copy()
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Strategy4 dataset: rows after dropna=%d", len(dataset))
    return dataset


def _fit_linear_regression_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_column: str,
) -> RegressionModel:
    """Solve a linear regression model on the provided training frame."""

    if train_df.empty:
        raise ValueError("Training data for regression is empty.")

    X_train = train_df[list(feature_cols)].astype(float).to_numpy()
    y_train = train_df[target_column].astype(float).to_numpy()
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data lacks usable samples for regression.")

    X_design = np.hstack([np.ones((len(X_train), 1)), X_train])
    coef, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
    intercept = float(coef[0])
    weights = coef[1:]
    coefficients = {feature_cols[i]: float(weights[i]) for i in range(len(feature_cols))}
    return RegressionModel(intercept=intercept, coefficients=coefficients, feature_order=list(feature_cols))


def _train_linear_regression(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_column: str,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> Tuple[RegressionModel, Dict[str, float]]:
    """Fit a regression model with a project-level random train/test split."""

    projects = sorted(dataset["project"].unique())
    if not projects:
        raise ValueError("Regression dataset is empty.")
    rng = random.Random(random_seed)
    rng.shuffle(projects)
    split_idx = max(1, int(len(projects) * (1 - test_fraction)))
    train_projects = set(projects[:split_idx])
    test_projects = set(projects[split_idx:]) or train_projects
    train_df = dataset[dataset["project"].isin(train_projects)]
    test_df = dataset[dataset["project"].isin(test_projects)]

    model = _fit_linear_regression_model(train_df, feature_cols, target_column)

    def _mae(frame: pd.DataFrame) -> float:
        if frame.empty:
            return float("nan")
        preds = model.predict(frame[list(feature_cols)])
        return float(np.mean(np.abs(preds - frame[target_column].to_numpy())))

    metrics = {
        "train_mae": _mae(train_df),
        "test_mae": _mae(test_df),
        "train_projects": len(train_projects),
        "test_projects": len(test_projects),
    }
    return model, metrics


def _leave_one_project_out_predictions(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_column: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return predictions and error metrics for leave-one-project-out evaluation."""

    if dataset.empty:
        empty_metrics = {
            "evaluation_mode": "leave_one_project_out",
            "lopo_overall": {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0.0},
            "lopo_project_metrics": {},
        }
        return dataset.copy(), empty_metrics

    prediction_frames: List[pd.DataFrame] = []
    project_metrics: Dict[str, Dict[str, float]] = {}
    all_actual: List[np.ndarray] = []
    all_predicted: List[np.ndarray] = []

    for project in sorted(dataset["project"].unique()):
        test_df = dataset[dataset["project"] == project]
        train_df = dataset[dataset["project"] != project]
        if test_df.empty:
            continue
        if train_df.empty:
            # Unable to perform LOPO when only one project exists; skip metrics but preserve rows.
            fallback_predictions = test_df.copy()
            fallback_predictions["predicted_additional_builds_raw"] = np.nan
            fallback_predictions["lopo_training_projects"] = 0
            fallback_predictions["lopo_evaluation_project"] = project
            prediction_frames.append(fallback_predictions)
            project_metrics[project] = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": float(len(test_df))}
            continue

        model = _fit_linear_regression_model(train_df, feature_cols, target_column)
        raw_pred = model.predict(test_df[list(feature_cols)])
        prediction_frame = test_df.copy()
        prediction_frame["predicted_additional_builds_raw"] = raw_pred.astype(float)
        prediction_frame["lopo_training_projects"] = len(train_df["project"].unique())
        prediction_frame["lopo_evaluation_project"] = project
        prediction_frames.append(prediction_frame)

        actual = test_df[target_column].astype(float).to_numpy()
        errors = raw_pred - actual
        stats = _compute_error_statistics(errors, actual)
        stats["count"] = float(len(test_df))
        project_metrics[project] = stats
        all_actual.append(actual)
        all_predicted.append(raw_pred.astype(float))

    if prediction_frames:
        combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        combined_predictions = dataset.iloc[0:0].copy()

    if all_actual and all_predicted:
        overall_actual = np.concatenate(all_actual)
        overall_pred = np.concatenate(all_predicted)
        overall_stats = _compute_error_statistics(overall_pred - overall_actual, overall_actual)
        overall_stats["count"] = float(len(overall_actual))
    else:
        overall_stats = {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "count": 0.0}

    metrics = {
        "evaluation_mode": "leave_one_project_out",
        "lopo_overall": overall_stats,
        "lopo_project_metrics": project_metrics,
    }
    return combined_predictions, metrics


def strategy4_cross_project_regression(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    build_counts_path: Optional[str] = None,
    predictions_root: Optional[str] = None,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    feature_cols: Optional[Sequence[str]] = None,
    mode: str = "multi",
    evaluation_mode: str = "leave_one_project_out",
    test_fraction: float = 0.2,
    random_seed: int = 42,
    *,
    walkforward_splits: Optional[int] = None,
    use_recent_training: Optional[bool] = None,
    diagnostics_output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, RegressionModel, Dict[str, float]]:
    """Predict additional build counts via cross-project linear regression triggered by JIT labels."""

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()
    build_counts_df = _load_build_counts(build_counts_path)
    predictions_root = _resolve_path("phase5.predictions_root", predictions_root)
    normalized_mode = (mode or "multi").strip().lower()
    if normalized_mode not in _VALID_STRATEGY4_MODES:
        raise ValueError(f"Unsupported Strategy4 mode: {mode!r} (expected one of {_VALID_STRATEGY4_MODES})")
    if feature_cols is not None:
        resolved_features = tuple(_dedupe_preserve_order(str(col).strip() for col in feature_cols if str(col).strip()))
    elif normalized_mode == "simple":
        resolved_features = SIMPLE_REGRESSION_FEATURES
    else:
        resolved_features = tuple(_default_strategy4_features())
    if not resolved_features:
        raise ValueError("Strategy4 requires at least one feature column; provide --strategy4-feature or ensure prediction settings export features.")
    required_feature_columns: Tuple[str, ...] = tuple(resolved_features) if normalized_mode == "simple" else tuple()
    feature_sig = _feature_signature(resolved_features)
    if LOGGER.isEnabledFor(logging.INFO):
        LOGGER.info(
            "Strategy4 regression mode=%s, features=%d, signature=%s",
            normalized_mode,
            len(resolved_features),
            feature_sig[:12],
        )
    resolved_walkforward_splits, resolved_use_recent = _resolve_walkforward_config(
        walkforward_splits,
        use_recent_training,
    )
    diagnostics_path = (
        Path(diagnostics_output_path)
        if diagnostics_output_path is not None
        else Path(__file__).resolve().parent / "docs" / "reports" / "strategy4_diagnostics.json"
    )

    dataset = _build_regression_dataset(
        detection_df,
        timelines,
        build_counts_df,
        predictions_root,
        resolved_features,
        risk_column,
        label_column,
        threshold,
        walkforward_splits=walkforward_splits,
        use_recent_training=use_recent_training,
        required_feature_columns=required_feature_columns,
    )
    if dataset.empty:
        raise ValueError("Regression dataset is empty; ensure input CSV files are prepared.")

    dataset["observed_additional_builds"] = dataset["observed_additional_builds"].astype(float)
    evaluation_mode_normalized = (evaluation_mode or "").strip().lower()
    metrics: Dict[str, Any] = {}

    if evaluation_mode_normalized in {"leave_one_project_out", "lopo"}:
        predictions, lopo_metrics = _leave_one_project_out_predictions(
            dataset,
            resolved_features,
            target_column="observed_additional_builds",
        )
        metrics.update(lopo_metrics)
        model = _fit_linear_regression_model(dataset, resolved_features, "observed_additional_builds")
        # Fill missing predictions (e.g., single-project datasets) with the full-data model.
        if "predicted_additional_builds_raw" not in predictions.columns:
            raw_full = model.predict(predictions[list(resolved_features)])
            predictions["predicted_additional_builds_raw"] = raw_full.astype(float)
        else:
            missing_mask = predictions["predicted_additional_builds_raw"].isna()
            if missing_mask.any():
                fallback_raw = model.predict(predictions.loc[missing_mask, list(resolved_features)])
                predictions.loc[missing_mask, "predicted_additional_builds_raw"] = fallback_raw.astype(float)
        metrics["evaluation_mode"] = "leave_one_project_out"
        metrics["lopo_heldout_projects"] = [str(p) for p in sorted(dataset["project"].unique())]
        base_predictions = predictions
    elif evaluation_mode_normalized in {"random_project_split", "random_split", ""}:
        model, base_metrics = _train_linear_regression(
            dataset,
            resolved_features,
            target_column="observed_additional_builds",
            test_fraction=test_fraction,
            random_seed=random_seed,
        )
        metrics.update(base_metrics)
        metrics["evaluation_mode"] = "random_project_split"
        raw_predictions = model.predict(dataset[list(resolved_features)])
        base_predictions = dataset.copy()
        base_predictions["predicted_additional_builds_raw"] = raw_predictions.astype(float)
    else:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode!r}")

    predictions = base_predictions.copy()
    raw_predictions = predictions["predicted_additional_builds_raw"].astype(float).to_numpy()
    clipped = np.clip(raw_predictions, a_min=0.0, a_max=None)
    predictions["prediction_was_clipped"] = clipped != raw_predictions
    rounded = np.ceil(clipped)
    predictions["predicted_additional_builds"] = rounded.astype(int)
    predictions["rounding_mode_used"] = "ceil"
    predictions["model_version"] = _STRATEGY4_MODEL_VERSION
    predictions["feature_signature"] = feature_sig

    fold_metrics, overall_metrics = _summarise_fold_performance(
        predictions,
        target_column="observed_additional_builds",
        prediction_column="predicted_additional_builds_raw",
        fold_column="walkforward_fold",
    )
    metrics["fold_metrics"] = fold_metrics
    metrics["overall_performance"] = overall_metrics
    metrics["model_version"] = _STRATEGY4_MODEL_VERSION
    metrics["feature_signature"] = feature_sig
    metrics["random_state_used"] = getattr(prediction_settings, "RANDOM_STATE", None) if prediction_settings is not None else None
    metrics["use_hpo"] = bool(getattr(prediction_settings, "USE_HYPERPARAM_OPTIMIZATION", False)) if prediction_settings is not None else False
    metrics["walkforward_splits"] = resolved_walkforward_splits
    metrics["use_recent_training"] = resolved_use_recent
    metrics["regression_mode"] = normalized_mode
    metrics["resolved_feature_columns"] = list(resolved_features)
    metrics.setdefault("train_mae", float(np.mean(np.abs(model.predict(dataset[list(resolved_features)]) - dataset["observed_additional_builds"].to_numpy()))))
    metrics.setdefault("train_projects", len(dataset["project"].unique()))

    fold_details = _collect_fold_details(predictions)
    diagnostics_payload = {
        "model": {
            "version": _STRATEGY4_MODEL_VERSION,
            "intercept": model.intercept,
            "coefficients": model.coefficients,
            "feature_order": list(resolved_features),
            "feature_signature": feature_sig,
            "mode": normalized_mode,
        },
        "settings": {
            "random_state": metrics["random_state_used"],
            "use_hyperparam_optimization": metrics["use_hpo"],
            "walkforward_splits": resolved_walkforward_splits,
            "use_recent_training": resolved_use_recent,
        },
        "metrics": metrics,
        "folds": fold_details,
    }
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("w", encoding="utf-8") as fh:
        json.dump(diagnostics_payload, fh, indent=2)

    records: List[Dict[str, object]] = []
    for _, row in predictions.iterrows():
        project = row["project"]
        if project not in timelines:
            continue
        timeline = timelines[project]
        merge_date = row.get("merge_date")
        if pd.isna(merge_date):
            continue
        alignment = _align_to_timeline(timeline, pd.to_datetime(merge_date, utc=True))
        scheduled = int(row["predicted_additional_builds"])
        if scheduled <= 0:
            continue
        if not bool(row.get("label_flag", False)):
            continue
        record: Dict[str, object] = {
            "project": project,
            "strategy": "cross_project_regression",
            "merge_date": alignment.merge_date,
            "day_index": alignment.day_index,
            "scheduled_additional_builds": scheduled,
            "observed_additional_builds": float(row.get("observed_additional_builds", np.nan)),
            "label_flag": bool(row.get("label_flag", False)),
            "label_source": row.get("label_source"),
        }
        for meta_field in (
            "walkforward_fold",
            "train_window_start",
            "train_window_end",
            "validation_window_start",
            "validation_window_end",
        ):
            if meta_field in row and not pd.isna(row.get(meta_field)):
                record[meta_field] = row.get(meta_field)
        record["predicted_additional_builds_raw"] = float(row.get("predicted_additional_builds_raw", np.nan))
        record["rounding_mode_used"] = row.get("rounding_mode_used", "ceil")
        record["prediction_was_clipped"] = bool(row.get("prediction_was_clipped", False))
        if "label_threshold" in row and not pd.isna(row.get("label_threshold")):
            record["label_threshold"] = float(row.get("label_threshold"))
        if risk_column in row and not pd.isna(row.get(risk_column)):
            record[risk_column] = float(row.get(risk_column))
        record["model_version"] = _STRATEGY4_MODEL_VERSION
        record["feature_signature"] = feature_sig
        record["regression_mode"] = normalized_mode
        records.append(record)

    schedule_df = pd.DataFrame(records)
    return schedule_df, model, metrics


if __name__ == "__main__":
    raise SystemExit("This module exposes strategy utilities and is not intended for CLI execution.")
