#!/usr/bin/env python3
"""Phase 3 additional build strategies for RQ3.

This module implements four scheduling strategies that consume the Phase 2 build
timeline outputs and the detection time table extracted from OSS-Fuzz metadata.
Each strategy emits a per-project schedule that can be consumed by the Phase 5
simulation framework.

Inputs (default locations are relative to this file):
- `../phase2_outputs/build_timelines/*.csv` for daily build timelines.
- `../rq3_dataset/detection_time_results.csv` for commit/detection intervals.
- `../rq3_dataset/project_build_counts.csv` for baseline build frequency.
- `../data/<project>/<project>_daily_aggregated_metrics.csv` for daily code
  churn metrics (used by strategies 3 and 4).
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Default relative paths (resolved at runtime)
_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_TIMELINE_DIR = os.path.join(_BASE_DIR, "phase2_outputs", "build_timelines")
_DEFAULT_DETECTION_PATH = os.path.join(_BASE_DIR, "../rq3_dataset", "detection_time_results.csv")
_DEFAULT_BUILD_COUNTS_PATH = os.path.join(_BASE_DIR, "../rq3_dataset", "project_build_counts.csv")
_DEFAULT_DATA_DIR = os.path.join(_BASE_DIR, "../data")
_DEFAULT_PREDICTIONS_ROOT = os.path.join(_BASE_DIR, "../outputs", "results", "xgboost")

RISK_COLUMN = "predicted_risk_VCCFinder_Coverage"


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


def _load_detection_table(path: str = _DEFAULT_DETECTION_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["reported_date"] = _ensure_datetime(df.get("reported_date"))
    df["commit_date"] = _ensure_datetime(df.get("commit_date"))
    df["detection_time_days"] = pd.to_numeric(df.get("detection_time_days"), errors="coerce")
    df["project"] = df.get("package_name", "").astype(str).str.strip()
    df = df[df["project"] != ""].copy()
    return df


def _load_build_timelines(directory: str = _DEFAULT_TIMELINE_DIR) -> Dict[str, pd.DataFrame]:
    timelines: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(directory):
        return timelines
    for entry in os.listdir(directory):
        if not entry.endswith("_build_timeline.csv"):
            continue
        project = entry.replace("_build_timeline.csv", "")
        path = os.path.join(directory, entry)
        df = pd.read_csv(path)
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


def _load_project_metrics(project: str, data_dir: str = _DEFAULT_DATA_DIR) -> Optional[pd.DataFrame]:
    csv_path = os.path.join(data_dir, project, f"{project}_daily_aggregated_metrics.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "merge_date" not in df.columns:
        return None
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


def _load_build_counts(path: str = _DEFAULT_BUILD_COUNTS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["project"] = df.get("project", "").astype(str).str.strip()
    df["builds_per_day"] = pd.to_numeric(df.get("builds_per_day"), errors="coerce").fillna(0).astype(float)
    return df[df["project"] != ""].copy()


def _prediction_csv_path(project: str, predictions_root: str) -> Optional[str]:
    if not predictions_root or not os.path.exists(predictions_root):
        return None
    candidate = os.path.join(predictions_root, project, f"{project}_daily_aggregated_metrics_with_predictions.csv")
    if os.path.isfile(candidate):
        return candidate
    alternate = os.path.join(predictions_root, f"{project}_daily_aggregated_metrics_with_predictions.csv")
    if os.path.isfile(alternate):
        return alternate
    return None


def _load_prediction_frame(
    project: str,
    predictions_root: str,
) -> Optional[pd.DataFrame]:
    path = _prediction_csv_path(project, predictions_root)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "merge_date" not in df.columns:
        return None
    df = df.copy()
    df["merge_date_ts"] = pd.to_datetime(df["merge_date"], utc=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["merge_date_ts"])
    return df


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
    predictions_root: str,
    risk_column: str,
    label_column: Optional[str],
    threshold: float,
    extra_columns: Optional[Sequence[str]] = None,
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
    merge_cols = ["merge_date_ts", "_strategy_label"]
    risk_present = risk_column in predictions.columns
    if risk_present:
        merge_cols.append(risk_column)
    if extra_columns:
        for col in extra_columns:
            if col in predictions.columns and col not in merge_cols:
                merge_cols.append(col)
    merged = pd.merge(timeline, predictions[merge_cols], on="merge_date_ts", how="left")
    merged["_strategy_label"] = merged["_strategy_label"].fillna(False).astype(bool)
    if risk_present:
        merged[risk_column] = pd.to_numeric(merged[risk_column], errors="coerce")
    else:
        merged[risk_column] = np.nan
    return merged, label_name, threshold_used, derived_from_threshold


def strategy1_median_schedule(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    predictions_root: str = _DEFAULT_PREDICTIONS_ROOT,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Trigger median-sized additional builds on days flagged by JIT labels.

    For each project we compute the median detection delay observed in
    `detection_time_results.csv`. When the specified JIT prediction dataset marks
    a calendar day as vulnerable (`predicted_label_*` or risk >= threshold), we
    schedule `ceil(median_detection_days * builds_per_day)` additional builds for
    that day on the build timeline.
    """

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()

    valid = detection_df.dropna(subset=["project", "detection_time_days"])
    valid = valid[valid["detection_time_days"].astype(float) >= 0]
    median_map = (
        valid.assign(project=valid["project"].astype(str).str.strip())
        .groupby("project")["detection_time_days"]
        .median()
        .astype(float)
        .to_dict()
    )

    records: List[Dict[str, object]] = []
    for project, median_days in median_map.items():
        if project not in timelines or not math.isfinite(median_days):
            continue
        timeline = timelines[project]
        if timeline.empty:
            continue
        merged, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project, timeline, predictions_root, risk_column, label_column, threshold
        )
        if merged is None:
            continue
        positive = merged["_strategy_label"]
        if not positive.any():
            continue
        for _, row in merged.loc[positive].iterrows():
            median_factor = max(float(median_days), 0.0)
            builds_per_day = float(row.get("builds_per_day", 0.0))
            scheduled = int(math.ceil(median_factor * builds_per_day)) if builds_per_day > 0 else int(math.ceil(median_factor))
            if scheduled <= 0:
                continue
            record: Dict[str, object] = {
                "project": project,
                "strategy": "median_label_trigger",
                "merge_date": row.get("merge_date_ts"),
                "day_index": int(row.get("day_index", 0)) if not pd.isna(row.get("day_index")) else None,
                "builds_per_day": builds_per_day,
                "median_detection_days": median_factor,
                "scheduled_additional_builds": scheduled,
                "label_source": label_name,
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            records.append(record)

    return pd.DataFrame(records)

# TODO：最大値は中央値，最小値は１にする
def strategy2_random_within_median_range(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    predictions_root: str = _DEFAULT_PREDICTIONS_ROOT,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Schedule random-sized additional builds only on days flagged by the JIT labels."""

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()

    stats: Dict[str, Tuple[float, float]] = {}
    grouped = detection_df.dropna(subset=["project", "detection_time_days"]).groupby("project")
    for project, group in grouped:
        durations = group["detection_time_days"].astype(float)
        durations = durations[durations > 0]
        if durations.empty:
            continue
        q1 = float(durations.quantile(0.25))
        q3 = float(durations.quantile(0.75))
        if not math.isfinite(q1):
            q1 = float(durations.min())
        if not math.isfinite(q3):
            q3 = float(durations.max())
        lower = max(q1, 0.0)
        upper = max(q3, lower)
        stats[project] = (lower, upper)

    records: List[Dict[str, object]] = []
    for project, timeline in (timelines or {}).items():
        if project not in stats or timeline.empty:
            continue
        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
        )
        if labelled is None:
            continue
        lower, upper = stats[project]
        if upper <= 0:
            continue
        rng = random.Random((hash(project) ^ random_seed) & 0xFFFFFFFF)
        positive = labelled["_strategy_label"]
        if not positive.any():
            continue
        for _, row in labelled.loc[positive].iterrows():
            if upper == lower:
                sampled_offset = upper
            else:
                sampled_offset = rng.uniform(lower, upper)
            builds_per_day = float(row.get("builds_per_day", 0.0))
            scheduled_builds = int(math.ceil(sampled_offset * builds_per_day)) if builds_per_day > 0 else int(math.ceil(sampled_offset))
            if scheduled_builds <= 0:
                continue
            record: Dict[str, object] = {
                "project": project,
                "strategy": "median_iqr_random",
                "merge_date": row.get("merge_date_ts"),
                "day_index": int(row.get("day_index", 0)) if not pd.isna(row.get("day_index")) else None,
                "builds_per_day": builds_per_day,
                "offset_days_q1": lower,
                "offset_days_q3": upper,
                "sampled_offset_days": sampled_offset,
                "scheduled_additional_builds": scheduled_builds,
                "label_source": label_name,
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            records.append(record)
    return pd.DataFrame(records)


def strategy3_line_change_proportional(
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    data_dir: str = _DEFAULT_DATA_DIR,
    predictions_root: str = _DEFAULT_PREDICTIONS_ROOT,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    scaling_factor: float = 0.5,
    clip_max: float = 5.0,
) -> pd.DataFrame:
    """Adjust additional build frequency proportional to daily line churn, gated by JIT labels."""

    timelines = timelines if timelines is not None else _load_build_timelines()

    rows: List[Dict[str, object]] = []
    for project, timeline in timelines.items():
        if timeline.empty:
            continue
        metrics_df = _prepare_line_change_metrics(project, data_dir)
        if metrics_df is None:
            continue
        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
        )
        if labelled is None:
            continue
        merged = pd.merge(
            labelled,
            metrics_df[["merge_date_ts", "line_change_total", "daily_commit_count"]],
            on="merge_date_ts",
            how="left",
        ).reset_index(drop=True)
        positive_mask = merged["_strategy_label"].fillna(False)
        if not positive_mask.any():
            continue
        line_change = merged["line_change_total"].fillna(0).astype(float)
        positive_values = line_change[line_change > 0]
        baseline = float(positive_values.median()) if not positive_values.empty else float(line_change.median())
        if not math.isfinite(baseline) or baseline <= 0:
            baseline = max(float(line_change.max()), 1.0)
        normalized = (line_change / baseline).clip(lower=0, upper=clip_max)
        expected_extra = normalized * scaling_factor
        scheduled_extra = expected_extra.round().astype(int)
        commit_count_columns = [col for col in merged.columns if col.startswith("daily_commit_count")]
        if commit_count_columns:
            commit_count_series = merged[commit_count_columns[0]].astype(float)
            for col in commit_count_columns[1:]:
                commit_count_series = commit_count_series.combine_first(merged[col].astype(float))
        else:
            commit_count_series = pd.Series(np.nan, index=merged.index, dtype=float)

        for idx, row in merged.loc[positive_mask].iterrows():
            scheduled = int(max(scheduled_extra.loc[idx], 0))
            if scheduled <= 0:
                continue
            record: Dict[str, object] = {
                "project": project,
                "strategy": "line_change_proportional",
                "merge_date": row.merge_date_ts,
                "day_index": int(row.day_index) if not pd.isna(row.day_index) else None,
                "builds_per_day": float(row.builds_per_day) if not pd.isna(row.builds_per_day) else np.nan,
                "line_change_total": float(line_change.loc[idx]),
                "normalized_line_change": float(normalized.loc[idx]),
                "expected_additional_builds": float(expected_extra.loc[idx]),
                "scheduled_additional_builds": scheduled,
                "daily_commit_count": float(commit_count_series.loc[idx]) if not pd.isna(commit_count_series.loc[idx]) else np.nan,
                "label_source": label_name,
            }
            if derived_from_threshold:
                record["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                record[risk_column] = float(row.get(risk_column))
            rows.append(record)

    return pd.DataFrame(rows)

# TODO：直近過去で代替を削除
def _build_regression_dataset(
    detection_df: pd.DataFrame,
    timelines: Dict[str, pd.DataFrame],
    build_counts_df: pd.DataFrame,
    predictions_root: str,
    feature_cols: Sequence[str],
    risk_column: str,
    label_column: Optional[str],
    threshold: float,
) -> pd.DataFrame:
    build_counts_map = dict(zip(build_counts_df["project"], build_counts_df["builds_per_day"]))
    detection_lookup: Dict[str, Dict[pd.Timestamp, List[float]]] = {}
    for _, record in detection_df.iterrows():
        project = (record.get("project") or "").strip()
        commit_ts = record.get("commit_date")
        detection_time = record.get("detection_time_days")
        if not project or pd.isna(commit_ts) or pd.isna(detection_time):
            continue
        commit_norm = pd.to_datetime(commit_ts).normalize()
        detection_lookup.setdefault(project, {}).setdefault(commit_norm, []).append(float(detection_time))

    rows: List[Dict[str, object]] = []
    for project, timeline in timelines.items():
        if timeline.empty:
            continue
        extra_columns = [col for col in feature_cols if col not in {"builds_per_day", "label_flag"}]
        labelled, label_name, threshold_used, derived_from_threshold = _prepare_labelled_timeline(
            project,
            timeline,
            predictions_root,
            risk_column,
            label_column,
            threshold,
            extra_columns=extra_columns,
        )
        if labelled is None:
            continue
        labelled = labelled.copy()
        labelled["label_flag"] = labelled["_strategy_label"].astype(bool)

        detection_map = detection_lookup.get(project, {})
        for _, row in labelled.iterrows():
            merge_date = row.get("merge_date_ts")
            if pd.isna(merge_date):
                continue
            builds_per_day = float(row.get("builds_per_day", 0.0))
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
            }
            if derived_from_threshold:
                schedule_row["label_threshold"] = threshold_used
            if risk_column in row and not pd.isna(row.get(risk_column)):
                schedule_row[risk_column] = float(row.get(risk_column))

            for col in feature_cols:
                raw_value = row.get(col, np.nan)
                if pd.isna(raw_value):
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
        return dataset
    return dataset.dropna(subset=["observed_additional_builds"] + list(feature_cols)).copy()


def _train_linear_regression(
    dataset: pd.DataFrame,
    feature_cols: Sequence[str],
    target_column: str,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> Tuple[RegressionModel, Dict[str, float]]:
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

    X_train = train_df[list(feature_cols)].astype(float).to_numpy()
    y_train = train_df[target_column].astype(float).to_numpy()
    X_design = np.hstack([np.ones((len(X_train), 1)), X_train])
    coef, _, _, _ = np.linalg.lstsq(X_design, y_train, rcond=None)
    intercept = float(coef[0])
    weights = coef[1:]
    coefficients = {feature_cols[i]: float(weights[i]) for i in range(len(feature_cols))}
    model = RegressionModel(intercept=intercept, coefficients=coefficients, feature_order=list(feature_cols))

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


def strategy4_cross_project_regression(
    detection_df: Optional[pd.DataFrame] = None,
    timelines: Optional[Dict[str, pd.DataFrame]] = None,
    build_counts_path: str = _DEFAULT_BUILD_COUNTS_PATH,
    predictions_root: str = _DEFAULT_PREDICTIONS_ROOT,
    risk_column: str = RISK_COLUMN,
    label_column: Optional[str] = None,
    threshold: float = 0.5,
    feature_cols: Sequence[str] = ("daily_commit_count", "files_changed", "lines_added", "lines_deleted", "builds_per_day", "label_flag"),
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, RegressionModel, Dict[str, float]]:
    """Predict additional build counts via cross-project linear regression triggered by JIT labels."""

    detection_df = detection_df if detection_df is not None else _load_detection_table()
    timelines = timelines if timelines is not None else _load_build_timelines()
    build_counts_df = _load_build_counts(build_counts_path)

    dataset = _build_regression_dataset(
        detection_df,
        timelines,
        build_counts_df,
        predictions_root,
        feature_cols,
        risk_column,
        label_column,
        threshold,
    )
    if dataset.empty:
        raise ValueError("Regression dataset is empty; ensure input CSV files are prepared.")

    dataset["observed_additional_builds"] = dataset["observed_additional_builds"].astype(float)
    model, metrics = _train_linear_regression(
        dataset,
        feature_cols,
        target_column="observed_additional_builds",
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

    predictions = dataset.copy()
    predictions["predicted_additional_builds"] = model.predict(predictions[list(feature_cols)])
    predictions["predicted_additional_builds"] = predictions["predicted_additional_builds"].clip(lower=0)
    predictions["predicted_additional_builds"] = np.ceil(predictions["predicted_additional_builds"]).astype(int)

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
        if "label_threshold" in row and not pd.isna(row.get("label_threshold")):
            record["label_threshold"] = float(row.get("label_threshold"))
        if risk_column in row and not pd.isna(row.get(risk_column)):
            record[risk_column] = float(row.get(risk_column))
        records.append(record)

    schedule_df = pd.DataFrame(records)
    return schedule_df, model, metrics


if __name__ == "__main__":
    raise SystemExit("This module exposes strategy utilities and is not intended for CLI execution.")
