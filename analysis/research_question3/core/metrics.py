"""Aggregation helpers shared across RQ3 simulations."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

__all__ = [
    "summarize_schedule",
    "summarize_schedule_by_project",
    "safe_ratio",
    "prepare_project_metrics",
    "aggregate_strategy_metrics",
    "prepare_daily_totals",
]


def summarize_schedule(
    name: str,
    df: pd.DataFrame,
    value_column: str = "scheduled_additional_builds",
) -> Dict[str, object]:
    """Convert a strategy schedule into a single-row metric summary."""

    if df.empty:
        return {
            "strategy": name,
            "rows": 0,
            "unique_projects": 0,
            "unique_days": 0,
            value_column: 0.0,
        }

    summary: Dict[str, object] = {
        "strategy": name,
        "rows": int(len(df)),
        "unique_projects": int(df["project"].nunique()) if "project" in df.columns else 0,
        "unique_days": int(df["merge_date"].nunique())
        if "merge_date" in df.columns
        else int(df["merge_date_ts"].nunique()),
        value_column: float(df[value_column].fillna(0).sum())
        if value_column in df.columns
        else 0.0,
    }

    if "median_detection_builds" in df.columns:
        summary["median_detection_builds_mean"] = float(
            df["median_detection_builds"].dropna().mean()
        )
    elif "median_detection_days" in df.columns:
        summary["median_detection_days_mean"] = float(
            df["median_detection_days"].dropna().mean()
        )
    if "sampled_offset_builds" in df.columns:
        summary["sampled_offset_builds_mean"] = float(
            df["sampled_offset_builds"].dropna().mean()
        )
    elif "sampled_offset_days" in df.columns:
        summary["sampled_offset_mean"] = float(
            df["sampled_offset_days"].dropna().mean()
        )
    if "normalized_line_change" in df.columns:
        summary["normalized_line_change_mean"] = float(
            df["normalized_line_change"].dropna().mean()
        )
    if "predicted_additional_builds" in df.columns:
        summary["predicted_additional_builds_sum"] = float(
            df["predicted_additional_builds"].fillna(0).sum()
        )

    return summary


def summarize_schedule_by_project(
    name: str,
    df: pd.DataFrame,
    value_column: str = "scheduled_additional_builds",
) -> pd.DataFrame:
    """Aggregate a strategy schedule per project."""

    columns = ["strategy", "project", "rows", "unique_days", value_column]
    if df.empty or "project" not in df.columns:
        return pd.DataFrame(columns=columns)

    day_column: Optional[str]
    if "merge_date" in df.columns:
        day_column = "merge_date"
    elif "merge_date_ts" in df.columns:
        day_column = "merge_date_ts"
    else:
        day_column = None

    grouped = df.groupby("project", dropna=False)
    result = grouped.size().rename("rows").to_frame()

    if day_column is not None:
        unique_days = grouped[day_column].nunique(dropna=True).rename("unique_days")
        result = result.join(unique_days, how="left")
    else:
        result["unique_days"] = 0

    if value_column in df.columns:
        totals = grouped[value_column].sum(min_count=1).rename(value_column)
        result = result.join(totals, how="left")
    else:
        result[value_column] = 0.0

    if "median_detection_builds" in df.columns:
        result["median_detection_builds_mean"] = grouped["median_detection_builds"].mean()
    elif "median_detection_days" in df.columns:
        result["median_detection_days_mean"] = grouped["median_detection_days"].mean()
    if "sampled_offset_builds" in df.columns:
        result["sampled_offset_builds_mean"] = grouped["sampled_offset_builds"].mean()
    elif "sampled_offset_days" in df.columns:
        result["sampled_offset_mean"] = grouped["sampled_offset_days"].mean()
    if "normalized_line_change" in df.columns:
        result["normalized_line_change_mean"] = grouped["normalized_line_change"].mean()
    if "predicted_additional_builds" in df.columns:
        total_predicted = grouped["predicted_additional_builds"].sum(min_count=1)
        result["predicted_additional_builds_sum"] = total_predicted

    result = result.reset_index()
    result.insert(0, "strategy", name)

    result["rows"] = result["rows"].astype(int)
    result["unique_days"] = result["unique_days"].fillna(0).astype(int)
    result[value_column] = result[value_column].fillna(0).astype(float)
    for col in (
        "median_detection_builds_mean",
        "median_detection_days_mean",
        "sampled_offset_builds_mean",
        "sampled_offset_mean",
        "normalized_line_change_mean",
        "predicted_additional_builds_sum",
    ):
        if col in result.columns:
            result[col] = result[col].astype(float)

    ordered_cols = columns + [col for col in result.columns if col not in columns]
    return result[ordered_cols]


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a safe division result, producing ``nan`` when denominator is zero."""

    return numerator / denominator if denominator else float("nan")


def prepare_project_metrics(
    schedules: Dict[str, pd.DataFrame],
    baseline_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-project schedule summaries with baseline detection metrics."""

    project_frames: List[pd.DataFrame] = []
    for strategy_name, schedule in schedules.items():
        summary = summarize_schedule_by_project(strategy_name, schedule)
        if summary.empty:
            project_frames.append(summary)
            continue
        merged = summary.merge(baseline_df, on="project", how="left")
        merged.rename(
            columns={
                "rows": "trigger_count",
                "unique_days": "trigger_days",
                "scheduled_additional_builds": "scheduled_builds",
            },
            inplace=True,
        )
        merged["avg_builds_per_trigger"] = merged.apply(
            lambda row: row["scheduled_builds"] / row["trigger_count"]
            if row["trigger_count"]
            else 0.0,
            axis=1,
        )
        merged["builds_per_day"] = merged["builds_per_day"].fillna(0.0)
        merged["baseline_detection_builds"] = merged["baseline_detection_builds"].fillna(
            0.0
        )
        merged["baseline_detection_days"] = merged["baseline_detection_days"].fillna(0.0)

        def _estimate_residual_days(row: pd.Series) -> float:
            if row["builds_per_day"] <= 0:
                return row["baseline_detection_days"]
            delta_days = row["scheduled_builds"] / row["builds_per_day"]
            return max(row["baseline_detection_days"] - delta_days, 0.0)

        merged["estimated_detection_days"] = merged.apply(_estimate_residual_days, axis=1)
        merged["estimated_detection_builds"] = merged.apply(
            lambda row: max(
                row["baseline_detection_builds"] - row["scheduled_builds"], 0.0
            ),
            axis=1,
        )
        project_frames.append(merged)

    if not project_frames:
        return pd.DataFrame()
    return pd.concat(project_frames, ignore_index=True, sort=False)


def aggregate_strategy_metrics(project_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-strategy totals and central tendency measures."""

    records: List[Dict[str, float]] = []
    for strategy, group in project_df.groupby("strategy"):
        if group.empty:
            continue
        record = {
            "strategy": strategy,
            "projects": int(group["project"].nunique()),
            "total_scheduled_builds": float(group["scheduled_builds"].sum()),
            "median_scheduled_builds": float(group["scheduled_builds"].median()),
            "mean_scheduled_builds": float(group["scheduled_builds"].mean()),
            "median_estimated_detection_days": float(
                group["estimated_detection_days"].median()
            ),
            "mean_estimated_detection_days": float(
                group["estimated_detection_days"].mean()
            ),
            "median_estimated_detection_builds": float(
                group["estimated_detection_builds"].median()
            ),
            "mean_estimated_detection_builds": float(
                group["estimated_detection_builds"].mean()
            ),
            "total_estimated_builds_saved": float(
                (
                    group["baseline_detection_builds"]
                    - group["estimated_detection_builds"]
                ).sum()
            ),
        }
        records.append(record)
    return pd.DataFrame(records)


def prepare_daily_totals(schedules: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Produce per-strategy daily totals of scheduled additional builds."""

    frames: List[pd.DataFrame] = []
    for strategy_name, schedule in schedules.items():
        if schedule.empty:
            continue
        daily = schedule.copy()
        if "merge_date_ts" in daily.columns:
            date_column: Optional[str] = "merge_date_ts"
        elif "merge_date" in daily.columns:
            date_column = "merge_date"
        else:
            date_column = None

        if date_column is None:
            continue

        daily["date"] = pd.to_datetime(daily[date_column], errors="coerce")
        daily = daily.dropna(subset=["date"])
        grouped = (
            daily.groupby(["project", "date"])["scheduled_additional_builds"].sum()
            .reset_index()
        )
        grouped.insert(0, "strategy", strategy_name)
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(
            columns=["strategy", "project", "date", "scheduled_additional_builds"]
        )
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(["strategy", "project", "date"], inplace=True)
    return result
