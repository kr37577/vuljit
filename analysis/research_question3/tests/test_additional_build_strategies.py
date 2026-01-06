from __future__ import annotations

import hashlib
import json
import importlib.util
import math
import pandas as pd
import pytest
import sys
import types
from pathlib import Path
import numpy as np
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "core"

if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(CORE_DIR)]
    sys.modules["core"] = core_pkg
    io_spec = importlib.util.spec_from_file_location("core.io", CORE_DIR / "io.py")
    io_module = importlib.util.module_from_spec(io_spec)
    io_spec.loader.exec_module(io_module)
    sys.modules["core.io"] = io_module
    setattr(core_pkg, "io", io_module)

strategies_spec = importlib.util.spec_from_file_location(
    "additional_build_strategies_under_test",
    ROOT / "additional_build_strategies.py",
)
strategies = importlib.util.module_from_spec(strategies_spec)
sys.modules["additional_build_strategies_under_test"] = strategies
strategies_spec.loader.exec_module(strategies)


def _one_day_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "merge_date_ts": [
                pd.Timestamp("2024-02-01", tz="UTC"),
                pd.Timestamp("2024-02-02", tz="UTC"),
            ],
            "day_index": [1, 2],
            "builds_per_day": [2.0, 2.0],
        }
    )


def _fake_labelled_timeline(
    project: str, timeline: pd.DataFrame, folds: list[str], train_ends: list[pd.Timestamp]
) -> pd.DataFrame:
    labelled = timeline.copy()
    labelled["_strategy_label"] = True
    labelled["project"] = project
    labelled["walkforward_fold"] = folds
    labelled["train_window_end"] = train_ends
    return labelled


def _expected_uniform(project: str, fold: str, base_seed: int, lower: float, upper: float) -> float:
    token = f"{project}::{fold}::{base_seed}"
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
    seed_int = int.from_bytes(digest, "little")
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return float(rng.uniform(lower, upper))


def _sample_detection_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": ["alpha", "alpha", "alpha", "beta", "beta", "alpha", "gamma"],
            "commit_date": [
                "2024-01-05T00:00:00+00:00",
                "2024-01-08T00:00:00+00:00",
                "2024-02-15T00:00:00+00:00",
                "2024-03-01T00:00:00+00:00",
                "2024-03-05T00:00:00+00:00",
                "2023-12-01T00:00:00+00:00",
                "2024-04-01T00:00:00+00:00",
            ],
            "detection_time_days": [2.0, 6.0, 14.0, 3.0, 9.0, -5.0, float("nan")],
        }
    )


def _fold_metadata() -> dict[str, dict[str, dict[str, object]]]:
    return {
        "alpha": {
            "fold-1": {
                "train_start": pd.Timestamp("2024-01-01", tz="UTC"),
                "train_end": pd.Timestamp("2024-01-31", tz="UTC"),
                "train_indices": list(range(10)),
                "test_indices": list(range(10, 15)),
            },
            "fold-2": {
                "train_start": pd.Timestamp("2024-02-01", tz="UTC"),
                "train_end": pd.Timestamp("2024-02-28", tz="UTC"),
                "train_indices": list(range(15, 20)),
                "test_indices": list(range(20, 25)),
            },
        },
        "beta": {
            "fold-1": {
                "train_start": pd.Timestamp("2024-02-01", tz="UTC"),
                "train_end": pd.Timestamp("2024-02-28", tz="UTC"),
                "train_indices": list(range(5)),
                "test_indices": list(range(5, 8)),
            },
            "fold-2": {
                "train_start": pd.Timestamp("2024-03-01", tz="UTC"),
                "train_end": pd.Timestamp("2024-03-31", tz="UTC"),
                "train_indices": list(range(8, 12)),
                "test_indices": list(range(12, 16)),
            },
        },
    }


def _strategy3_labelled_frame() -> pd.DataFrame:
    dates = [
        pd.Timestamp("2024-02-01", tz="UTC"),
        pd.Timestamp("2024-02-02", tz="UTC"),
    ]
    return pd.DataFrame(
        {
            "merge_date_ts": dates,
            "day_index": [1, 2],
            "builds_per_day": [2.0, 2.0],
            "_strategy_label": [True, True],
            "walkforward_fold": ["fold-1", "fold-1"],
        }
    )


def _strategy3_metrics_frame() -> pd.DataFrame:
    dates = [
        pd.Timestamp("2024-02-01", tz="UTC"),
        pd.Timestamp("2024-02-02", tz="UTC"),
    ]
    return pd.DataFrame(
        {
            "merge_date_ts": dates,
            "line_change_total": [30.0, 10.0],
            "daily_commit_count": [5.0, 3.0],
        }
    )


def _run_strategy3_with_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    timeline: pd.DataFrame,
    labelled: pd.DataFrame,
    metrics: pd.DataFrame,
    stats: dict[str, dict[str, dict[str, float]]],
    rounding_kwargs: Optional[dict[str, object]] = None,
    strategy_kwargs: Optional[dict[str, object]] = None,
) -> pd.DataFrame:
    detection_df = _sample_detection_table()

    def fake_prepare_labelled(project, timeline_frame, *_args, **_kwargs):
        assert project == "alpha"
        return labelled.copy(), "label", 0.5, False

    def fake_prepare_metrics(project, data_dir):
        assert project == "alpha"
        return metrics.copy()

    def fake_compute_stats(df, **_kwargs):
        assert df is detection_df
        return stats

    fold_metadata = {"assignments": pd.DataFrame(), "folds": {}}

    def fake_walkforward_metadata(project, *_args, **_kwargs):
        assert project == "alpha"
        return fold_metadata

    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare_labelled)
    monkeypatch.setattr(strategies, "_prepare_line_change_metrics", fake_prepare_metrics)
    monkeypatch.setattr(strategies, "_compute_project_fold_statistics", fake_compute_stats)
    monkeypatch.setattr(strategies, "_get_project_walkforward_metadata", fake_walkforward_metadata)

    kwargs = dict(rounding_kwargs or {})
    if strategy_kwargs:
        kwargs.update(strategy_kwargs)
    return strategies.strategy3_line_change_proportional(
        detection_df=detection_df,
        timelines={"alpha": timeline},
        data_dir=str(tmp_path),
        predictions_root=str(tmp_path),
        **kwargs,
    )


def test_strategy3_fold_budget_uses_fold_statistics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline()
    labelled = _strategy3_labelled_frame()
    metrics = _strategy3_metrics_frame()

    stats = {
        "__global__": {"median": 7.0, "count": 10.0},
        "alpha": {
            "__overall__": {"median": 5.0, "count": 20.0},
            "fold-1": {"median": 3.0, "count": 4.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
    )

    assert not result.empty
    assert set(result["project"]) == {"alpha"}
    assert set(result["fold_budget_source"]) == {"fold"}
    assert np.allclose(result["fold_budget"].to_numpy(dtype=float), 6.0)
    assert np.allclose(result["fold_sample_count"].to_numpy(dtype=float), 4.0)
    assert all(result["fold_positive_days"] == 2)


def test_strategy3_line_weight_share_uses_fold_baseline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline()
    labelled = _strategy3_labelled_frame()
    metrics = _strategy3_metrics_frame()

    stats = {
        "__global__": {"median": 8.0, "count": 10.0},
        "alpha": {
            "__overall__": {"median": 6.0, "count": 12.0},
            "fold-1": {"median": 4.0, "count": 5.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
    )

    assert len(result) == 2
    assert result["line_churn_baseline"].iloc[0] == pytest.approx(20.0)
    assert pytest.approx(result["line_weight_share"].sum()) == 1.0
    shares = dict(zip(result["day_index"], result["line_weight_share"]))
    assert pytest.approx(shares[1]) == 0.75
    assert pytest.approx(shares[2]) == 0.25
    assert not result["baseline_zero_fallback"].any()


def test_strategy3_line_weight_share_single_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline().iloc[:1].copy()
    labelled = timeline.copy()
    labelled["_strategy_label"] = [True]
    labelled["walkforward_fold"] = ["fold-2"]
    metrics = pd.DataFrame(
        {
            "merge_date_ts": [timeline["merge_date_ts"].iloc[0]],
            "line_change_total": [0.0],
            "daily_commit_count": [np.nan],
        }
    )

    stats = {
        "__global__": {"median": 8.0, "count": 10.0},
        "alpha": {
            "__overall__": {"median": 6.0, "count": 12.0},
            "fold-2": {"median": 5.0, "count": 3.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
    )

    assert len(result) == 1
    assert result["line_weight_share"].iloc[0] == pytest.approx(1.0)
    assert bool(result["baseline_zero_fallback"].iloc[0])
    assert result["line_churn_baseline"].iloc[0] == 0.0


def test_strategy3_fold_median_fallbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline()
    labelled = _strategy3_labelled_frame()
    metrics = _strategy3_metrics_frame()

    stats_project = {
        "__global__": {"median": 9.0, "count": 30.0},
        "alpha": {
            "__overall__": {"median": 4.0, "count": 12.0},
            "fold-1": {"median": float("nan"), "count": float("nan")},
        },
    }
    result_project = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats_project,
    )
    assert set(result_project["fold_budget_source"]) == {"project"}
    assert pytest.approx(result_project["fold_budget"].iloc[0]) == 8.0

    stats_global = {
        "__global__": {"median": 6.0, "count": 30.0},
        "alpha": {
            "__overall__": {"median": float("nan"), "count": float("nan")},
            "fold-1": {"median": float("nan"), "count": float("nan")},
        },
    }
    result_global = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats_global,
    )
    assert set(result_global["fold_budget_source"]) == {"global"}
    assert pytest.approx(result_global["fold_budget"].iloc[0]) == 12.0


def test_strategy3_expected_builds_and_rounding(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline()
    labelled = _strategy3_labelled_frame()
    metrics = _strategy3_metrics_frame()
    stats = {
        "__global__": {"median": 8.0, "count": 10.0},
        "alpha": {
            "__overall__": {"median": 6.0, "count": 12.0},
            "fold-1": {"median": 4.0, "count": 5.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
    )

    assert len(result) == 2
    expected_map = dict(zip(result["day_index"], result["expected_additional_builds_raw"]))
    assert pytest.approx(expected_map[1]) == 6.0
    assert pytest.approx(expected_map[2]) == 2.0
    rounded_map = dict(zip(result["day_index"], result["rounded_additional_builds"]))
    scheduled_map = dict(zip(result["day_index"], result["scheduled_additional_builds"]))
    assert rounded_map == scheduled_map
    assert set(result["rounding_mode_used"]) == {"ceil"}
    assert not result["fold_overflow_used"].any()
    assert pytest.approx(sum(scheduled_map.values())) == pytest.approx(result["fold_budget"].iloc[0])


def test_strategy3_overflow_adjusts_smallest_share(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = pd.DataFrame(
        {
            "merge_date_ts": [
                pd.Timestamp("2024-02-01", tz="UTC"),
                pd.Timestamp("2024-02-02", tz="UTC"),
            ],
            "day_index": [1, 2],
            "builds_per_day": [1.0, 1.0],
        }
    )
    labelled = timeline.copy()
    labelled["_strategy_label"] = [True, True]
    labelled["walkforward_fold"] = ["fold-3", "fold-3"]
    metrics = pd.DataFrame(
        {
            "merge_date_ts": timeline["merge_date_ts"],
            "line_change_total": [30.0, 20.0],
            "daily_commit_count": [np.nan, np.nan],
        }
    )
    stats = {
        "__global__": {"median": 5.0, "count": 8.0},
        "alpha": {
            "__overall__": {"median": 4.0, "count": 6.0},
            "fold-3": {"median": 1.5, "count": 4.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
    )

    assert len(result) == 2
    rounded_map = dict(zip(result["day_index"], result["rounded_additional_builds"]))
    assert rounded_map == {1: 2, 2: 2}
    scheduled_map = dict(zip(result["day_index"], result["scheduled_additional_builds"]))
    assert scheduled_map[1] == 2
    assert scheduled_map[2] == 1
    assert pytest.approx(sum(scheduled_map.values())) == pytest.approx(result["fold_budget"].iloc[0])
    assert result["fold_overflow_used"].any()


def test_strategy3_rounding_mode_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = pd.DataFrame(
        {
            "merge_date_ts": [
                pd.Timestamp("2024-02-01", tz="UTC"),
                pd.Timestamp("2024-02-02", tz="UTC"),
            ],
            "day_index": [1, 2],
            "builds_per_day": [1.0, 1.0],
        }
    )
    labelled = timeline.copy()
    labelled["_strategy_label"] = [True, True]
    labelled["walkforward_fold"] = ["fold-4", "fold-4"]
    metrics = pd.DataFrame(
        {
            "merge_date_ts": timeline["merge_date_ts"],
            "line_change_total": [25.0, 15.0],
            "daily_commit_count": [np.nan, np.nan],
        }
    )
    stats = {
        "__global__": {"median": 5.0, "count": 8.0},
        "alpha": {
            "__overall__": {"median": 4.0, "count": 6.0},
            "fold-4": {"median": 1.5, "count": 4.0},
        },
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
        rounding_kwargs={"rounding_mode": "floor"},
    )

    assert len(result) == 2
    assert set(result["rounding_mode_used"]) == {"floor"}
    rounded_map = dict(zip(result["day_index"], result["rounded_additional_builds"]))
    scheduled_map = dict(zip(result["day_index"], result["scheduled_additional_builds"]))
    assert rounded_map == scheduled_map
    assert scheduled_map[1] == 1
    assert scheduled_map[2] == 1
    assert not result["fold_overflow_used"].any()
    assert sum(scheduled_map.values()) < result["fold_budget"].iloc[0]


def test_strategy3_cross_project_uses_global_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    timeline = _one_day_timeline()
    labelled = _strategy3_labelled_frame().drop(columns=["walkforward_fold"])
    metrics = _strategy3_metrics_frame()

    stats = {
        "__global__": {"median": 5.0, "count": 20.0},
        "__global_exclusive__": {"alpha": {"median": 6.0, "count": 15.0}},
        "alpha": {"__overall__": {"median": float("nan"), "count": float("nan")}},
    }

    result = _run_strategy3_with_mocks(
        monkeypatch,
        tmp_path,
        timeline=timeline,
        labelled=labelled,
        metrics=metrics,
        stats=stats,
        strategy_kwargs={"mode": "cross_project", "global_budget": 10.0},
    )

    assert not result.empty
    assert result["strategy_mode"].unique().tolist() == ["cross_project"]
    assert set(result["fold_budget_source"]) == {"global_lopo"}

def test_compute_project_fold_statistics_handles_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = _sample_detection_table()
    build_counts_df = pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "builds_per_day": [2.0, 2.0],
        }
    )

    stats = strategies._compute_project_fold_statistics(
        detection_df,
        _fold_metadata(),
        build_counts_df=build_counts_df,
    )

    assert "__global__" in stats
    assert stats["__global__"]["median"] == pytest.approx(12.0)
    assert stats["__global__"]["q1"] == pytest.approx(6.0)
    assert stats["__global__"]["q3"] == pytest.approx(18.0)

    alpha_fold1 = stats["alpha"]["fold-1"]
    assert alpha_fold1["median"] == pytest.approx(8.0)
    assert alpha_fold1["q1"] <= alpha_fold1["median"] <= alpha_fold1["q3"]

    alpha_fold2 = stats["alpha"]["fold-2"]
    assert alpha_fold2["median"] == pytest.approx(28.0)
    assert alpha_fold2["q1"] == pytest.approx(28.0)
    assert alpha_fold2["q3"] == pytest.approx(28.0)

    beta_fold1 = stats["beta"]["fold-1"]
    assert math.isnan(beta_fold1["median"])
    assert math.isnan(beta_fold1["q1"])
    assert math.isnan(beta_fold1["q3"])

    beta_fold2 = stats["beta"]["fold-2"]
    assert beta_fold2["median"] == pytest.approx(12.0)
    assert beta_fold2["q1"] <= beta_fold2["median"] <= beta_fold2["q3"]

    assert stats["alpha"]["__overall__"]["median"] == pytest.approx(12.0)
    assert stats["beta"]["__overall__"]["median"] == pytest.approx(12.0)


def test_compute_project_fold_statistics_produces_lopo_stats() -> None:
    detection_df = pd.DataFrame(
        {
            "project": ["alpha", "alpha", "beta", "beta"],
            "commit_date": [
                "2024-01-05T00:00:00+00:00",
                "2024-01-15T00:00:00+00:00",
                "2024-02-01T00:00:00+00:00",
                "2024-02-08T00:00:00+00:00",
            ],
            "detection_time_days": [2.0, 4.0, 6.0, 8.0],
        }
    )

    build_counts_df = pd.DataFrame(
        {
            "project": ["alpha", "beta"],
            "builds_per_day": [2.0, 2.0],
        }
    )
    stats = strategies._compute_project_fold_statistics(
        detection_df,
        compute_lopo=True,
        build_counts_df=build_counts_df,
    )

    assert "__global_exclusive__" in stats
    lopo = stats["__global_exclusive__"]
    assert pytest.approx(lopo["alpha"]["median"]) == 14.0  # median of beta-only values
    assert pytest.approx(lopo["beta"]["median"]) == 6.0  # median of alpha-only values
    assert lopo["alpha"]["count"] == 2.0
    assert lopo["beta"]["count"] == 2.0


def test_strategy1_uses_peer_project_median(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = _sample_detection_table()
    timelines = {"alpha": _one_day_timeline()}

    fold_stats = {
        "__global__": {"median": 11.0, "q1": 7.0, "q3": 13.0},
        "alpha": {
            "__overall__": {"median": 5.0, "q1": 2.0, "q3": 8.0},
            "fold-1": {"median": 3.0, "q1": 1.0, "q3": 4.0},
            "fold-2": {"median": float("nan"), "q1": float("nan"), "q3": float("nan")},
        },
    }

    def fake_prepare(project, timeline, *_, **__):
        labelled = _fake_labelled_timeline(
            project,
            timeline,
            folds=["fold-1", "fold-2"],
            train_ends=[
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-28", tz="UTC"),
            ],
        )
        return labelled, "test_label", 0.5, False

    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare)
    monkeypatch.setattr(strategies, "_compute_project_fold_statistics", lambda *_args, **_kwargs: fold_stats)

    schedule = strategies.strategy1_median_schedule(
        detection_df=detection_df,
        timelines=timelines,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
    )

    assert list(schedule["project"]) == ["alpha", "alpha"]
    assert set(schedule["walkforward_fold"]) == {"fold-1", "fold-2"}

    fold1_row = schedule.set_index("walkforward_fold").loc["fold-1"]
    assert fold1_row["median_detection_builds"] == pytest.approx(3.0)
    assert fold1_row["scheduled_additional_builds"] == 3
    assert fold1_row["train_window_end"] == pd.Timestamp("2024-01-31", tz="UTC")

    fold2_row = schedule.set_index("walkforward_fold").loc["fold-2"]
    assert fold2_row["median_detection_builds"] == pytest.approx(5.0)
    assert fold2_row["scheduled_additional_builds"] == 5
    assert fold2_row["train_window_end"] == pd.Timestamp("2024-02-28", tz="UTC")


def test_strategy1_cross_project_uses_global_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = _sample_detection_table()
    timelines = {"alpha": _one_day_timeline()}

    fold_stats = {
        "__global__": {"median": 8.0, "q1": 4.0, "q3": 10.0},
        "__global_exclusive__": {"alpha": {"median": 9.0, "q1": 5.0, "q3": 12.0, "count": 4.0}},
        "alpha": {"__overall__": {"median": float("nan")}},
    }

    def fake_prepare(project, timeline, *_, **__):
        labelled = _fake_labelled_timeline(
            project,
            timeline,
            folds=["fold-1", "fold-2"],
            train_ends=[
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-28", tz="UTC"),
            ],
        )
        labelled = labelled.drop(columns=["walkforward_fold"])
        return labelled, "label", 0.5, False

    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare)
    monkeypatch.setattr(strategies, "_compute_project_fold_statistics", lambda *_args, **_kwargs: fold_stats)

    schedule = strategies.strategy1_median_schedule(
        detection_df=detection_df,
        timelines=timelines,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
        mode="cross_project",
    )

    assert not schedule.empty
    assert set(schedule["median_source"]) == {"global_lopo"}
    assert schedule["strategy_mode"].unique().tolist() == ["cross_project"]

def test_strategy2_random_range_uses_peer_medians(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = _sample_detection_table()
    timelines = {"alpha": _one_day_timeline()}

    fold_stats = {
        "__global__": {"median": 11.0, "q1": 7.0, "q3": 13.0},
        "alpha": {
            "__overall__": {"median": 5.0, "q1": 2.0, "q3": 8.0},
            "fold-1": {"median": 3.0, "q1": 1.0, "q3": 4.0},
            "fold-2": {"median": float("nan"), "q1": float("nan"), "q3": float("nan")},
        },
    }

    def fake_prepare(project, timeline, *_, **__):
        labelled = _fake_labelled_timeline(
            project,
            timeline,
            folds=["fold-1", "fold-2"],
            train_ends=[
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-28", tz="UTC"),
            ],
        )
        return labelled, "test_label", 0.5, False

    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare)
    monkeypatch.setattr(strategies, "_compute_project_fold_statistics", lambda *_args, **_kwargs: fold_stats)

    schedule = strategies.strategy2_random_within_median_range(
        detection_df=detection_df,
        timelines=timelines,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
        random_seed=123,
    )

    assert list(schedule["project"]) == ["alpha", "alpha"]

    fold1_expected_offset = _expected_uniform("alpha", "fold-1", 123, 1.0, 4.0)
    fold1_row = schedule.set_index("walkforward_fold").loc["fold-1"]
    assert fold1_row["offset_builds_q1"] == pytest.approx(1.0)
    assert fold1_row["offset_builds_q3"] == pytest.approx(4.0)
    assert fold1_row["sampled_offset_builds"] == pytest.approx(fold1_expected_offset)
    assert fold1_row["scheduled_additional_builds"] == math.ceil(fold1_expected_offset)

    fold2_row = schedule.set_index("walkforward_fold").loc["fold-2"]
    assert fold2_row["offset_builds_q1"] == pytest.approx(2.0)
    assert fold2_row["offset_builds_q3"] == pytest.approx(8.0)
    assert fold2_row["sampled_offset_builds"] == pytest.approx(_expected_uniform("alpha", "fold-2", 123, 2.0, 8.0))
    assert fold2_row["scheduled_additional_builds"] == math.ceil(fold2_row["sampled_offset_builds"])


def test_strategy2_cross_project_uses_global_quartiles(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = _sample_detection_table()
    timelines = {"alpha": _one_day_timeline()}

    fold_stats = {
        "__global__": {"median": 6.0, "q1": 3.0, "q3": 7.0},
        "__global_exclusive__": {
            "alpha": {"median": 7.0, "q1": 4.0, "q3": 9.0},
        },
        "alpha": {"__overall__": {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}},
    }

    def fake_prepare(project, timeline, *_, **__):
        labelled = _fake_labelled_timeline(
            project,
            timeline,
            folds=["fold-1", "fold-2"],
            train_ends=[
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-28", tz="UTC"),
            ],
        )
        labelled = labelled.drop(columns=["walkforward_fold"])
        return labelled, "label", 0.5, False

    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare)
    monkeypatch.setattr(strategies, "_compute_project_fold_statistics", lambda *_args, **_kwargs: fold_stats)

    schedule = strategies.strategy2_random_within_median_range(
        detection_df=detection_df,
        timelines=timelines,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
        random_seed=1,
        mode="cross_project",
    )

    assert not schedule.empty
    assert schedule["quartile_source"].unique().tolist() == ["global_lopo"]
    assert schedule["strategy_mode"].unique().tolist() == ["cross_project"]


def test_strategies_emit_fold_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    detection_df = pd.DataFrame(
        {
            "project": ["alpha", "alpha", "alpha", "alpha"],
            "commit_date": [
                "2024-01-05T00:00:00+00:00",
                "2024-01-25T00:00:00+00:00",
                "2024-02-10T00:00:00+00:00",
                "2024-02-20T00:00:00+00:00",
            ],
            "detection_time_days": [2.0, 4.0, 6.0, 8.0],
        }
    )
    timeline = pd.DataFrame(
        {
            "merge_date_ts": [
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-28", tz="UTC"),
            ],
            "day_index": [10, 11],
            "builds_per_day": [2.0, 2.0],
        }
    )
    assignments = pd.DataFrame(
        {
            "merge_date_ts": timeline["merge_date_ts"],
            "walkforward_fold": ["fold-1", "fold-2"],
            "train_window_start": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-02-01", tz="UTC"),
            ],
            "train_window_end": [
                pd.Timestamp("2024-01-31", tz="UTC"),
                pd.Timestamp("2024-02-29", tz="UTC"),
            ],
        }
    )

    def fake_get_metadata(project, _pred_root, **_kwargs):
        assert project == "alpha"
        return {
            "folds": {
                "fold-1": {
                    "train_start": assignments.loc[0, "train_window_start"],
                    "train_end": assignments.loc[0, "train_window_end"],
                },
                "fold-2": {
                    "train_start": assignments.loc[1, "train_window_start"],
                    "train_end": assignments.loc[1, "train_window_end"],
                },
            },
            "assignments": assignments,
        }

    def fake_prepare(project, project_timeline, *_args, walkforward_assignments=None, **_kwargs):
        assert project == "alpha"
        labelled = project_timeline.copy()
        labelled["_strategy_label"] = True
        if walkforward_assignments is not None:
            labelled = labelled.merge(walkforward_assignments, on="merge_date_ts", how="left")
        return labelled, "integration_label", 0.5, False

    monkeypatch.setattr(strategies, "_get_project_walkforward_metadata", fake_get_metadata)
    monkeypatch.setattr(strategies, "_prepare_labelled_timeline", fake_prepare)

    timeline_map = {"alpha": timeline}

    schedule1 = strategies.strategy1_median_schedule(
        detection_df=detection_df,
        timelines=timeline_map,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
    )
    assert set(schedule1["walkforward_fold"]) == {"fold-1", "fold-2"}

    fold1_row = schedule1.set_index("walkforward_fold").loc["fold-1"]
    assert fold1_row["median_detection_builds"] == pytest.approx(3.0)
    assert fold1_row["train_window_start"] == assignments.loc[0, "train_window_start"]
    assert fold1_row["train_window_end"] == assignments.loc[0, "train_window_end"]

    fold2_row = schedule1.set_index("walkforward_fold").loc["fold-2"]
    assert fold2_row["median_detection_builds"] == pytest.approx(7.0)
    assert fold2_row["train_window_start"] == assignments.loc[1, "train_window_start"]
    assert fold2_row["train_window_end"] == assignments.loc[1, "train_window_end"]

    schedule2 = strategies.strategy2_random_within_median_range(
        detection_df=detection_df,
        timelines=timeline_map,
        predictions_root="unused",
        risk_column="risk",
        label_column=None,
        threshold=0.5,
        random_seed=7,
    )
    assert set(schedule2["walkforward_fold"]) == {"fold-1", "fold-2"}

    fold1_q1 = 2.5
    fold1_q3 = 3.5
    fold1_result = schedule2.set_index("walkforward_fold").loc["fold-1"]
    assert fold1_result["offset_builds_q1"] == pytest.approx(fold1_q1)
    assert fold1_result["offset_builds_q3"] == pytest.approx(fold1_q3)
    assert fold1_result["train_window_end"] == assignments.loc[0, "train_window_end"]
    expected_sample_fold1 = _expected_uniform("alpha", "fold-1", 7, fold1_q1, fold1_q3)
    assert fold1_result["sampled_offset_builds"] == pytest.approx(expected_sample_fold1)

    fold2_q1 = 6.5
    fold2_q3 = 7.5
    fold2_result = schedule2.set_index("walkforward_fold").loc["fold-2"]
    assert fold2_result["offset_builds_q1"] == pytest.approx(fold2_q1)
    assert fold2_result["offset_builds_q3"] == pytest.approx(fold2_q3)
    assert fold2_result["train_window_start"] == assignments.loc[1, "train_window_start"]
    expected_sample_fold2 = _expected_uniform("alpha", "fold-2", 7, fold2_q1, fold2_q3)
    assert fold2_result["sampled_offset_builds"] == pytest.approx(expected_sample_fold2)


def test_strategy4_default_features_follow_prediction_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected_kamei = ["feat_alpha"]
    expected_vcc = ["feat_beta"]
    expected_cov = ["feat_gamma"]

    settings_ns = types.SimpleNamespace(
        KAMEI_FEATURES=expected_kamei,
        VCCFINDER_FEATURES=expected_vcc,
        PROJECT_TOTAL_PERCENT_FEATURES=expected_cov,
        PROJECT_COMMIT_PERCENT_FEATURES=[],
        PROJECT_CALCULATION_FEATURES=[],
        PROJECT_ALL_PERCENT_FEATURES=[],
    )
    monkeypatch.setattr(strategies, "prediction_settings", settings_ns, raising=False)

    captured: dict[str, list[str]] = {}

    train_start = pd.Timestamp("2023-12-01", tz="UTC")
    train_end = pd.Timestamp("2023-12-31", tz="UTC")
    val_start = pd.Timestamp("2024-01-01", tz="UTC")
    val_end = pd.Timestamp("2024-01-10", tz="UTC")

    def fake_build_regression_dataset(
        detection_df,
        timelines,
        build_counts_df,
        predictions_root,
        feature_cols,
        risk_column,
        label_column,
        threshold,
        **kwargs,
    ):
        features = list(feature_cols)
        captured["requested_features"] = features
        data: dict[str, list[object]] = {
            "project": ["proj"],
            "merge_date": [pd.Timestamp("2024-01-05", tz="UTC")],
            "observed_additional_builds": [1.0],
            "builds_per_day": [2.0],
            "label_flag": [True],
            "label_source": ["mock"],
            "walkforward_fold": ["fold-1"],
            "train_window_start": [train_start],
            "train_window_end": [train_end],
            "validation_window_start": [val_start],
            "validation_window_end": [val_end],
        }
        for feature in features:
            data[feature] = [1.0]
        return pd.DataFrame(data)

    def fake_train_linear_regression(dataset, feature_cols, target_column, **kwargs):
        coefficients = {col: 0.0 for col in feature_cols}
        model = strategies.RegressionModel(
            intercept=1.0,
            coefficients=coefficients,
            feature_order=list(feature_cols),
        )
        return model, {"train_mae": 0.0}

    timeline = pd.DataFrame(
        {
            "merge_date": ["2024-01-05"],
            "merge_date_ts": [pd.Timestamp("2024-01-05", tz="UTC")],
            "day_index": [10],
            "builds_per_day": [2.0],
        }
    )
    build_counts_df = pd.DataFrame({"project": ["proj"], "builds_per_day": [2.0]})

    monkeypatch.setattr(strategies, "_build_regression_dataset", fake_build_regression_dataset)
    monkeypatch.setattr(strategies, "_train_linear_regression", fake_train_linear_regression)
    monkeypatch.setattr(strategies, "_load_detection_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(strategies, "_load_build_timelines", lambda *_args, **_kwargs: {"proj": timeline})
    monkeypatch.setattr(strategies, "_load_build_counts", lambda *_args, **_kwargs: build_counts_df)
    def fake_align(timeline_df: pd.DataFrame, target_date: pd.Timestamp) -> strategies.Alignment:
        match = timeline_df.loc[timeline_df["merge_date_ts"] == target_date]
        day_index = int(match["day_index"].iloc[0]) if not match.empty else None
        return strategies.Alignment(
            merge_date=target_date,
            day_index=day_index,
            status="within_range",
        )

    monkeypatch.setattr(strategies, "_align_to_timeline", fake_align)

    diagnostics_file = tmp_path / "strategy4_diag.json"
    schedule, model, metrics = strategies.strategy4_cross_project_regression(
        diagnostics_output_path=str(diagnostics_file),
        evaluation_mode="random_project_split",
    )

    expected_features = expected_kamei + expected_vcc + expected_cov + ["builds_per_day", "label_flag"]
    expected_signature = hashlib.sha256("\n".join(expected_features).encode("utf-8")).hexdigest()
    assert captured["requested_features"] == expected_features
    assert list(model.feature_order) == expected_features
    assert schedule["scheduled_additional_builds"].tolist() == [1]
    row = schedule.iloc[0]
    assert row["predicted_additional_builds_raw"] == pytest.approx(1.0 + len(expected_features) * 0.0 + 0.0)
    assert row["rounding_mode_used"] == "ceil"
    assert not bool(row["prediction_was_clipped"])
    assert row["walkforward_fold"] == "fold-1"
    assert row["train_window_start"] == train_start
    assert row["validation_window_end"] == val_end
    assert metrics["train_mae"] == 0.0
    assert metrics["model_version"] == "linear_regression_v1"
    assert metrics["feature_signature"] == expected_signature
    assert row["model_version"] == "linear_regression_v1"
    assert row["feature_signature"] == expected_signature


def test_strategy4_reports_fold_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "project": ["proj", "proj"],
            "merge_date": [
                pd.Timestamp("2024-01-05", tz="UTC"),
                pd.Timestamp("2024-01-06", tz="UTC"),
            ],
            "observed_additional_builds": [2.0, 4.0],
            "builds_per_day": [2.0, 2.0],
            "label_flag": [True, True],
            "label_source": ["mock", "mock"],
            "walkforward_fold": ["fold-1", "fold-2"],
            "train_window_start": [
                pd.Timestamp("2023-12-01", tz="UTC"),
                pd.Timestamp("2023-12-15", tz="UTC"),
            ],
            "train_window_end": [
                pd.Timestamp("2023-12-31", tz="UTC"),
                pd.Timestamp("2023-12-31", tz="UTC"),
            ],
            "validation_window_start": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-06", tz="UTC"),
            ],
            "validation_window_end": [
                pd.Timestamp("2024-01-05", tz="UTC"),
                pd.Timestamp("2024-01-06", tz="UTC"),
            ],
            "feature_x": [1.5, 6.0],
        }
    )

    def fake_build_regression_dataset(*_args, **_kwargs):
        return dataset.copy()

    def fake_train_linear_regression(*_args, **_kwargs):
        model = strategies.RegressionModel(
            intercept=0.0,
            coefficients={"feature_x": 1.0},
            feature_order=["feature_x"],
        )
        return model, {"train_mae": 0.0}

    timeline = pd.DataFrame(
        {
            "merge_date": ["2024-01-05", "2024-01-06"],
            "merge_date_ts": [
                pd.Timestamp("2024-01-05", tz="UTC"),
                pd.Timestamp("2024-01-06", tz="UTC"),
            ],
            "day_index": [10, 11],
            "builds_per_day": [2.0, 2.0],
        }
    )
    build_counts_df = pd.DataFrame({"project": ["proj"], "builds_per_day": [2.0]})

    monkeypatch.setattr(strategies, "_build_regression_dataset", fake_build_regression_dataset)
    monkeypatch.setattr(strategies, "_train_linear_regression", fake_train_linear_regression)
    monkeypatch.setattr(strategies, "_load_detection_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(strategies, "_load_build_timelines", lambda *_args, **_kwargs: {"proj": timeline})
    monkeypatch.setattr(strategies, "_load_build_counts", lambda *_args, **_kwargs: build_counts_df)
    monkeypatch.setattr(
        strategies,
        "_align_to_timeline",
        lambda _timeline, target_date: strategies.Alignment(
            merge_date=target_date,
            day_index=10,
            status="within_range",
        ),
    )

    diagnostics_file = tmp_path / "diag.json"
    schedule, _model, metrics = strategies.strategy4_cross_project_regression(
        feature_cols=("feature_x",),
        diagnostics_output_path=str(diagnostics_file),
        evaluation_mode="random_project_split",
    )

    fold_metrics = metrics["fold_metrics"]
    overall = metrics["overall_performance"]

    assert pytest.approx(fold_metrics["fold-1"]["mae"], rel=1e-6) == 0.5
    assert pytest.approx(fold_metrics["fold-1"]["rmse"], rel=1e-6) == 0.5
    assert pytest.approx(fold_metrics["fold-1"]["mape"], rel=1e-6) == 0.25
    assert fold_metrics["fold-1"]["count"] == 1.0

    assert pytest.approx(fold_metrics["fold-2"]["mae"], rel=1e-6) == 2.0
    assert pytest.approx(fold_metrics["fold-2"]["rmse"], rel=1e-6) == 2.0
    assert pytest.approx(fold_metrics["fold-2"]["mape"], rel=1e-6) == 0.5
    assert fold_metrics["fold-2"]["count"] == 1.0

    assert pytest.approx(overall["mae"], rel=1e-6) == 1.25
    assert pytest.approx(overall["rmse"], rel=1e-6) == math.sqrt((0.25 + 4.0) / 2.0)
    assert pytest.approx(overall["mape"], rel=1e-6) == 0.375
    assert overall["count"] == 2.0
    assert schedule["walkforward_fold"].tolist() == ["fold-1", "fold-2"]
    assert schedule["model_version"].tolist() == ["linear_regression_v1", "linear_regression_v1"]

    payload = json.loads(diagnostics_file.read_text(encoding="utf-8"))
    assert payload["model"]["version"] == "linear_regression_v1"
    assert payload["metrics"]["overall_performance"]["mae"] == pytest.approx(1.25)
    assert payload["folds"]["fold-1"]["train_window_start"] == "2023-12-01T00:00:00+00:00"
    assert payload["folds"]["fold-2"]["validation_window_end"] == "2024-01-06T00:00:00+00:00"


def test_strategy4_leave_one_project_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    merge_dates = [
        pd.Timestamp("2024-01-03", tz="UTC"),
        pd.Timestamp("2024-01-04", tz="UTC"),
        pd.Timestamp("2024-01-05", tz="UTC"),
    ]
    dataset = pd.DataFrame(
        {
            "project": ["projA", "projB", "projC"],
            "merge_date": merge_dates,
            "observed_additional_builds": [2.0, 4.0, 6.0],
            "builds_per_day": [1.0, 2.0, 3.0],
            "label_flag": [True, True, True],
            "label_source": ["mock", "mock", "mock"],
            "walkforward_fold": ["fold-1", "fold-1", "fold-1"],
            "train_window_start": [pd.Timestamp("2023-12-01", tz="UTC")] * 3,
            "train_window_end": [pd.Timestamp("2023-12-31", tz="UTC")] * 3,
            "validation_window_start": merge_dates,
            "validation_window_end": merge_dates,
        }
    )

    def fake_build_regression_dataset(*_args, **_kwargs):
        return dataset.copy()

    timelines = {
        "projA": pd.DataFrame(
            {
                "merge_date": ["2024-01-03"],
                "merge_date_ts": [pd.Timestamp("2024-01-03", tz="UTC")],
                "day_index": [1],
                "builds_per_day": [1.0],
            }
        ),
        "projB": pd.DataFrame(
            {
                "merge_date": ["2024-01-04"],
                "merge_date_ts": [pd.Timestamp("2024-01-04", tz="UTC")],
                "day_index": [2],
                "builds_per_day": [2.0],
            }
        ),
        "projC": pd.DataFrame(
            {
                "merge_date": ["2024-01-05"],
                "merge_date_ts": [pd.Timestamp("2024-01-05", tz="UTC")],
                "day_index": [3],
                "builds_per_day": [3.0],
            }
        ),
    }
    build_counts_df = pd.DataFrame(
        {"project": ["projA", "projB", "projC"], "builds_per_day": [1.0, 2.0, 3.0]}
    )

    monkeypatch.setattr(strategies, "_build_regression_dataset", fake_build_regression_dataset)
    monkeypatch.setattr(strategies, "_load_detection_table", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(strategies, "_load_build_timelines", lambda *_args, **_kwargs: timelines)
    monkeypatch.setattr(strategies, "_load_build_counts", lambda *_args, **_kwargs: build_counts_df)

    def fake_align(timeline_df: pd.DataFrame, target_date: pd.Timestamp) -> strategies.Alignment:
        row = timeline_df.loc[timeline_df["merge_date_ts"] == target_date]
        day_index = int(row["day_index"].iloc[0]) if not row.empty else None
        return strategies.Alignment(merge_date=target_date, day_index=day_index, status="within_range")

    monkeypatch.setattr(strategies, "_align_to_timeline", fake_align)

    diagnostics_file = tmp_path / "lopo_diag.json"
    schedule, model, metrics = strategies.strategy4_cross_project_regression(
        feature_cols=("builds_per_day",),
        diagnostics_output_path=str(diagnostics_file),
        evaluation_mode="leave_one_project_out",
    )

    assert metrics["evaluation_mode"] == "leave_one_project_out"
    assert metrics["lopo_overall"]["mae"] == pytest.approx(0.0)
    assert metrics["lopo_project_metrics"]["projA"]["mae"] == pytest.approx(0.0)
    project_to_builds = dict(zip(schedule["project"], schedule["scheduled_additional_builds"]))
    assert project_to_builds == {"projA": 2, "projB": 4, "projC": 6}
    assert isinstance(model, strategies.RegressionModel)
    assert diagnostics_file.exists()


def test_strategy4_simple_mode_forces_line_change(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "project": ["proj"],
            "merge_date": [pd.Timestamp("2024-01-05", tz="UTC")],
            "observed_additional_builds": [2.0],
            "builds_per_day": [2.0],
            "label_flag": [True],
            "label_source": ["mock"],
            "walkforward_fold": ["fold-1"],
            "train_window_start": [pd.Timestamp("2023-12-01", tz="UTC")],
            "train_window_end": [pd.Timestamp("2023-12-31", tz="UTC")],
            "validation_window_start": [pd.Timestamp("2024-01-01", tz="UTC")],
            "validation_window_end": [pd.Timestamp("2024-01-05", tz="UTC")],
            "line_change_total": [10.0],
        }
    )
    captured: Dict[str, tuple[str, ...]] = {}

    def fake_build_regression_dataset(*_args, **kwargs):
        captured["required"] = tuple(kwargs.get("required_feature_columns", ()))
        return dataset.copy()

    timeline = pd.DataFrame(
        {
            "merge_date": ["2024-01-05"],
            "merge_date_ts": [pd.Timestamp("2024-01-05", tz="UTC")],
            "day_index": [10],
            "builds_per_day": [2.0],
        }
    )
    build_counts_df = pd.DataFrame({"project": ["proj"], "builds_per_day": [2.0]})

    def fake_align(_timeline: pd.DataFrame, target_date: pd.Timestamp) -> strategies.Alignment:
        return strategies.Alignment(merge_date=target_date, day_index=10, status="within_range")

    monkeypatch.setattr(strategies, "_build_regression_dataset", fake_build_regression_dataset)
    monkeypatch.setattr(strategies, "_load_build_counts", lambda *_args, **_kwargs: build_counts_df)
    monkeypatch.setattr(strategies, "_align_to_timeline", fake_align)

    schedule, model, metrics = strategies.strategy4_cross_project_regression(
        detection_df=pd.DataFrame(),
        timelines={"proj": timeline},
        predictions_root=str(tmp_path),
        mode="simple",
        evaluation_mode="random_project_split",
    )

    assert model.feature_order == ["line_change_total"]
    assert metrics["regression_mode"] == "simple"
    assert captured["required"] == ("line_change_total",)
    assert schedule["regression_mode"].unique().tolist() == ["simple"]
