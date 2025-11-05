"""Utilities for building cross-project training datasets."""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

import data_preparation


@dataclass
class ProjectDataset:
    name: str
    csv_path: str
    X: pd.DataFrame
    y: pd.Series
    feature_columns: List[str]


@dataclass
class CrossProjectTrainingSet:
    projects: List[str]
    X: pd.DataFrame
    y: pd.Series
    skipped_projects: Dict[str, str]


def _resolve_project_csv(project_name: str, base_dir: str) -> Optional[str]:
    pattern = os.path.join(base_dir, project_name, '*_daily_aggregated_metrics.csv')
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_project_dataset(project_name: str, base_dir: str) -> Optional[ProjectDataset]:
    csv_path = _resolve_project_csv(project_name, base_dir)
    if not csv_path:
        print(f"  [CROSS] プロジェクト '{project_name}' のCSVが見つかりませんでした ({base_dir}).")
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        print(f"  [CROSS] プロジェクト '{project_name}' の読み込みに失敗しました: {exc}")
        return None

    prepared = data_preparation.preprocess_dataframe_for_within_project(df, df)
    if not prepared:
        print(f"  [CROSS] プロジェクト '{project_name}' の前処理をスキップします。")
        return None

    X, y, _, feature_columns = prepared
    return ProjectDataset(project_name, csv_path, X, y, feature_columns)


def build_training_set(
    project_names: List[str],
    base_dir: str,
    align_to_columns: List[str],
) -> Optional[CrossProjectTrainingSet]:
    datasets: List[ProjectDataset] = []
    skipped: Dict[str, str] = {}

    for name in project_names:
        dataset = load_project_dataset(name, base_dir)
        if not dataset:
            skipped[name] = 'load_failed'
            continue
        datasets.append(dataset)

    if not datasets:
        print("  [CROSS] 有効な学習用プロジェクトがありません。")
        return None

    aligned_frames: List[pd.DataFrame] = []
    aligned_targets: List[pd.Series] = []

    for dataset in datasets:
        aligned = dataset.X
        if align_to_columns:
            aligned = aligned.reindex(columns=align_to_columns, fill_value=0)
        else:
            align_to_columns = aligned.columns.tolist()
        aligned_frames.append(aligned.reset_index(drop=True))
        aligned_targets.append(dataset.y.reset_index(drop=True))

    combined_X = pd.concat(aligned_frames, axis=0, ignore_index=True)
    combined_y = pd.concat(aligned_targets, axis=0, ignore_index=True)

    if combined_X.empty or combined_y.empty:
        print("  [CROSS] 結合後の学習データが空です。")
        return None

    combined_y = combined_y.astype(int)

    if combined_y.nunique() < 2:
        print("  [CROSS] 学習データに複数クラスが存在しません。")
        return None

    project_list = [dataset.name for dataset in datasets]
    return CrossProjectTrainingSet(project_list, combined_X, combined_y, skipped)
