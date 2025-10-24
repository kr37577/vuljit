#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(2) Risk Prediction — RQ3の前処理として、
"RQ1の最良モデルを訓練期間(前半50%)で再学習し、テスト期間(後半50%)の各日 i に確率 F_i を付与"
を実装するスタンドアロン・スクリプト。

想定入出力:
- 入力: settings.BASE_DATA_DIRECTORY 配下の各プロジェクト
         <project>/<project>_daily_aggregated_metrics.csv
- 出力: settings.RESULTS_BASE_DIRECTORY/<project>/<project>_daily_aggregated_metrics_with_predictions.csv
         (列 "predicted_risk_VCCFinder_ProjTotal" を必ず含む)

備考:
- RQ3スクリプト (rq3_test_now.py など) が既定で参照する列名 DEFAULT_RISK_COL
  = "predicted_risk_VCCFinder_ProjTotal" と合致させる。
- 訓練/テストの分割は、日付順に並べた上で前半50%を訓練、後半50%をテスト。
- settings.SELECTED_MODEL と features 定義（KAMEI/VCCFINDER/PROJECT_TOTAL_PERCENT_FEATURES）を再利用。
- settings.USE_HYPERPARAM_OPTIMIZATION が True の場合のみ RandomizedSearchCV を実行。

依存:
- settings.py, data_preparation.py, model_definition.py が同一ディレクトリに存在すること。
"""

from __future__ import annotations
import os
import glob
import argparse
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# 外部依存（導入済み前提）
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# プロジェクト内モジュール
import settings
import data_preparation

# ========= ユーティリティ =========

def _pick_best_feature_set(df_columns: List[str]) -> Tuple[str, List[str]]:
    """RQ1の最良構成: VCCFinder + Project Total を既定で選択。
    対応する列名: settings.VCCFINDER_FEATURES と settings.PROJECT_TOTAL_PERCENT_FEATURES
    存在するカラムのみを使う。
    戻り値: (説明ラベル, 使用特徴量一覧)
    """
    vcc = [c for c in settings.VCCFINDER_FEATURES if c in df_columns]
    pjt = [c for c in settings.PROJECT_TOTAL_PERCENT_FEATURES if c in df_columns]
    feats = sorted(set(vcc + pjt))
    return ("VCCFinder + Project Total", feats)


def _downsample_majority(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """少数クラス数に合わせて多数クラスをランダムダウンサンプリング。
    imblearn 未使用の軽量実装。
    """
    y = y.astype(int)
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index
    n_pos = len(pos_idx)
    if n_pos == 0:
        return X, y
    if len(neg_idx) <= n_pos:
        return X, y
    rng = np.random.default_rng(random_state)
    keep_neg = rng.choice(neg_idx, size=n_pos, replace=False)
    keep_idx = np.concatenate([pos_idx.values, keep_neg])
    Xb = X.loc[keep_idx]
    yb = y.loc[keep_idx]
    return Xb, yb


def _build_model(random_state: int = 42) -> object:
    mdl_name = settings.SELECTED_MODEL.lower()
    if mdl_name == 'xgboost':
        if not _HAS_XGB:
            raise RuntimeError("XGBoost が利用できません。settings.SELECTED_MODEL を 'random_forest' に変更してください。")
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist',
        )
    # default: RF
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        random_state=random_state,
        n_jobs=-1,
    )


def _hyperparam_space(model: object) -> dict:
    if isinstance(model, RandomForestClassifier):
        return {
            # 'n_estimators': [200, 300, 400, 600, 800],
            # 'max_depth': [None, 6, 10, 16, 24],
            # 'min_samples_split': [2, 4, 8],
            # 'min_samples_leaf': [1, 2, 4],
            # 'max_features': ['sqrt', 'log2', None],
            "classifier__n_estimators": [200, 500, 1000, 1500],
            "classifier__max_depth": [None, 5, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10, 20, 50],
            "classifier__min_samples_leaf": [1, 2, 5, 10, 20],
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.8, 1.0],
            "classifier__class_weight": [None, "balanced"]
        }
    if _HAS_XGB and isinstance(model, XGBClassifier):
        return {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.03, 0.05, 0.1],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0],
            'reg_alpha': [0.0, 0.1, 0.5],
        }
    return {}


def train_on_train_predict_test(
    df_daily: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = 'is_vcc',
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Index, pd.Index]:
    """日付順に 50/50 で分割し、訓練(前半)→再学習→テスト(後半)に確率を付与。
    戻り値: (pred_proba_series, train_index, test_index)
    """
    if 'merge_date' not in df_daily.columns:
        raise ValueError("'merge_date' 列が必要です。")

    # 安全にソート
    d = df_daily.copy()
    d['merge_date'] = pd.to_datetime(d['merge_date']).dt.date
    d.sort_values('merge_date', inplace=True)

    # 特徴量の存在確認
    feats = [c for c in feature_cols if c in d.columns]
    if not feats:
        raise ValueError("使用可能な特徴量が見つかりませんでした。")

    # ラベル作成（存在しなければ vcc_commit_count を日単位ラベルとみなす）
    if label_col not in d.columns:
        if 'vcc_commit_count' in d.columns:
            d[label_col] = (d['vcc_commit_count'].fillna(0).astype(float) > 0).astype(int)
        else:
            raise ValueError("ラベル列が存在せず、vcc_commit_count も見つかりません。")

    # 50/50 split by index（日数基準の単純分割）
    n = len(d)
    split = max(1, int(n * 0.5))
    train_idx = d.index[:split]
    test_idx  = d.index[split:]
    if len(test_idx) == 0:
        raise ValueError("テスト期間が空です。データ日数を確認してください。")

    # 欠損処理（木系モデル想定なので単純補完）
    X = d[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    y = d[label_col].astype(int)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test = X.loc[test_idx]

    # 少数クラスチェック
    if y_train.sum() < settings.MIN_SAMPLES_THRESHOLD:
        raise ValueError("訓練期間の陽性サンプルがしきい値未満のため学習不能です。")
    if y_train.nunique() < 2:
        raise ValueError("訓練期間でターゲットが単一クラスです。")

    # ダウンサンプリング（random_under を踏襲）
    if settings.SAMPLING_METHOD == 'random_under':
        Xb, yb = _downsample_majority(X_train, y_train, random_state=random_state)
    else:
        Xb, yb = X_train, y_train  # SMOTE 等は未サポート（簡潔化）

    # モデル構築
    base_model = _build_model(random_state=random_state)

    if getattr(settings, 'USE_HYPERPARAM_OPTIMIZATION', False):
        param_dist = _hyperparam_space(base_model)
        if param_dist:
            rs = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=settings.RANDOM_SEARCH_N_ITER,
                scoring='MCC',  # F1マクロを最適化
                cv=settings.RANDOM_SEARCH_CV,
                n_jobs=-1,
                verbose=0,
                random_state=random_state,
            )
            rs.fit(Xb, yb)
            model = rs.best_estimator_
        else:
            model = base_model.fit(Xb, yb)
    else:
        model = base_model.fit(Xb, yb)

    # 予測確率（陽性クラス）
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function をシグモイドで確率化
        if hasattr(model, 'decision_function'):
            z = model.decision_function(X_test)
            proba = 1.0 / (1.0 + np.exp(-z))
        else:
            # RF/XGB 以外は未想定
            raise RuntimeError('確率を返せないモデルです。')

    s = pd.Series(proba, index=test_idx, name='predicted_risk_VCCFinder_ProjTotal')
    return s, train_idx, test_idx


# ========= メイン =========

def main():
    ap = argparse.ArgumentParser(description='(2) Risk Prediction — 再学習してテスト期間に ˆF_i を付与')
    ap.add_argument('-p', '--project', type=str, help='単一プロジェクト名（未指定なら全件）')
    args = ap.parse_args()

    # 入力探索
    if args.project:
        pattern = os.path.join(settings.BASE_DATA_DIRECTORY, args.project, '*_daily_aggregated_metrics.csv')
    else:
        pattern = os.path.join(settings.BASE_DATA_DIRECTORY, '*', '*_daily_aggregated_metrics.csv')

    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        print(f"入力CSVが見つかりません: {pattern}")
        return

    out_root = settings.RESULTS_BASE_DIRECTORY
    os.makedirs(out_root, exist_ok=True)

    for idx, csv_path in enumerate(csv_files, 1):
        project = os.path.basename(os.path.dirname(csv_path))
        print(f"[{idx}/{len(csv_files)}] Project={project} : {csv_path}")
        proj_out_dir = os.path.join(out_root, project)
        os.makedirs(proj_out_dir, exist_ok=True)

        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  × 読み込み失敗: {e}")
            continue

        # Within-Project の前処理（既存関数を再利用）
        prepped = data_preparation.preprocess_dataframe_for_within_project(df, df)
        if not prepped:
            print("  × 前処理結果が空")
            continue
        X_project_full, y_project, _, feature_columns_full = prepped

        # 使用特徴量（VCCFinder + Project Total）
        label, feats = _pick_best_feature_set(feature_columns_full)
        if not feats:
            print("  × 使用可能な特徴量がありません")
            continue

        # 学習→テスト期間に予測確率付与
        try:
            proba_s, train_idx, test_idx = train_on_train_predict_test(
                df_daily=df.loc[X_project_full.index],
                feature_cols=feats,
                label_col='is_vcc_day' if 'is_vcc_day' in df.columns else 'is_vcc_day',
                random_state=settings.RANDOM_STATE,
            )
        except Exception as e:
            print(f"  × 学習/推論に失敗: {e}")
            continue

        # 出力: 元DFのモデル対象行だけに、テスト期間のみに列を追加
        out_df = df.loc[X_project_full.index].copy()
        col_name = 'predicted_risk_VCCFinder_ProjTotal'
        out_df[col_name] = np.nan
        out_df.loc[proba_s.index, col_name] = proba_s.values

        out_csv = os.path.join(proj_out_dir, f"{project}_daily_aggregated_metrics_with_predictions.csv")
        try:
            out_df.to_csv(out_csv, index=False)
            print(f"  ✓ 出力: {out_csv}")
        except Exception as e:
            print(f"  × 出力失敗: {e}")
            continue

        # 監査用メタを出力
        meta = {
            'project': project,
            'feature_set': label,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'selected_model': settings.SELECTED_MODEL,
            'used_hyperparam_optimization': bool(getattr(settings, 'USE_HYPERPARAM_OPTIMIZATION', False)),
            'sampling': settings.SAMPLING_METHOD,
            'random_state': settings.RANDOM_STATE,
            'output_column': col_name,
        }
        with open(os.path.join(proj_out_dir, 'risk_prediction_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
