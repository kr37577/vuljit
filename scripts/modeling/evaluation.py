# evaluation.py

import pandas as pd
import numpy as np
import warnings
import math
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, matthews_corrcoef,
    precision_recall_curve, auc,
    average_precision_score,
)
from typing import Tuple, Optional, List, Dict, Any
import os
import time
import tempfile
import joblib
from pathlib import Path
import json
import settings
import model_definition



def evaluate_model_performance(y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """モデルの性能を計算する。陽性が0件のFoldでは precision/recall/MCC を未定義(None)として扱う。"""
    expected_labels = [0, 1]

    report_dict = classification_report(
        y_true, y_pred, zero_division=0, output_dict=True, labels=expected_labels,
        target_names=[f"class_{l}" for l in expected_labels]
    )

    # 陽性数・予測陽性数を取得
    y_true_s = pd.Series(y_true).astype(int)
    y_pred_s = pd.Series(y_pred).astype(int)
    support_pos = int((y_true_s == 1).sum())
    pred_pos = int((y_pred_s == 1).sum())

    # 陽性が0件のFoldでは precision/recall/MCC を未定義(None)へ正規化
    # ただし precision は「予測陽性が1件以上」の場合のみ 0 とする
    try:
        c1 = report_dict.get('class_1', {})
        if support_pos == 0:
            # recall: 未定義
            if 'recall' in c1:
                c1['recall'] = None
            # precision: 予測陽性があれば0、なければ未定義
            if 'precision' in c1:
                c1['precision'] = 0.0 if pred_pos >= 1 else None
            # f1-score: precision/recall が未定義であれば未定義
            if 'f1-score' in c1:
                prec_v = c1.get('precision')
                rec_v = c1.get('recall')
                c1['f1-score'] = None if (prec_v is None or rec_v is None) else c1['f1-score']
            report_dict['class_1'] = c1
    except Exception:
        pass

    # MCC は陽性0件のFoldでは未定義(None)
    mcc_val = matthews_corrcoef(y_true, y_pred) if support_pos > 0 else None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": mcc_val,
        "classification_report_dict": report_dict,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=expected_labels).tolist()
    }
    # 追加: fold規模の情報（サニティチェック用）
    metrics["n_test"] = int(len(y_true))
    metrics["n_pos"] = int((pd.Series(y_true).astype(int) == 1).sum())
    metrics["pos_rate"] = (metrics["n_pos"] / metrics["n_test"]) if metrics["n_test"] > 0 else None

    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
        # AP（Average Precision）を正式なPR-AUCとして記録
        ap = average_precision_score(y_true, y_pred_proba)
        metrics["auc_pr"] = ap
        metrics["average_precision"] = ap
        # 参考: 線形補間の台形公式によるPR曲線面積（診断用に保持、集計では未使用）
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics["pr_auc_trapezoid"] = auc(recall, precision)
        except Exception:
            metrics["pr_auc_trapezoid"] = None
    else:
        metrics["auc_roc"] = None
        metrics["auc_pr"] = None
        metrics["average_precision"] = None
        metrics["pr_auc_trapezoid"] = None
    return metrics


def train_and_evaluate_fold(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, project_name: str, run_random_state: int
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[np.ndarray]]:
    """1 Fold分のモデル学習と評価を行う（内部CVなし、決定的パラメータで1回fit）"""
    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        return None, None, None

    if y_train.nunique() < 2:
        print("      警告: このFoldの学習データには1クラスしか含まれていないため、学習と評価をスキップします。")
        return None, None, None

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale = float(n_neg / max(1, n_pos))
    pipeline = model_definition.get_pipeline(run_random_state, scale_pos_weight=scale)
    param_dist = model_definition.get_param_distribution()
    
    # ハイパーパラ最適化を行うかどうかで分岐
    search = None
    best_model = None
    if getattr(settings, 'USE_HYPERPARAM_OPTIMIZATION', False) and param_dist:
        cv_splitter = TimeSeriesSplit(n_splits=settings.RANDOM_SEARCH_CV)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=settings.RANDOM_SEARCH_N_ITER,
            scoring=settings.SCORING,
            refit='PR_AUC',
            cv=cv_splitter,  # 内部CV分割数
            random_state=run_random_state,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        # 直接 pipeline を学習させる（ハイパーパラ探索なし）
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # === モデルとハイパーパラ探索結果の保存（atomic） ===
    try:
        timestamp = int(time.time())
        run_tag = f"run_{run_random_state}_{timestamp}"

        # 保存先ディレクトリを settings から取得（存在しない場合は作成）
        model_dir = getattr(settings, 'MODEL_OUTPUT_DIRECTORY', None)
        logs_dir = getattr(settings, 'LOGS_DIRECTORY', None)
        # プロジェクト名をサブディレクトリとして使う（簡易サニタイズ）
        safe_project = str(project_name).replace(os.sep, '_').replace(' ', '_')
        if model_dir:
            model_dir = os.path.join(model_dir, safe_project)
        if logs_dir:
            logs_dir = os.path.join(logs_dir, safe_project)

        if getattr(settings, 'SAVE_HYPERPARAM_RESULTS', False) and logs_dir and search is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            target = os.path.join(logs_dir, f"cv_results_{run_tag}.pkl")
            fd, tmp_path = tempfile.mkstemp(dir=logs_dir, prefix=f"cvtmp_{run_tag}_", suffix='.pkl')
            os.close(fd)
            joblib.dump(search.cv_results_, tmp_path)
            os.replace(tmp_path, target)
            # Also save JSON metadata (best_params, best_score) atomically
            try:
                meta = {
                    'run_tag': run_tag,
                    'timestamp': timestamp,
                    'best_params': getattr(search, 'best_params_', None),
                    'best_score': getattr(search, 'best_score_', None),
                    'refit': getattr(search, 'refit', None),
                }
                # Ensure JSON serializable by converting numpy types and non-serializable objects
                def _to_serializable(o):
                    if o is None or isinstance(o, (str, bool, int, float)):
                        return o
                    if isinstance(o, np.generic):
                        return o.item()
                    if isinstance(o, dict):
                        return {str(k): _to_serializable(v) for k, v in o.items()}
                    if isinstance(o, (list, tuple)):
                        return [_to_serializable(v) for v in o]
                    try:
                        return str(o)
                    except Exception:
                        return repr(o)

                serializable_meta = _to_serializable(meta)
                fd2, tmp_json = tempfile.mkstemp(dir=logs_dir, prefix=f"metatmp_{run_tag}_", suffix='.json')
                os.close(fd2)
                with open(tmp_json, 'w') as jf:
                    json.dump(serializable_meta, jf, indent=2)
                os.replace(tmp_json, os.path.join(logs_dir, f"metadata_{run_tag}.json"))
            except Exception as e:
                print(f"      Warning: failed to save metadata JSON: {e}")

        if getattr(settings, 'SAVE_BEST_MODEL', False) and model_dir and best_model is not None:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            target = os.path.join(model_dir, f"best_model_{run_tag}.joblib")
            fd, tmp_path = tempfile.mkstemp(dir=model_dir, prefix=f"modeltmp_{run_tag}_", suffix='.joblib')
            os.close(fd)
            joblib.dump(best_model, tmp_path)
            os.replace(tmp_path, target)
            # If logs_dir available, save a small metadata JSON for the saved model (atomic)
            try:
                if logs_dir:
                    Path(logs_dir).mkdir(parents=True, exist_ok=True)
                    model_meta = {
                        'run_tag': run_tag,
                        'timestamp': timestamp,
                        'model_class': type(best_model).__name__,
                        'note': 'best_model_saved'
                    }
                    # try to include model params if available, but fall back to stringified repr
                    try:
                        params = best_model.get_params()
                        model_meta['model_params'] = _to_serializable(params)
                    except Exception:
                        model_meta['model_params'] = str(getattr(best_model, '__class__', best_model))

                    fd3, tmp_json2 = tempfile.mkstemp(dir=logs_dir, prefix=f"modelmetatmp_{run_tag}_", suffix='.json')
                    os.close(fd3)
                    with open(tmp_json2, 'w') as jf2:
                        json.dump(model_meta, jf2, indent=2)
                    os.replace(tmp_json2, os.path.join(logs_dir, f"model_metadata_{run_tag}.json"))
            except Exception as e:
                print(f"      Warning: failed to save model metadata JSON: {e}")
    except Exception as e:
        # 保存失敗は致命的ではないがログとして出す
        print(f"警告: モデル/ハイパーパラ結果の保存に失敗しました: {e}")

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    fold_metrics = evaluate_model_performance(y_test, y_pred, y_pred_proba)
        
    # # 内部CVの完全撤廃: 直接fit
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     pipeline.fit(X_train, y_train)

    # y_pred = pipeline.predict(X_test)
    # y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    # fold_metrics = evaluate_model_performance(y_test, y_pred, y_pred_proba)

    # 特徴量重要度の抽出
    final_classifier = best_model.steps[-1][1]
    importances_df = None
    if hasattr(final_classifier, 'feature_importances_'):
        importances_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_classifier.feature_importances_
        })

    return fold_metrics, importances_df, y_pred_proba


def _run_stratified_k_fold(
    X_project: pd.DataFrame, y_project: pd.Series, project_name: str, run_random_state: int
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    """【従来手法】Stratified K-Fold による交差検証を実行"""
    skf = StratifiedKFold(n_splits=settings.N_SPLITS_K, shuffle=True, random_state=run_random_state)
    fold_metrics_list, fold_importances_list = [], []
    
        # ▼▼▼【修正箇所】アウトオブサンプル予測を格納するSeriesを初期化 ▼▼▼
    out_of_sample_predictions = pd.Series(index=X_project.index, dtype=float, name="predicted_probability")


    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X_project, y_project)):
        print(f"    --- Stratified K-Fold {fold_num + 1}/{settings.N_SPLITS_K} ---")
        X_train, X_test = X_project.iloc[train_idx], X_project.iloc[test_idx]
        y_train, y_test = y_project.iloc[train_idx], y_project.iloc[test_idx]
        
                # ▼▼▼【修正箇所】y_pred_proba を受け取る ▼▼▼
        metrics, importances_df, y_pred_proba = train_and_evaluate_fold(X_train, y_train, X_test, y_test, project_name, run_random_state)

        
        if metrics:
            # ▼▼▼【修正箇所】Fold番号を追加 ▼▼▼
            metrics['fold'] = fold_num
            fold_metrics_list.append(metrics)
        if importances_df is not None: 
            fold_importances_list.append(importances_df)
        if y_pred_proba is not None:
            # test_idxは位置ベースなので、ilocを使って元のインデックスを取得し、値を設定
            test_indices = X_project.iloc[test_idx].index
            out_of_sample_predictions.loc[test_indices] = y_pred_proba


    return fold_metrics_list, fold_importances_list, out_of_sample_predictions

def _run_time_series_validation(
    X_project: pd.DataFrame, y_project: pd.Series, project_name: str, run_random_state: int
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    """【新手法】時系列を考慮した交差検証を実行"""
    n_splits = settings.N_SPLITS_TIMESERIES
    if len(y_project) < n_splits * 2: # 少なくとも学習とテストに1チャンクずつ必要
        print(f"  プロジェクト: サンプル数({len(y_project)})が時系列分割数({n_splits})に対して小さすぎるためスキップします。")
        return [], [], pd.Series(index=X_project.index, dtype=float, name="predicted_probability")

    fold_metrics_list, fold_importances_list = [], []
    out_of_sample_predictions = pd.Series(index=X_project.index, dtype=float, name="predicted_probability")

    # データをn_splits個のチャンクに分割
    chunk_size = math.ceil(len(X_project) / n_splits)
    indices = list(range(len(X_project)))
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    # 逐次的な学習とテスト
    for i in range(1, len(chunks)):
        print(f"    --- Time-series Fold {i}/{len(chunks) - 1} ---")
        print(f"      (Testing on chunk {i})")

        if settings.USE_ONLY_RECENT_FOR_TRAINING:
            # スライディングウィンドウ方式: 直前のチャンクのみで学習
            train_idx = chunks[i-1]
            print(f"      Training on recent chunk {i-1} ({len(train_idx)} samples)")
        else:
            # 累積学習方式: これまでの全チャンクで学習
            train_idx = [idx for sublist in chunks[:i] for idx in sublist]
            print(f"      Training on cumulative chunks 0 to {i-1} ({len(train_idx)} samples)")

        test_idx = chunks[i]
        print(f"      Testing on chunk {i} ({len(test_idx)} samples)")


        X_train, X_test = X_project.iloc[train_idx], X_project.iloc[test_idx]
        y_train, y_test = y_project.iloc[train_idx], y_project.iloc[test_idx]

        # テストセットが1クラスでも学習・予測は実施する（AUC系は後段でNoneにする）
        if y_test.nunique() < 2:
            print(f"      注意: テストセット (Chunk {i}) が単一クラスです。AUC系は算出不可になりますが、予測は生成します。")

        metrics, importances_df, y_pred_proba = train_and_evaluate_fold(
            X_train, y_train, X_test, y_test, project_name, run_random_state
        )

        if metrics:
            # ▼▼▼【修正箇所】Fold番号を追加 ▼▼▼
            metrics['fold'] = i
            fold_metrics_list.append(metrics)
        if importances_df is not None:
            fold_importances_list.append(importances_df)
        if y_pred_proba is not None:
            # test_idxは位置ベースなので、ilocを使って元のインデックスを取得し、値を設定
            test_indices = X_project.iloc[test_idx].index
            out_of_sample_predictions.loc[test_indices] = y_pred_proba

    return fold_metrics_list, fold_importances_list, out_of_sample_predictions


def run_cross_validation_for_project(
    X_project: pd.DataFrame, y_project: pd.Series, project_name: str, run_random_state: int
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    """
    単一プロジェクトに対して設定された評価手法を実行するラッパー関数。
    """
    if y_project.nunique() < 2:
        print(f"  プロジェクト '{project_name}': データに1クラスしか存在しないためCVをスキップします。")
        return [], [], pd.Series(index=X_project.index, dtype=float, name="predicted_probability")

    if settings.EVALUATION_METHOD == 'time_series':
        print(f"--- プロジェクト '{project_name}' で時系列評価を開始 ---")
        return _run_time_series_validation(X_project, y_project, project_name, run_random_state)
    elif settings.EVALUATION_METHOD == 'stratified_k_fold':
        print(f"--- プロジェクト '{project_name}' で Stratified K-Fold 評価を開始 ---")
        if len(y_project) < settings.N_SPLITS_K:
             print(f"  プロジェクト '{project_name}': サンプル数がCVの分割数({settings.N_SPLITS_K})未満です。スキップします。")
             return [], [], pd.Series(index=X_project.index, dtype=float, name="predicted_probability")
        return _run_stratified_k_fold(X_project, y_project, project_name, run_random_state)
    else:
        raise ValueError(f"不明な評価手法が選択されています: {settings.EVALUATION_METHOD}")


def _run_cross_project_stratified(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    project_name: str,
    run_random_state: int,
    X_external: pd.DataFrame,
    y_external: pd.Series,
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    skf = StratifiedKFold(n_splits=settings.N_SPLITS_K, shuffle=True, random_state=run_random_state)
    fold_metrics_list: List[Dict] = []
    fold_importances_list: List[pd.DataFrame] = []
    out_of_sample_predictions = pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    for fold_num, (_, test_idx) in enumerate(skf.split(X_target, y_target)):
        print(f"    --- Cross-Project Stratified Fold {fold_num + 1}/{settings.N_SPLITS_K} ---")
        X_test = X_target.iloc[test_idx]
        y_test = y_target.iloc[test_idx]

        if y_test.nunique() < 2:
            print("      注意: テストセットが単一クラスです。AUC系は算出不可になりますが、予測は生成します。")

        metrics, importances_df, y_pred_proba = train_and_evaluate_fold(
            X_external, y_external, X_test, y_test, project_name, run_random_state
        )

        if metrics:
            metrics['fold'] = fold_num
            fold_metrics_list.append(metrics)
        if importances_df is not None:
            fold_importances_list.append(importances_df)
        if y_pred_proba is not None:
            test_indices = X_target.iloc[test_idx].index
            out_of_sample_predictions.loc[test_indices] = y_pred_proba

    return fold_metrics_list, fold_importances_list, out_of_sample_predictions


def _run_cross_project_time_series(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    project_name: str,
    run_random_state: int,
    X_external: pd.DataFrame,
    y_external: pd.Series,
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    n_splits = settings.N_SPLITS_TIMESERIES
    if len(y_target) < n_splits * 2:
        print(f"  プロジェクト: サンプル数({len(y_target)})が時系列分割数({n_splits})に対して小さすぎるためスキップします。")
        return [], [], pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    fold_metrics_list: List[Dict] = []
    fold_importances_list: List[pd.DataFrame] = []
    out_of_sample_predictions = pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    chunk_size = math.ceil(len(X_target) / n_splits)
    indices = list(range(len(X_target)))
    chunks = [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

    if settings.USE_ONLY_RECENT_FOR_TRAINING:
        print("  [CROSS] USE_ONLY_RECENT_FOR_TRAINING はクロスプロジェクト評価では無視されます。")

    for i in range(1, len(chunks)):
        print(f"    --- Cross-Project Time-series Fold {i}/{len(chunks) - 1} ---")
        test_idx = chunks[i]
        print(f"      Testing on chunk {i} ({len(test_idx)} samples)")

        X_test = X_target.iloc[test_idx]
        y_test = y_target.iloc[test_idx]
        if y_test.nunique() < 2:
            print("      注意: テストセットが単一クラスです。AUC系は算出不可になりますが、予測は生成します。")

        metrics, importances_df, y_pred_proba = train_and_evaluate_fold(
            X_external, y_external, X_test, y_test, project_name, run_random_state
        )

        if metrics:
            metrics['fold'] = i
            fold_metrics_list.append(metrics)
        if importances_df is not None:
            fold_importances_list.append(importances_df)
        if y_pred_proba is not None:
            test_indices = X_target.iloc[test_idx].index
            out_of_sample_predictions.loc[test_indices] = y_pred_proba

    return fold_metrics_list, fold_importances_list, out_of_sample_predictions


def _run_cross_project_full_holdout(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    project_name: str,
    run_random_state: int,
    X_external: pd.DataFrame,
    y_external: pd.Series,
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    fold_metrics_list: List[Dict] = []
    fold_importances_list: List[pd.DataFrame] = []
    out_of_sample_predictions = pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    metrics, importances_df, y_pred_proba = train_and_evaluate_fold(
        X_external, y_external, X_target, y_target, project_name, run_random_state
    )

    if metrics:
        metrics['fold'] = 0
        fold_metrics_list.append(metrics)
    if importances_df is not None:
        fold_importances_list.append(importances_df)
    if y_pred_proba is not None:
        out_of_sample_predictions.loc[X_target.index] = y_pred_proba

    return fold_metrics_list, fold_importances_list, out_of_sample_predictions


def run_cross_project_validation(
    X_target: pd.DataFrame,
    y_target: pd.Series,
    project_name: str,
    run_random_state: int,
    X_external: pd.DataFrame,
    y_external: pd.Series,
    eval_mode: str = 'fold',
) -> Tuple[List[Dict], List[pd.DataFrame], pd.Series]:
    """クロスプロジェクト設定でモデルを評価する。"""
    if X_external is None or y_external is None:
        print("  [CROSS] 外部学習データが空のためスキップします。")
        return [], [], pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    if X_external.empty or y_external.empty:
        print("  [CROSS] 外部学習データが空のためスキップします。")
        return [], [], pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    if y_external.nunique() < 2:
        print("  [CROSS] 外部学習データに複数のクラスが存在しないためスキップします。")
        return [], [], pd.Series(index=X_target.index, dtype=float, name="predicted_probability")

    eval_mode = (eval_mode or 'fold').lower()
    if eval_mode == 'full':
        print(f"--- プロジェクト '{project_name}' でクロスプロジェクト単一ホールドアウト評価を開始 ---")
        return _run_cross_project_full_holdout(X_target, y_target, project_name, run_random_state, X_external, y_external)

    if settings.EVALUATION_METHOD == 'time_series':
        print(f"--- プロジェクト '{project_name}' でクロスプロジェクト時系列評価を開始 ---")
        return _run_cross_project_time_series(X_target, y_target, project_name, run_random_state, X_external, y_external)
    elif settings.EVALUATION_METHOD == 'stratified_k_fold':
        if len(y_target) < settings.N_SPLITS_K:
            print(f"  プロジェクト '{project_name}': サンプル数がCVの分割数({settings.N_SPLITS_K})未満です。スキップします。")
            return [], [], pd.Series(index=X_target.index, dtype=float, name="predicted_probability")
        print(f"--- プロジェクト '{project_name}' でクロスプロジェクト Stratified K-Fold 評価を開始 ---")
        return _run_cross_project_stratified(X_target, y_target, project_name, run_random_state, X_external, y_external)
    else:
        raise ValueError(f"不明な評価手法が選択されています: {settings.EVALUATION_METHOD}")
