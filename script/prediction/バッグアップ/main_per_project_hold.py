# main.py
import os
import pandas as pd
import glob
import json
from typing import List, Dict, Any, Optional, Tuple
import argparse

# code/ ディレクトリ内の各モジュールをインポート
import settings
import data_preparation
import evaluation
import reporting


def time_series_train_test_split(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    train_ratio: float = 0.7,
    order_by: Optional[str] = None,
    ensure_datetime: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series], pd.Index, pd.Index]:
    """
    単純な時系列ホールドアウト分割を行うユーティリティ。
    - 既に X が時系列順（昇順）に並んでいる前提で先頭 train_ratio を学習、残りをテストに分割
    - order_by を指定すると、その列で安全に並び替えを行ってから分割（X 側の列を参照）

    Returns:
        X_train, X_test, y_train, y_test, train_index, test_index
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X は DataFrame を期待します。")
    if y is not None and not isinstance(y, pd.Series):
        raise ValueError("y は Series を期待します。")

    df_idx = X.index

    # 必要に応じて並び替え（X 側の列を使用）
    if order_by is not None:
        if order_by not in X.columns:
            raise ValueError(f"order_by='{order_by}' が X のカラムに存在しません。")
        order_col = X[order_by]
        if ensure_datetime:
            try:
                order_col = pd.to_datetime(order_col, errors='coerce', utc=True)
            except Exception:
                pass
        # 安全な安定ソート
        sorted_idx = order_col.sort_values(kind='mergesort').index
        X = X.loc[sorted_idx]
        if y is not None:
            y = y.loc[sorted_idx]
        df_idx = sorted_idx

    n = len(df_idx)
    if n < 2:
        raise ValueError("分割できる十分なサンプルがありません。")

    # 比率をクリップし、少なくとも1件ずつ確保
    r = float(train_ratio)
    r = max(0.05, min(0.95, r))
    split_point = max(1, min(n - 1, int(n * r)))

    train_index = df_idx[:split_point]
    test_index = df_idx[split_point:]

    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index] if y is not None else None
    y_test = y.loc[test_index] if y is not None else None

    return X_train, X_test, y_train, y_test, train_index, test_index

def get_text_metric_features(df: pd.DataFrame) -> list[str]:
    """
    DataFrameからテキストメトリクス関連の特徴量カラムを自動検出します。
    テキストメトリクスは 'text_'で始まり、'_file' または '_patch'で終わるカラム名と定義します。

    Args:
        df (pd.DataFrame): 特徴量を含むDataFrame。

    Returns:
        list[str]: テキストメトリクス関連のカラム名のリスト。
    """
    text_metric_columns = [
        col for col in df.columns
        if col.startswith('text_') and (col.endswith('_file') or col.endswith('_patch'))
    ]
    return text_metric_columns


def _run_single_experiment(X_project_full: pd.DataFrame, y_project: pd.Series, feature_columns: List[str], project_name: str, experiment_name: str):
    """単一の実験（特定の特徴量セット）をN回繰り返し、比率ホールドアウトで学習/評価する。"""
    # ランダムベースラインは特徴量を使わないため、空でも実行
    if not feature_columns and settings.SELECTED_MODEL != 'random':
        print("  特徴量が選択されていないため、スキップします。")
        return None, None, None, None
    
    if settings.SELECTED_MODEL == 'random':
        X_project_exp = X_project_full.copy()
    else:
        existing_features = [col for col in feature_columns if col in X_project_full.columns]
        if not existing_features:
            print("  有効な特徴量がデータセットに存在しないため、スキップします。")
            return None, None, None, None
        X_project_exp = X_project_full[existing_features].copy()
       
    all_runs_metrics = []
    all_runs_importances = []
    all_runs_oos_preds = []
    
    print(f"  {settings.N_REPETITIONS}回の繰り返し（比率ホールドアウト）評価を開始します...")
    for i in range(settings.N_REPETITIONS):
        print(f"    --- Repetition {i + 1}/{settings.N_REPETITIONS} ---")
        run_random_state = settings.RANDOM_STATE + i

        # 比率で時系列ホールドアウト分割（先頭を学習、後半をテスト）
        X_train, X_test, y_train, y_test, train_idx, test_idx = time_series_train_test_split(
            X_project_exp, y_project, train_ratio=settings.SIMPLE_SPLIT_TRAIN_RATIO
        )

        # 単回学習・評価
        metrics, importances_df, y_pred_proba = evaluation.train_and_evaluate_fold(
            X_train, y_train, X_test, y_test, run_random_state
        )

        if metrics is None:
            print("      学習/評価をスキップ（データ不十分または単一クラス）")
            continue

        metrics['fold'] = 1              # 単一分割
        metrics['repetition'] = i
        all_runs_metrics.append(metrics)

        if importances_df is not None:
            all_runs_importances.append(importances_df)

        if y_pred_proba is not None:
            # テスト部のみ確率を埋める
            oos_series = pd.Series(index=X_project_exp.index, dtype=float, name=evaluation.PRED_SERIES_NAME)
            oos_series.loc[test_idx] = y_pred_proba
            all_runs_oos_preds.append(oos_series)
    
    print(f"  {settings.N_REPETITIONS}回の繰り返し評価が完了しました。")
    
    avg_metrics, avg_importances = reporting.summarize_project_results(
        all_runs_metrics, all_runs_importances, project_name
    )
    
    # Foldごとの詳細（ここでは各反復=1 fold）
    per_fold_metrics_df = pd.DataFrame(all_runs_metrics)
    
    # 反復平均の予測確率（テスト部のみ値が入り、学習部はNaNのまま）
    avg_preds = None
    if all_runs_oos_preds:
        preds_df = pd.concat(all_runs_oos_preds, axis=1)
        avg_preds = preds_df.mean(axis=1)

    return avg_metrics, avg_importances, per_fold_metrics_df, avg_preds


def run_experiment_for_project(X_project_full: pd.DataFrame, y_project: pd.Series, feature_columns_full: List[str], project_name: str) -> Dict[str, Any]:
    """単一プロジェクトに対して、定義された実験セットを実行する。"""
    
    kamei_features_exist = [col for col in settings.KAMEI_FEATURES if col in feature_columns_full]
    vccfinder_features_exist = [col for col in settings.VCCFINDER_FEATURES if col in feature_columns_full]
    proj_commit_features_exist = [col for col in settings.PROJECT_COMMIT_PERCENT_FEATURES if col in feature_columns_full]
    proj_total_features_exist = [col for col in settings.PROJECT_TOTAL_PERCENT_FEATURES if col in feature_columns_full]
    proj_all_features_exist = [col for col in settings.PROJECT_ALL_PERCENT_FEATURES if col in feature_columns_full]

    # experiments = {
    #     "exp1": {"name": "実験1: Kamei", "features": kamei_features_exist},
    #     # "exp2": {"name": "実験2: Kamei + Project Commit", "features": list(set(kamei_features_exist + proj_commit_features_exist))},
    #     "exp2": {"name": "実験3: Kamei + Project Total", "features": list(set(kamei_features_exist + proj_total_features_exist))},
    #     # "exp4": {"name": "実験4: Kamei + Project All", "features": list(set(kamei_features_exist + proj_all_features_exist))},
    #     "exp3": {"name": "実験5: VCCFinder", "features": vccfinder_features_exist},
    #     # "exp6": {"name": "実験6: VCCFinder + Project Commit", "features": list(set(vccfinder_features_exist + proj_commit_features_exist))},
    #     "exp4": {"name": "実験7: VCCFinder + Project Total", "features": list(set(vccfinder_features_exist + proj_total_features_exist))},
    #     # "exp8": {"name": "実験8: VCCFinder + Project All", "features": list(set(vccfinder_features_exist + proj_all_features_exist))},
    #     # "exp9": {"name": "実験9: Project Commit", "features": proj_commit_features_exist},
    #     "exp5": {"name": "実験10: Project Total", "features": proj_total_features_exist},
    #     # "exp11": {"name": "実験11: Project All", "features": proj_all_features_exist},
    # }
    if settings.SELECTED_MODEL == 'random':
        # ベースラインは1本だけ（特徴量に依存させない）
        experiments = {"exp0": {"name": "実験R: RandomBaseline", "features": []}}
    else:
        experiments = {
            "exp1": {"name": "実験1: Kamei", "features": kamei_features_exist},
            "exp2": {"name": "実験2: Kamei + Project Total", "features": list(set(kamei_features_exist + proj_total_features_exist))},
            "exp3": {"name": "実験3: VCCFinder", "features": vccfinder_features_exist},
            "exp4": {"name": "実験4: VCCFinder + Project Total", "features": list(set(vccfinder_features_exist + proj_total_features_exist))},
            "exp5": {"name": "実験5: Project Total", "features": proj_total_features_exist},
        }

    project_results = {}
    for exp_key, exp_info in experiments.items():
        metrics, importances, per_fold_df, avg_preds = _run_single_experiment(
            X_project_full, y_project, exp_info["features"], project_name, exp_info["name"]
        )
        project_results[f"{exp_key}_metrics"] = metrics
        project_results[f"{exp_key}_importances"] = importances
        project_results[f"{exp_key}_per_fold_metrics"] = per_fold_df
        project_results[f"feature_columns_{exp_key}"] = exp_info["features"]
        project_results[f"predicted_risk_{exp_key}"] = avg_preds

    return project_results


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="プロジェクトごとのVCC予測モデルの学習と評価を実行します。")
    parser.add_argument("-p", "--project", type=str, help="処理対象の単一プロジェクト名を指定します。指定しない場合は全プロジェクトが対象です。")
    args = parser.parse_args()

    if args.project:
        print(f"指定されたプロジェクト '{args.project}' を処理します。")
        search_pattern = os.path.join(settings.BASE_DATA_DIRECTORY, args.project, '*_daily_aggregated_metrics.csv')
    else:
        print("全プロジェクトを処理します。")
        search_pattern = os.path.join(settings.BASE_DATA_DIRECTORY,'*/*_daily_aggregated_metrics.csv')

    csv_files = glob.glob(search_pattern, recursive=True)

    if not csv_files:
        if args.project:
            print(f"エラー: プロジェクト '{args.project}' のCSVファイルがパス '{search_pattern}' に見つかりません。")
        else:
            print(f"エラー: パス '{settings.BASE_DATA_DIRECTORY}' にCSVファイルが見つかりません。")
        return

    print(f"{len(csv_files)}個のプロジェクトCSVファイルを処理します。")
    results_base_dir = settings.RESULTS_BASE_DIRECTORY
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"結果は '{results_base_dir}' に保存されます。")

    # experiment_definitions = {
    #     "exp1": {"name": "実験1: Kamei"},
    #     # "exp2": {"name": "実験2: Kamei + Project Commit"},
    #     "exp2": {"name": "実験3: Kamei + Project Total"},
    #     # "exp4": {"name": "実験4: Kamei + Project All"},
    #     "exp3": {"name": "実験5: VCCFinder"},
    #     # "exp6": {"name": "実験6: VCCFinder + Project Commit"},
    #     "exp4": {"name": "実験7: VCCFinder + Project Total"},
    #     # "exp8": {"name": "実験8: VCCFinder + Project All"},
    #     # "exp9": {"name": "実験9: Project Commit"},
    #     "exp5": {"name": "実験10: Project Total"},
    #     # "exp11": {"name": "実験11: Project All"},
    # }
    if settings.SELECTED_MODEL == 'random':
        experiment_definitions = {"exp0": {"name": "実験R: RandomBaseline"}}
    else:
        experiment_definitions = {
            "exp1": {"name": "実験1: Kamei"},
            "exp2": {"name": "実験3: Kamei + Project Total"},
            "exp3": {"name": "実験5: VCCFinder"},
            "exp4": {"name": "実験7: VCCFinder + Project Total"},
            "exp5": {"name": "実験10: Project Total"},
        }

    for project_idx, csv_file in enumerate(csv_files):
        project_name = os.path.basename(os.path.dirname(csv_file))
        
        print(f"\n--- プロジェクト {project_idx + 1}/{len(csv_files)}: '{project_name}' ({csv_file}) ---")

        project_results_dir = os.path.join(results_base_dir, project_name)
        os.makedirs(project_results_dir, exist_ok=True)

        try:
            df = pd.read_csv(csv_file, low_memory=False)
        except Exception as e:
            print(f"  エラー: ファイル '{csv_file}' の読み込みに失敗しました: {e}")
            continue

        prepared_output = data_preparation.preprocess_dataframe_for_within_project(df, df)
        if not prepared_output:
            continue
        
        X_project_full, y_project, _, feature_columns_full = prepared_output
        
        if y_project.value_counts().min() < settings.MIN_SAMPLES_THRESHOLD:
            print(f"  プロジェクト '{project_name}': 少数クラスのサンプル数がしきい値 ({settings.MIN_SAMPLES_THRESHOLD}) 未満のためスキップします。")
            continue

        if y_project.nunique() < 2:
            print(f"  プロジェクト '{project_name}': ターゲット変数が1クラスのみのためスキップします。")
            continue
            
        project_results = run_experiment_for_project(X_project_full, y_project, feature_columns_full, project_name)
        
        # 元のデータフレームから、実際にモデルで使用されたデータ行のみを抽出
        df_with_predictions = df.loc[X_project_full.index].copy()
        
        for exp_key, exp_name in experiment_definitions.items():
            metrics = project_results.get(f"{exp_key}_metrics")
            importances_df = project_results.get(f"{exp_key}_importances")
            per_fold_df = project_results.get(f"{exp_key}_per_fold_metrics")
            
            # 予測確率のSeriesを取得し、元のデータフレームに追加
            predicted_risks = project_results.get(f"predicted_risk_{exp_key}")
            if predicted_risks is not None:
                column_name = f"predicted_risk_{exp_name}"
                df_with_predictions[column_name] = predicted_risks

            if metrics:
                metrics_path = os.path.join(project_results_dir, f"{exp_key}_metrics.json")
                try:
                    with open(metrics_path, 'w') as f:
                        serializable_metrics = json.loads(pd.Series(metrics).to_json())
                        json.dump(serializable_metrics, f, indent=4)
                    print(f"  {exp_key} の評価指標を保存しました: {metrics_path}")
                except Exception as e:
                    print(f"  エラー: {exp_key} の評価指標の保存に失敗しました: {e}")

            if importances_df is not None and not importances_df.empty:
                importances_path = os.path.join(project_results_dir, f"{exp_key}_importances.csv")
                try:
                    importances_df.to_csv(importances_path, index=False)
                    print(f"  {exp_key} の特徴量重要度を保存しました: {importances_path}")
                except Exception as e:
                    print(f"  エラー: {exp_key} の特徴量重要度の保存に失敗しました: {e}")

            # Foldごとの詳細な結果をCSVとして保存
            if per_fold_df is not None and not per_fold_df.empty:
                per_fold_path = os.path.join(project_results_dir, f"{exp_key}_per_fold_metrics.csv")
                try:
                    # classification_report_dict/混同行列は保存前に削除
                    if 'classification_report_dict' in per_fold_df.columns:
                        per_fold_df.drop(columns=['classification_report_dict'], inplace=True)
                    if 'confusion_matrix' in per_fold_df.columns:
                        per_fold_df.drop(columns=['confusion_matrix'], inplace=True)
                        
                    per_fold_df.to_csv(per_fold_path, index=False)
                    print(f"  {exp_key} のFoldごとの詳細な性能を保存しました: {per_fold_path}")
                except Exception as e:
                    print(f"  エラー: {exp_key} のFoldごとの性能の保存に失敗しました: {e}")
                    
        # 予測確率が追加されたDataFrameを新しいCSVファイルに保存
        output_prediction_csv_path = os.path.join(project_results_dir, f"{project_name}_daily_aggregated_metrics_with_predictions.csv")
        df_with_predictions.to_csv(output_prediction_csv_path, index=False)
        print(f"  予測確率を追加したデータを保存しました: {output_prediction_csv_path}")

        del df, X_project_full, y_project, project_results, df_with_predictions

    print("\n--- 全てのプロジェクトの処理が完了しました ---")


if __name__ == "__main__":
    main()