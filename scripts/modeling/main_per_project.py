# main.py
import os
import re
import pandas as pd
import glob
import json
from typing import List, Dict, Any, Optional
import argparse
import numpy as np
import random
from collections.abc import Mapping

# code/ ディレクトリ内の各モジュールをインポート
import settings
import data_preparation
import evaluation
import reporting
import cross_project_data


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


def _normalize_for_json(value: Any) -> Any:
    """
    JSONに安全に書き出せるよう、np.nan や numpy 型を標準のPython型へ正規化する。
    """
    if value is None:
        return None

    if isinstance(value, Mapping):
        return {k: _normalize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(v) for v in value]

    if isinstance(value, np.generic):
        python_value = value.item()
        if isinstance(python_value, float) and (np.isnan(python_value) or np.isinf(python_value)):
            return None
        return python_value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value

    if isinstance(value, (pd.Series, pd.Index)):
        return [_normalize_for_json(v) for v in value.tolist()]

    return value


def _parse_project_list(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(',') if item.strip()]


def _read_project_list_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    try:
        with open(path, 'r') as handle:
            return [line.strip() for line in handle if line.strip()]
    except FileNotFoundError:
        print(f"[WARN] 指定されたトレーニングプロジェクトファイルが見つかりません: {path}")
    except Exception as exc:
        print(f"[WARN] トレーニングプロジェクトファイルの読み取りに失敗しました ({path}): {exc}")
    return []


def _discover_project_csvs(base_dir: str) -> Dict[str, str]:
    pattern = os.path.join(base_dir, '*', '*_daily_aggregated_metrics.csv')
    mapping: Dict[str, str] = {}
    for csv_path in sorted(glob.glob(pattern)):
        project_name = os.path.basename(os.path.dirname(csv_path))
        mapping.setdefault(project_name, csv_path)
    return mapping


def _sanitize_for_path(value: str) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_')
    return sanitized or 'default'


def _format_scope_label(scope: str, projects: List[str]) -> str:
    if scope in {'all', 'exclude_target'}:
        return scope
    if not projects:
        return 'list_empty'
    sorted_names = sorted(projects)
    preview = '_'.join(sorted_names)
    if len(preview) > 60:
        preview = '_'.join(sorted_names[:5]) + f"_plus{len(sorted_names) - 5}"
    return f"list_{preview}"


def _resolve_training_projects(
    scope: str,
    explicit_projects: List[str],
    available_projects: Dict[str, str],
    target_project: str,
) -> List[str]:
    scope = (scope or 'list').lower()
    explicit_set = {p for p in explicit_projects if p}

    if scope == 'all':
        candidates = sorted(available_projects.keys())
    elif scope == 'exclude_target':
        candidates = sorted(p for p in available_projects.keys() if p != target_project)
    else:
        candidates = sorted(explicit_set)

    if scope != 'list' and explicit_set:
        candidates = [p for p in candidates if p in explicit_set]

    deduped: List[str] = []
    seen = set()
    for project in candidates:
        if project == target_project or project in seen:
            continue
        if project not in available_projects:
            continue
        deduped.append(project)
        seen.add(project)

    return deduped


def _run_single_experiment(
    X_project_full: pd.DataFrame,
    y_project: pd.Series,
    feature_columns: List[str],
    project_name: str,
    experiment_name: str,
    external_training: Optional[Dict[str, Any]] = None,
    cross_project_eval_mode: str = 'fold',
):
    """単一の実験（特定の特徴量セット）をN回繰り返し実行し、結果をまとめて返す。"""
    
    # ランダムベースラインは特徴量を使わないため、空でも実行
    if not feature_columns and settings.SELECTED_MODEL != 'random':
        print("  特徴量が選択されていないため、スキップします。")
        return None, None, None, None, []

    alignment_features: Optional[List[str]] = None

    if settings.SELECTED_MODEL == 'random':
        X_project_exp = X_project_full.copy()
        used_feature_columns = []
        alignment_features = list(X_project_full.columns)
    else:
        existing_features = [col for col in feature_columns if col in X_project_full.columns]
        if not existing_features:
            print("  有効な特徴量がデータセットに存在しないため、スキップします。")
            return None, None, None, None, []
        # ▼ 修正: 早期returnの後にあった代入をここに移動
        X_project_exp = X_project_full[existing_features].copy()
        used_feature_columns = existing_features
        alignment_features = existing_features
       

    all_runs_metrics = []
    all_runs_importances = []
    all_runs_oos_preds = []
    
    if external_training and not alignment_features:
        alignment_features = list(X_project_full.columns)

    external_payload = None
    if external_training:
        X_external_source = external_training.get('X')
        y_external_source = external_training.get('y')
        if X_external_source is None or y_external_source is None:
            print("  [CROSS] 外部学習データが無効なため、この実験をスキップします。")
            return None, None, None, None, []
        if alignment_features:
            X_external_exp = X_external_source.reindex(columns=alignment_features, fill_value=0)
        else:
            X_external_exp = X_external_source.copy()
        external_payload = {
            'X': X_external_exp,
            'y': y_external_source,
            'meta': external_training,
            'eval_mode': cross_project_eval_mode,
        }

    print(f"  {settings.N_REPETITIONS}回の繰り返し評価を開始します...")
    for i in range(settings.N_REPETITIONS):
        print(f"    --- Repetition {i + 1}/{settings.N_REPETITIONS} ---")
        run_random_state = settings.RANDOM_STATE + i
        # Seed global RNGs for reproducibility per run
        np.random.seed(run_random_state)
        random.seed(run_random_state)
        
        if external_payload:
            fold_metrics, fold_importances, out_of_sample_predictions = evaluation.run_cross_project_validation(
                X_project_exp,
                y_project,
                project_name,
                run_random_state,
                external_payload['X'],
                external_payload['y'],
                eval_mode=external_payload.get('eval_mode', 'fold')
            )
        else:
            fold_metrics, fold_importances, out_of_sample_predictions = evaluation.run_cross_validation_for_project(
                X_project_exp, y_project, project_name, run_random_state
            )
        
        if fold_metrics:
            # ▼▼▼【修正箇所】繰り返し回数の情報を追加 ▼▼▼
            for metric_dict in fold_metrics:
                metric_dict['repetition'] = i
                if external_training:
                    metric_dict['train_projects'] = external_training.get('projects', [])
                    metric_dict['train_scope'] = external_training.get('scope_label')
                    metric_dict['mode'] = 'cross_project'
                    metric_dict['cross_eval_mode'] = cross_project_eval_mode
                else:
                    metric_dict['mode'] = 'within_project'
            all_runs_metrics.extend(fold_metrics)
        if fold_importances:
            all_runs_importances.extend(fold_importances)
        if out_of_sample_predictions is not None:
            all_runs_oos_preds.append(out_of_sample_predictions)
    
    
    print(f"  {settings.N_REPETITIONS}回の繰り返し評価が完了しました。")
    
    avg_metrics, avg_importances = reporting.summarize_project_results(
        all_runs_metrics, all_runs_importances, project_name
    )
    
    # ▼▼▼【修正箇所】Foldごとの詳細なメトリクスをDataFrameに変換して返す ▼▼▼
    per_fold_metrics_df = pd.DataFrame(all_runs_metrics)
    
    
    # 【追加】繰り返し実行した予測確率の平均を計算
    avg_preds = None
    if all_runs_oos_preds:
        # リスト内のSeriesを連結してDataFrameを作成し、行ごとに平均を計算
        preds_df = pd.concat(all_runs_oos_preds, axis=1)
        avg_preds = preds_df.mean(axis=1)

    return avg_metrics, avg_importances, per_fold_metrics_df, avg_preds, used_feature_columns


def run_experiment_for_project(
    X_project_full: pd.DataFrame,
    y_project: pd.Series,
    feature_columns_full: List[str],
    project_name: str,
    external_training: Optional[Dict[str, Any]] = None,
    cross_project_eval_mode: str = 'fold',
) -> Dict[str, Any]:
    """単一プロジェクトに対して、定義された実験セットを実行する。"""
    
    kamei_features_exist = [col for col in settings.KAMEI_FEATURES if col in feature_columns_full]
    vccfinder_features_exist = [col for col in settings.VCCFINDER_FEATURES if col in feature_columns_full]
    proj_commit_features_exist = [col for col in settings.PROJECT_COMMIT_PERCENT_FEATURES if col in feature_columns_full]
    proj_total_features_exist = [col for col in settings.PROJECT_TOTAL_PERCENT_FEATURES if col in feature_columns_full]
    proj_all_features_exist = [col for col in settings.PROJECT_ALL_PERCENT_FEATURES if col in feature_columns_full]
    if settings.SELECTED_MODEL == 'random':
        # ベースラインは1本だけ（特徴量に依存させない）
        experiments = {"exp0": {"name": "expR: RandomBaseline", "features": []}}
    else:
        experiments = {
            "exp1": {"name": "exp1: Kamei", "features": kamei_features_exist},
            "exp2": {"name": "exp2: Kamei + Ccoverage", "features": list(set(kamei_features_exist + proj_total_features_exist))},
            "exp3": {"name": "exp3: VCCFinder", "features": vccfinder_features_exist},
            "exp4": {"name": "exp4: VCCFinder + Coverage", "features": list(set(vccfinder_features_exist + proj_total_features_exist))},
            "exp5": {"name": "exp5: Coverage", "features": proj_total_features_exist},
        }

    project_results = {}
    for exp_key, exp_info in experiments.items():
        # ▼▼▼【修正箇所】per_fold_df を受け取る ▼▼▼
        metrics, importances, per_fold_df, avg_preds, used_features = _run_single_experiment(
            X_project_full,
            y_project,
            exp_info["features"],
            project_name,
            exp_info["name"],
            external_training=external_training,
            cross_project_eval_mode=cross_project_eval_mode,
        )
        project_results[f"{exp_key}_metrics"] = metrics
        project_results[f"{exp_key}_importances"] = importances
        # ▼▼▼【修正箇所】per_fold_df を結果辞書に追加 ▼▼▼
        project_results[f"{exp_key}_per_fold_metrics"] = per_fold_df
        project_results[f"feature_columns_{exp_key}"] = used_features
        # 【追加】平均予測確率を結果に追加
        project_results[f"predicted_risk_{exp_key}"] = avg_preds

    return project_results


def main():
    """メイン実行関数"""
    valid_scopes = {'list', 'all', 'exclude_target'}
    default_train_scope = settings.CROSS_PROJECT_DEFAULT_SCOPE if settings.CROSS_PROJECT_DEFAULT_SCOPE in valid_scopes else 'list'
    scope_default_overridden = settings.CROSS_PROJECT_DEFAULT_SCOPE not in valid_scopes

    parser = argparse.ArgumentParser(description="プロジェクトごとのVCC予測モデルの学習と評価を実行します。")
    parser.add_argument("-p", "--project", type=str, help="処理対象の単一プロジェクト名を指定します。指定しない場合は全プロジェクトが対象です。")
    parser.add_argument("--cross-project", action="store_true", help="ターゲット以外のプロジェクトで学習し、指定プロジェクトで評価するクロスプロジェクトモードを有効化します。")
    parser.add_argument(
        "--train-projects",
        type=str,
        default=settings.CROSS_PROJECT_TRAIN_PROJECTS,
        help="クロスプロジェクト学習に使用するプロジェクト名をカンマ区切りで指定します。",
    )
    parser.add_argument(
        "--train-projects-file",
        type=str,
        default=settings.CROSS_PROJECT_TRAIN_PROJECTS_FILE,
        help="クロスプロジェクト学習に使用するプロジェクト名が1行ずつ記載されたファイルを指定します。",
    )
    parser.add_argument(
        "--train-scope",
        choices=sorted(valid_scopes),
        default=default_train_scope,
        help="クロスプロジェクト学習時のトレーニングプロジェクト選択ポリシーです。",
    )
    parser.add_argument(
        "--cross-project-mode",
        choices=['fold', 'full'],
        default='fold',
        help="クロスプロジェクト評価でFoldを使うか(fullでターゲット全体を一括評価)。",
    )
    args = parser.parse_args()

    if args.cross_project and scope_default_overridden:
        print(f"[WARN] 無効な既定クロスプロジェクトスコープ '{settings.CROSS_PROJECT_DEFAULT_SCOPE}' が指定されたため 'list' を使用します。")

    cross_mode = bool(args.cross_project)
    train_scope = (args.train_scope or 'list').lower()
    cross_eval_mode = (args.cross_project_mode or 'fold').lower()

    if cross_mode and not args.project:
        print("エラー: クロスプロジェクトモードでは --project でターゲットを指定してください。")
        return

    available_project_csvs = _discover_project_csvs(settings.BASE_DATA_DIRECTORY)
    if not available_project_csvs:
        print(f"エラー: パス '{settings.BASE_DATA_DIRECTORY}' にCSVファイルが見つかりません。")
        return

    if args.project:
        csv_path = available_project_csvs.get(args.project)
        if not csv_path:
            print(f"エラー: プロジェクト '{args.project}' のCSVが見つかりません。")
            return
        csv_files = [csv_path]
        print(f"指定されたプロジェクト '{args.project}' を処理します。")
    else:
        csv_files = list(available_project_csvs.values())
        print("全プロジェクトを処理します。")

    if not csv_files:
        print("エラー: 処理対象のCSVファイルが見つかりません。")
        return

    explicit_projects = _parse_project_list(args.train_projects)
    explicit_projects.extend(_read_project_list_file(args.train_projects_file))

    if cross_mode and train_scope == 'list' and not explicit_projects:
        print("エラー: --train-scope list の場合、--train-projects または --train-projects-file で学習対象を指定してください。")
        return

    print(f"{len(csv_files)}個のプロジェクトCSVファイルを処理します。")
    results_base_dir = settings.RESULTS_BASE_DIRECTORY
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"結果は '{results_base_dir}' に保存されます。")


   
    if settings.SELECTED_MODEL == 'random':
        experiment_definitions = {"exp0": {"name": "expR: RandomBaseline", "canonical": "RandomBaseline"}}
    else:
        # name はログ等の表示用、canonical は列名など安定な識別子として利用
        experiment_definitions = {
            "exp1": {"name": "exp1: Kamei",                       "canonical": "Kamei"},
            "exp2": {"name": "exp2: Kamei + Coverage",            "canonical": "Kamei_Coverage"},
            "exp3": {"name": "exp3: VCCFinder",                  "canonical": "VCCFinder"},
            "exp4": {"name": "exp4: VCCFinder + Coverage",       "canonical": "VCCFinder_Coverage"},
            "exp5": {"name": "exp5: Coverage",                   "canonical": "Coverage"},
        }

    for project_idx, csv_file in enumerate(csv_files):
        project_name = os.path.basename(os.path.dirname(csv_file))
        
        print(f"\n--- プロジェクト {project_idx + 1}/{len(csv_files)}: '{project_name}' ({csv_file}) ---")

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

        external_training_payload = None
        scope_tag = None
        scope_label = None
        if cross_mode:
            training_projects = _resolve_training_projects(
                train_scope,
                explicit_projects,
                available_project_csvs,
                project_name,
            )
            if not training_projects:
                print("  [CROSS] 学習対象のプロジェクトが決定できなかったためスキップします。")
                continue

            training_set = cross_project_data.build_training_set(
                training_projects,
                settings.BASE_DATA_DIRECTORY,
                feature_columns_full,
            )
            if not training_set:
                print("  [CROSS] 学習データの構築に失敗したためスキップします。")
                continue

            scope_label = _format_scope_label(train_scope, training_set.projects)
            scope_tag = _sanitize_for_path(scope_label)

            external_training_payload = {
                'X': training_set.X,
                'y': training_set.y,
                'projects': training_set.projects,
                'scope_label': scope_label,
                'eval_mode': cross_eval_mode,
            }

            print(f"  [CROSS] {len(training_set.projects)} プロジェクト ({len(training_set.X)} サンプル) を学習に使用します。")
            if training_set.skipped_projects:
                skipped_msg = ', '.join(f"{k}:{v}" for k, v in training_set.skipped_projects.items())
                print(f"  [CROSS] スキップされたプロジェクト: {skipped_msg}")

        project_results_dir = os.path.join(results_base_dir, project_name)
        if cross_mode and scope_tag:
            project_results_dir = os.path.join(
                project_results_dir,
                settings.CROSS_PROJECT_RESULTS_SUBDIR,
                scope_tag,
            )
        os.makedirs(project_results_dir, exist_ok=True)

        project_results = run_experiment_for_project(
            X_project_full,
            y_project,
            feature_columns_full,
            project_name,
            external_training=external_training_payload,
            cross_project_eval_mode=cross_eval_mode,
        )
        
        # 【変更点】元のデータフレームから、実際にモデルで使用されたデータ行のみを抽出
        df_with_predictions = df.loc[X_project_full.index].copy()
        
        
        for exp_key, exp_info in experiment_definitions.items():
            metrics = project_results.get(f"{exp_key}_metrics")
            importances_df = project_results.get(f"{exp_key}_importances")
            per_fold_df = project_results.get(f"{exp_key}_per_fold_metrics") # ▼▼▼【追加】▼▼▼
            
            
            # 【追加】予測確率のSeriesを取得し、元のデータフレームに追加
            predicted_risks = project_results.get(f"predicted_risk_{exp_key}")
            if predicted_risks is not None:
                # どのモデルの予測かを安定名で表現（RQ3既定: VCCFinder_ProjTotal など）
                canonical = exp_info.get("canonical", exp_key)
                prob_col = f"predicted_risk_{canonical}"
                # 予測確率（Series）を列として追加。インデックスが自動的に揃えられる。
                df_with_predictions[prob_col] = predicted_risks

                # 予測確率を二値化したラベル列も出力（既定閾値は settings.PREDICTION_LABEL_THRESHOLD）
                thr = getattr(settings, 'PREDICTION_LABEL_THRESHOLD', 0.5)
                label_col = f"predicted_label_{canonical}"
                # NaNを保持するために Nullable Int64 dtype で作成
                try:
                    import pandas as _pd
                    pr = _pd.to_numeric(predicted_risks, errors='coerce')
                    labels = _pd.Series(_pd.NA, index=pr.index, dtype='Int64')
                    labels.loc[pr.notna() & (pr >= thr)] = 1
                    labels.loc[pr.notna() & (pr < thr)] = 0
                    df_with_predictions[label_col] = labels
                except Exception:
                    # フォールバック（型エラー等の際は単純比較で0/1、NaNは0として扱う）
                    df_with_predictions[label_col] = (df_with_predictions[prob_col] >= thr).astype(int)

            

            if metrics:
                metrics_path = os.path.join(project_results_dir, f"{exp_key}_metrics.json")
                try:
                    serializable_metrics = _normalize_for_json(metrics)
                    with open(metrics_path, 'w') as f:
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

            # ▼▼▼【追加】Foldごとの詳細な結果をCSVとして保存 ▼▼▼
            if per_fold_df is not None and not per_fold_df.empty:
                per_fold_path = os.path.join(project_results_dir, f"{exp_key}_per_fold_metrics.csv")
                try:
                    # classification_report_dict は複雑なオブジェクトなので、保存前に削除
                    if 'classification_report_dict' in per_fold_df.columns:
                        per_fold_df.drop(columns=['classification_report_dict'], inplace=True)
                    if 'confusion_matrix' in per_fold_df.columns:
                        per_fold_df.drop(columns=['confusion_matrix'], inplace=True)
                        
                    per_fold_df.to_csv(per_fold_path, index=False)
                    print(f"  {exp_key} のFoldごとの詳細な性能を保存しました: {per_fold_path}")
                except Exception as e:
                    print(f"  エラー: {exp_key} のFoldごとの性能の保存に失敗しました: {e}")
                    
        # 【追加】予測確率が追加されたDataFrameを新しいCSVファイルに保存
        prediction_suffix = ''
        if cross_mode and scope_tag:
            prediction_suffix = f"_cross_project_{scope_tag}"
        output_prediction_csv_path = os.path.join(
            project_results_dir,
            f"{project_name}_daily_aggregated_metrics_with_predictions{prediction_suffix}.csv",
        )
        df_with_predictions.to_csv(output_prediction_csv_path, index=False)
        print(f"  予測確率を追加したデータを保存しました: {output_prediction_csv_path}")


        del df, X_project_full, y_project, project_results,  df_with_predictions

    print("\n--- 全てのプロジェクトの処理が完了しました ---")


if __name__ == "__main__":
    main()
