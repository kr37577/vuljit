import os
import glob
import pandas as pd
import numpy as np
import argparse

import settings
import data_preparation
from evaluation_regression import run_time_series_regression


def _resolve_experiments(feature_columns_full: list[str]) -> dict:
    """classification側(main_per_project.py)と同様の実験定義を構築し、
    実データに存在する特徴量のみを含むようフィルタする。
    """
    kamei_features_exist = [col for col in settings.KAMEI_FEATURES if col in feature_columns_full]
    vccfinder_features_exist = [col for col in settings.VCCFINDER_FEATURES if col in feature_columns_full]
    proj_total_features_exist = [col for col in settings.PROJECT_TOTAL_PERCENT_FEATURES if col in feature_columns_full]

    experiments = {
        "exp1": {"name": "exp1: Kamei", "features": kamei_features_exist},
        "exp2": {"name": "exp2: Kamei + Coverage", "features": list(set(kamei_features_exist + proj_total_features_exist))},
        "exp3": {"name": "exp3: VCCFinder", "features": vccfinder_features_exist},
        "exp4": {"name": "exp4: VCCFinder + Coverage", "features": list(set(vccfinder_features_exist + proj_total_features_exist))},
        "exp5": {"name": "exp5: Coverage", "features": proj_total_features_exist},
    }
    return experiments


def run_project_regression(csv_file: str, project_name: str, out_dir: str, repetitions: int = None,
                           model_name: str = 'random_forest', target_type: str = 'count',
                           experiment: str = 'exp4'):
    try:
        df = pd.read_csv(csv_file, low_memory=False)
    except Exception as e:
        print(f"  Error: failed to read '{csv_file}': {e}")
        return

    prepared = data_preparation.preprocess_dataframe_for_within_project(df, df)
    if not prepared:
        return
    X_full, _, _, feature_cols = prepared

    # 実験定義から使用特徴量を決定
    experiments = _resolve_experiments(feature_cols)
    if experiment not in experiments:
        print(f"  Error: unknown experiment '{experiment}'. Choose from exp1..exp5.")
        return
    selected_feats = [c for c in experiments[experiment]["features"] if c in X_full.columns]
    if not selected_feats:
        print(f"  Skip '{project_name}': no features for {experiment} present in data.")
        return
    X_full = X_full[selected_feats].copy()

    # Align regression target (counts) with processed indices
    y_reg = pd.to_numeric(df.loc[X_full.index].get('vcc_commit_count', 0), errors='coerce').fillna(0).clip(lower=0).astype(float)
    if y_reg.sum() <= 0:
        print(f"  Skip '{project_name}': no positive counts in target.")
        return

    reps = repetitions if repetitions is not None else settings.N_REPETITIONS
    all_fold_metrics = []
    oos_list = []

    for r in range(reps):
        seed = settings.RANDOM_STATE + r
        print(f"    Regression repetition {r+1}/{reps} (seed={seed}) model={model_name} target={target_type} exp={experiment}")
        fold_metrics, oos = run_time_series_regression(
            X_full, y_reg, project_name, seed, model_name=model_name, target_type=target_type,
        )
        for m in fold_metrics:
            m['repetition'] = r
            m['experiment'] = experiment
            m['n_features'] = len(selected_feats)
        all_fold_metrics.extend(fold_metrics)
        oos_list.append(oos)

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    proj_dir = os.path.join(out_dir, project_name)
    os.makedirs(proj_dir, exist_ok=True)
    metrics_df.to_csv(os.path.join(proj_dir, f'regression_fold_metrics_{model_name}_{target_type}_{experiment}.csv'), index=False)

    # Average OOS predictions across repetitions
    avg_oos = None
    if oos_list:
        avg_oos = pd.concat(oos_list, axis=1).mean(axis=1)
        df_with_pred = df.loc[X_full.index].copy()
        # Predicted counts always available from oos
        canonical_map = {
            'exp1': 'Kamei',
            'exp2': 'Kamei_Coverage',
            'exp3': 'VCCFinder',
            'exp4': 'VCCFinder_Coverage',
            'exp5': 'Coverage',
        }
        canonical = canonical_map.get(experiment, experiment)
        count_col = f'predicted_count_{canonical}'
        df_with_pred[count_col] = avg_oos.values

        # If size available, also provide density column for convenience
        size_series = None
        for cand in ('change_size', 'total_lines_changed'):
            if cand in df_with_pred.columns:
                size_series = pd.to_numeric(df_with_pred[cand], errors='coerce').fillna(0.0).astype(float)
                break
        if size_series is None and 'lines_added' in df_with_pred.columns and 'lines_deleted' in df_with_pred.columns:
            size_series = pd.to_numeric(df_with_pred['lines_added'], errors='coerce').fillna(0.0).astype(float) \
                          + pd.to_numeric(df_with_pred['lines_deleted'], errors='coerce').fillna(0.0).astype(float)
        if size_series is not None:
            denom = np.maximum(size_series.values, 1.0)
            df_with_pred[f'predicted_density_{canonical}'] = (df_with_pred[count_col].values.astype(float) / denom)

        # Save enriched daily CSV (counts); include model/target/experiment in name to avoid clashes
        out_csv = os.path.join(proj_dir, f"{project_name}_daily_aggregated_metrics_with_predicted_counts_{model_name}_{target_type}_{experiment}.csv")
        df_with_pred.to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser(description='Per-project regression to predict defect counts (daily).')
    ap.add_argument('--project', default=None, help='Single project name to run; default=all projects in BASE_DATA_DIRECTORY')
    ap.add_argument('--model', default=os.getenv('VULJIT_REGRESSION_MODEL', 'xgboost'),
                    choices=['random_forest', 'linear', 'cart', 'xgboost', 'dummy'])
    ap.add_argument('--target', default=os.getenv('VULJIT_REGRESSION_TARGET', 'count'),
                    choices=['count', 'density'])
    ap.add_argument('--experiment', default=os.getenv('VULJIT_REGRESSION_EXPERIMENT', 'all'),
                    choices=['exp1','exp2','exp3','exp4','exp5','all'],
                    help='特徴量セットの選択。all を指定すると全実験(exp1..exp5)を実行')
    args = ap.parse_args()

    if args.project:
        search = os.path.join(settings.BASE_DATA_DIRECTORY, args.project, '*_daily_aggregated_metrics.csv')
    else:
        search = os.path.join(settings.BASE_DATA_DIRECTORY, '*/*_daily_aggregated_metrics.csv')

    files = glob.glob(search)
    if not files:
        print(f"No project CSVs found under {search}")
        return

    # Use settings.RESULTS_BASE_DIRECTORY but segregate under 'regression'
    base_out = os.path.join(settings.RESULTS_BASE_DIRECTORY + '_regression')
    os.makedirs(base_out, exist_ok=True)
    print(f"Outputs -> {base_out}")

    # 実行する実験セットを決定
    if args.experiment == 'all':
        experiments_to_run = ['exp1','exp2','exp3','exp4','exp5']
    else:
        experiments_to_run = [args.experiment]

    for idx, csv in enumerate(files, 1):
        project = os.path.basename(os.path.dirname(csv))
        print(f"\n--- [{idx}/{len(files)}] Project: {project}")
        for exp in experiments_to_run:
            print(f"      -> experiment: {exp}")
            run_project_regression(csv, project, base_out,
                                   model_name=args.model,
                                   target_type=args.target,
                                   experiment=exp)


if __name__ == '__main__':
    main()
