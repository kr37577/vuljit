import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
import warnings
import re
import argparse

# グラフの日本語表示設定 (必要に応じて)
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']

# 警告を非表示にする
warnings.simplefilter('ignore', FutureWarning)

COLOR_PALETTE = {
    "XGBoost": "#1f77b4",       # ブルー
    "RandomForest": "#ff7f0e",  # オレンジ
    "Random": "#2ca02c"         # 緑 (追加)
}
## ------------------------------------------------------
## 設定
## ------------------------------------------------------
# 比較したいモデルのデータが格納されているベースディレクトリを辞書で設定します。
# キー: モデル名 (グラフの凡例などで使用)
# バリュー: 対応する結果が格納されているディレクトリのパス
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "datasets" / "model_outputs" 
# /work/riku-ka/vuljit/datasets/model_outputs
BASE_DIRS = {
    "XGBoost": RESULTS_ROOT / "xgboost",
    "RandomForest": RESULTS_ROOT / "random_forest",
    "Random": RESULTS_ROOT / "random",
}


## ------------------------------------------------------
## データ読み込み関数
## ------------------------------------------------------
def load_experiment_data(base_dir: Path, project: str, exp_number: int) -> tuple[pd.DataFrame | None, dict | None]:
    """
    指定されたベースディレクトリ、プロジェクト、実験番号のデータを読み込む関数
    """
    project_path = Path(project)
    importance_path = base_dir / project_path / f"exp{exp_number}_importances.csv"
    metrics_path = base_dir / project_path / f"exp{exp_number}_metrics.json"

    importance_df, metrics_dict = None, None

    # ファイルが存在する場合のみ読み込む
    if importance_path.exists():
        importance_df = pd.read_csv(importance_path)
        if 'importance' in importance_df.columns:
            importance_df['importance'] = importance_df['importance'].clip(lower=0)

    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics_dict = json.load(f)

    return importance_df, metrics_dict


## ------------------------------------------------------
## モデル間の性能を比較・可視化する関数 (★改良版)
## ------------------------------------------------------
def visualize_per_model_importance(all_metrics_df: pd.DataFrame, all_importances_df: pd.DataFrame, exp_num: int, num_projects: int):
    """
    評価指標はモデル間で比較し、特徴量重要度はモデルごとに個別のグラフで可視化する関数
    """
    model_names = all_metrics_df['model'].unique()
    print(f"\n--- Visualizing Performance for Exp {exp_num} ({num_projects} projects) across {len(model_names)} models ---")

    # 1. 評価指標の分布をモデル間で比較・可視化 (変更なし)
    plt.figure(figsize=(18, 9))

    median_order = all_metrics_df.groupby('Metric')['Value'].median().sort_values(ascending=False).index

    sns.violinplot(x='Metric', y='Value', hue='model', data=all_metrics_df, order=median_order,
                   palette=COLOR_PALETTE, inner='box', linewidth=1.5, saturation=0.8)

    plt.title(f'Metrics Comparison (Violin Plot) for Exp {exp_num} (across {num_projects} projects)', fontsize=18, weight='bold')
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    # ▼ 変更: Y軸範囲をデータから動的に決定（負のMCCも可視化）
    min_val = all_metrics_df['Value'].min(skipna=True)
    max_val = all_metrics_df['Value'].max(skipna=True)
    y_min = 0 if pd.isna(min_val) or min_val >= 0 else min(min_val * 1.05, -1.0)
    y_max = 1.0 if pd.isna(max_val) else max(1.0, max_val * 1.05)
    plt.ylim(y_min, y_max)
    # plt.ylim(0, 1.05)  # ← 固定範囲は削除
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model', fontsize=12)
    plt.tight_layout()
    # グラフを保存
    metrics_filename = f"exp{exp_num}_metrics_comparison.png"
    plt.savefig(metrics_filename)
    print(f"  📈 Metrics comparison plot saved as: {metrics_filename}")
    plt.show()
    plt.close()

    # 2. 特徴量重要度の分布をモデルごとに個別に可視化 (★変更点)
    print("  - Generating separate feature importance plots for each model...")
    for model_name in model_names:
        plt.figure(figsize=(14, 12))

        # 対象モデルの重要度データのみを抽出
        model_importances_df = all_importances_df[all_importances_df['model'] == model_name]
        
        if model_importances_df.empty:
            continue

        # モデルごとにTop 20の特徴量を決定
        median_importances = model_importances_df.groupby('feature')['importance'].median().sort_values(ascending=False)
        top_20_features = median_importances.head(20).index
        top_features_df = model_importances_df[model_importances_df['feature'].isin(top_20_features)]
        
        # 単一モデルのバイオリンプロットを描画 (hueは不要)
        sns.violinplot(x='importance', y='feature', data=top_features_df, order=top_20_features,
                       orient='h', color='skyblue', inner='box', linewidth=1.5, saturation=0.8)

        plt.title(f'Feature Importance ({model_name}) for Exp {exp_num} - Top 20', fontsize=18, weight='bold')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # モデル名をファイル名に含めて保存
        importance_filename = f"exp{exp_num}_feature_importance_{model_name}.png"
        plt.savefig(importance_filename)
        print(f"  📈 Feature importance plot for '{model_name}' saved as: {importance_filename}")
        plt.show()
        plt.close()


## ------------------------------------------------------
## 上位N件の性能出力ユーティリティ
## ------------------------------------------------------
def export_top_n_performance(final_metrics_df: pd.DataFrame, exp_num: int, out_dir: Path, top_n: int, metric_name: str, make_plots: bool = False):
    """
    指定メトリクス(metric_name)で、モデルごとにプロジェクト上位N件を抽出してCSV保存。
    必要に応じてモデル別の棒グラフも保存します。

    出力: 
      - exp{exp_num}_top{N}_{metric}_by_model.csv
      - (オプション) exp{exp_num}_top{N}_{metric}_{model}.png
    """
    if top_n is None or top_n <= 0:
        return

    # メトリクス名一致（大文字小文字を無視）でフィルタ
    metric_mask = final_metrics_df['Metric'].str.lower() == str(metric_name).lower()
    df_metric = final_metrics_df.loc[metric_mask].copy()
    if df_metric.empty:
        print(f"  ⚠️ No rows found for metric '{metric_name}'. Skipping top-{top_n} export.")
        return

    # NaNを除外し、Value降順（高い方が良い前提）でランク付け
    df_metric = df_metric.dropna(subset=['Value'])
    if df_metric.empty:
        print(f"  ⚠️ All values for '{metric_name}' are NaN. Skipping top-{top_n} export.")
        return

    # モデルごとに上位N件を抽出
    top_rows = []
    for model_name, g in df_metric.groupby('model'):
        g_sorted = g.sort_values('Value', ascending=False).head(top_n).copy()
        if g_sorted.empty:
            continue
        g_sorted['Rank'] = range(1, len(g_sorted) + 1)
        # 列の並びを整える
        g_sorted = g_sorted[['model', 'Rank', 'project', 'Metric', 'Value']]
        top_rows.append(g_sorted)

    if not top_rows:
        print(f"  ⚠️ No top-{top_n} rows computed for metric '{metric_name}'.")
        return

    top_df = pd.concat(top_rows, ignore_index=True)
    csv_path = out_dir / f"exp{exp_num}_top{top_n}_{metric_name}_by_model.csv"
    top_df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved top-{top_n} by model for '{metric_name}' to: {csv_path.name}")

    # オプション: モデルごとにバーグラフ出力
    if make_plots:
        for model_name, g in top_df.groupby('model'):
            if g.empty:
                continue
            plt.figure(figsize=(12, max(4, 0.5 * len(g))))
            g_sorted = g.sort_values('Value', ascending=False)
            plt.barh(g_sorted['project'], g_sorted['Value'], color=COLOR_PALETTE.get(model_name, '#888888'))
            plt.gca().invert_yaxis()  # 上位を上に
            plt.xlabel(metric_name)
            plt.ylabel('Project')
            plt.title(f"Top {len(g_sorted)} {metric_name} — {model_name} (Exp {exp_num})")
            plt.tight_layout()
            fig_path = out_dir / f"exp{exp_num}_top{top_n}_{metric_name}_{model_name}.png"
            plt.savefig(fig_path, dpi=150)
            print(f"  📈 Saved plot: {fig_path.name}")
            plt.close()


## ------------------------------------------------------
## 陽性(is_vcc=True)日数 上位N件の性能出力 + 可視化
## ------------------------------------------------------
def _find_daily_csv_for_project(project: str, valid_models: list[str]) -> Path | None:
    """モデルに依存しない日別集計CSVを、存在するモデルディレクトリから優先順で取得"""
    preference = ["XGBoost", "RandomForest", "Random"]
    project_path = Path(project)
    for model_name in preference:
        if model_name not in valid_models:
            continue
        project_root = BASE_DIRS[model_name] / project_path
        if not project_root.exists():
            continue
        candidates = sorted(project_root.glob("*_daily_aggregated_metrics_with_predictions.csv"))
        if candidates:
            return candidates[0]
    return None


def _count_positive_days(csv_path: Path) -> int:
    """is_vcc=True の行数をカウント"""
    try:
        df = pd.read_csv(csv_path, usecols=["is_vcc"])
    except Exception:
        df = pd.read_csv(csv_path)
        if "is_vcc" not in df.columns:
            return 0
    s = df["is_vcc"]
    if s.dtype == bool:
        return int(s.sum())
    return int(s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"]).sum())


def export_top_by_positive_days(final_metrics_df: pd.DataFrame,
                                exp_num: int,
                                out_dir: Path,
                                projects: list[str],
                                valid_models: list[str],
                                top_n: int,
                                metric_name: str | None = None,
                                make_plots: bool = False):
    """
    プロジェクトの陽性(is_vcc=True)日数の多い順に上位N件を抽出し、
    それらの性能（final_metrics_df に含まれる全メトリクス、任意で特定メトリクス）をCSV保存。

    出力:
      - exp{exp_num}_top{N}_by_positive_days_all_metrics.csv
      - exp{exp_num}_top{N}_by_positive_days_{metric}.csv (metric_name 指定時)
      - (オプション) exp{exp_num}_top{N}_by_positive_days_{metric}_{model}.png
    """
    if top_n is None or top_n <= 0:
        return

    pos_counts = []
    for proj in projects:
        p = _find_daily_csv_for_project(proj, valid_models)
        if p is None:
            continue
        try:
            cnt = _count_positive_days(p)
        except Exception:
            cnt = 0
        pos_counts.append((proj, cnt))

    if not pos_counts:
        print("  ⚠️ No positive-day counts could be computed (daily CSV not found). Skipping.")
        return

    pos_df = pd.DataFrame(pos_counts, columns=["project", "positive_days"]).sort_values("positive_days", ascending=False)
    top_projects = pos_df.head(top_n)["project"].tolist()

    # すべてのメトリクスで出力（long形式のまま、positive_days列を付与）
    all_metrics_subset = final_metrics_df[final_metrics_df["project"].isin(top_projects)].copy()
    all_metrics_subset = all_metrics_subset.merge(pos_df, on="project", how="left")
    all_metrics_subset = all_metrics_subset.sort_values(["positive_days", "model", "Metric"], ascending=[False, True, True])

    csv_all = out_dir / f"exp{exp_num}_top{len(top_projects)}_by_positive_days_all_metrics.csv"
    all_metrics_subset.to_csv(csv_all, index=False)
    print(f"  ✅ Saved positives-based top list (all metrics): {csv_all.name}")

    # 特定メトリクスに絞った出力 + プロット
    if metric_name:
        met_mask = all_metrics_subset["Metric"].str.lower() == str(metric_name).lower()
        met_df = all_metrics_subset.loc[met_mask].copy()
        csv_metric = out_dir / f"exp{exp_num}_top{len(top_projects)}_by_positive_days_{metric_name}.csv"
        met_df.to_csv(csv_metric, index=False)
        print(f"  ✅ Saved positives-based top list for metric '{metric_name}': {csv_metric.name}")

        if make_plots and not met_df.empty:
            for model_name, g in met_df.groupby('model'):
                if g.empty:
                    continue
                plt.figure(figsize=(12, max(4, 0.5 * len(g))))
                g_sorted = g.sort_values(["positive_days", "Value"], ascending=[False, False])
                labels = [f"{proj} ({pdays})" for proj, pdays in zip(g_sorted['project'], g_sorted['positive_days'])]
                plt.barh(labels, g_sorted['Value'], color=COLOR_PALETTE.get(model_name, '#888888'))
                plt.gca().invert_yaxis()
                plt.xlabel(metric_name)
                plt.ylabel('Project (positive days)')
                plt.title(f"Top {len(g_sorted)} by positives — {model_name} — {metric_name} (Exp {exp_num})")
                plt.tight_layout()
                fig_path = out_dir / f"exp{exp_num}_top{len(top_projects)}_by_positive_days_{metric_name}_{model_name}.png"
                plt.savefig(fig_path, dpi=150)
                print(f"  📈 Saved plot: {fig_path.name}")
                plt.close()

## ------------------------------------------------------
## メイン処理 (可視化関数の呼び出し先を変更)
## ------------------------------------------------------
def main():
    """
    メインの実行関数
    """
    print("--- Starting Experiment Analysis ---")

    # 引数
    parser = argparse.ArgumentParser(description='Compare model results and optionally export top-N performance.')
    parser.add_argument('--top-n', type=int, default=0, help='上位N件の性能を出力（0で無効）')
    parser.add_argument('--top-metric', type=str, default='MCC', help='上位抽出に用いるメトリクス名（例: MCC, AUC_ROC, F1-Score など）')
    parser.add_argument('--plot-top', action='store_true', help='上位N件の棒グラフも保存する')
    parser.add_argument('--top-by-positives', type=int, default=0, help='陽性日数(is_vcc=True)の上位N件プロジェクトの性能を出力（0で無効）')
    parser.add_argument('--positives-metric', type=str, default='MCC', help='陽性上位プロジェクトで併せて出力する特定メトリクス名（例: MCC）')
    parser.add_argument('--plot-positives-top', action='store_true', help='陽性上位N件の棒グラフも保存する')
    args = parser.parse_args()
    
    output_summary_dir = REPO_ROOT / "datasets" / "derived_artifacts" / "rq1_rq2" / "evaluation_summary_comparison"
    output_summary_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Aggregated results will be saved to: {output_summary_dir.resolve()}")

    # 1. プロジェクトと実験番号を自動検出 (★修正箇所)
    projects_per_model = {}
    all_exp_numbers = set()
    valid_models = []

    for model_name, base_dir in BASE_DIRS.items():
        if not base_dir.exists() or not base_dir.is_dir():
            print(f"⚠️ Warning: Directory for model '{model_name}' not found. Skipping: {base_dir}")
            continue
        
        print(f"🔍 Searching in '{model_name}' directory: {base_dir}")
        projects_in_dir = set()
        for metrics_path in base_dir.rglob("exp*_metrics.json"):
            try:
                rel_project = metrics_path.parent.relative_to(base_dir).as_posix()
            except ValueError:
                continue
            projects_in_dir.add(rel_project)
            match = re.search(r'exp(\d+)_metrics.json', metrics_path.name)
            if match:
                all_exp_numbers.add(int(match.group(1)))
        if not projects_in_dir:
            print(f"⚠️ Warning: No experiment directories found for '{model_name}'.")
            continue
        valid_models.append(model_name)
        projects_per_model[model_name] = projects_in_dir

    # 全てのモデルに共通するプロジェクトのリスト（積集合）を作成
    if not projects_per_model:
        print("❌ Error: No valid model directories found.")
        return
        
    initial_common_projects = set.intersection(*projects_per_model.values())

    if not initial_common_projects:
        print(f"❌ Error: No common project directories found across all specified models: {valid_models}")
        return
    if not all_exp_numbers:
        print("❌ Error: No experiment files found.")
        return

    sorted_initial_projects = sorted(list(initial_common_projects))
    sorted_exp_numbers = sorted(list(all_exp_numbers))
    print(f"\n✅ Found {len(sorted_initial_projects)} common project directories: {sorted_initial_projects}")
    print(f"✅ Found unique experiments: {sorted_exp_numbers}")

    # ▼ 追加: 全モデル×全実験で共通のプロジェクト集合を事前に算出
    def effective_exp(model_name: str, exp: int) -> int:
        return 0 if model_name == "Random" else exp

    per_exp_common_projects = {}
    for exp_num in sorted_exp_numbers:
        projects_with_data_per_model = {}
        for model_name, base_dir in BASE_DIRS.items():
            if model_name not in valid_models:
                continue
            eff_exp = effective_exp(model_name, exp_num)
            projects_found = set()
            for project in sorted_initial_projects:
                _, met_dict = load_experiment_data(base_dir, project, eff_exp)
                if met_dict is not None:
                    projects_found.add(project)
            projects_with_data_per_model[model_name] = projects_found

        if projects_with_data_per_model:
            per_exp_common_projects[exp_num] = set.intersection(*projects_with_data_per_model.values())
        else:
            per_exp_common_projects[exp_num] = set()

    global_common_projects = set()
    non_empty_sets = [s for s in per_exp_common_projects.values() if len(s) > 0]
    if non_empty_sets:
        global_common_projects = set.intersection(*non_empty_sets)

    use_global_common = len(global_common_projects) > 0
    if use_global_common:
        sorted_global_common = sorted(list(global_common_projects))
        print(f"✅ Using a fixed common project set across all experiments: {len(sorted_global_common)} projects")
    else:
        print("⚠️ No global intersection across all models and experiments. Falling back to per-experiment intersections.")

    # 2. 実験番号ごとにデータを集約・可視化
    for exp_num in sorted_exp_numbers:
        print(f"\n{'='*25} Processing Experiment {exp_num} {'='*25}")

        # ▼ 変更: 全expで同一集合を使用（フォールバック時のみ従来処理）
        if use_global_common:
            sorted_common_projects_for_exp = sorted_global_common
            print(f"  -> Using fixed {len(sorted_common_projects_for_exp)} projects for all experiments.")
        else:
            # 共通プロジェクトは「metrics があること」のみで判定（importances は不要）
            projects_with_data_per_model = {}
            for model_name, base_dir in BASE_DIRS.items():
                if model_name not in valid_models:
                    continue
                projects_found = set()
                eff_exp = effective_exp(model_name, exp_num)
                for project in sorted_initial_projects:
                    imp_df, met_dict = load_experiment_data(base_dir, project, eff_exp)
                    if met_dict is not None:
                        projects_found.add(project)
                projects_with_data_per_model[model_name] = projects_found

            final_common_projects = set.intersection(*projects_with_data_per_model.values())

            random_only_exp = False
            if not final_common_projects:
                if exp_num == 0 and "Random" in projects_with_data_per_model and projects_with_data_per_model["Random"]:
                    sorted_common_projects_for_exp = sorted(list(projects_with_data_per_model["Random"]))
                    random_only_exp = True
                    print(f"  -> Exp {exp_num} has only 'Random'. Using {len(sorted_common_projects_for_exp)} projects.")
                else:
                    print(f"  - Skipping Exp {exp_num}: No projects found with data for all models.")
                    continue
            else:
                sorted_common_projects_for_exp = sorted(list(final_common_projects))
                print(f"  -> Analyzing {len(sorted_common_projects_for_exp)} projects for this experiment: {sorted_common_projects_for_exp}")

        all_models_importances_list = []
        all_models_metrics_list = []
        
        for model_name, base_dir in BASE_DIRS.items():
            if not base_dir.exists():
                continue
            if not use_global_common:
                # フォールバック時の random_only_exp は維持
                if 'random_only_exp' in locals() and random_only_exp and model_name != "Random":
                    continue

            importances_per_model, metrics_per_model = [], []
            projects_with_data_count = 0

            eff_exp = effective_exp(model_name, exp_num)

            for project in sorted_common_projects_for_exp:
                imp_df, met_dict = load_experiment_data(base_dir, project, eff_exp)

                if met_dict is not None:
                    projects_with_data_count += 1
                    main_metrics = {
                        'MCC': met_dict.get('mcc'), 'AUC_ROC': met_dict.get('auc_roc'),
                        'Accuracy': met_dict.get('accuracy'),
                        'Precision': met_dict.get('classification_report_dict', {}).get('class_1', {}).get('precision'),
                        'Recall': met_dict.get('classification_report_dict', {}).get('class_1', {}).get('recall'),
                        'F1-Score': met_dict.get('classification_report_dict', {}).get('class_1', {}).get('f1-score'),
                        'PR_AUC': met_dict.get('classification_report_dict', {}).get('class_1', {}).get('pr_auc')
                    }
                    temp_df = pd.DataFrame(list(main_metrics.items()), columns=['Metric', 'Value']).dropna()
                    temp_df['project'] = project
                    metrics_per_model.append(temp_df)

                if imp_df is not None:
                    imp_df['project'] = project
                    importances_per_model.append(imp_df)
            
            if importances_per_model or metrics_per_model:
                print(f"  - Found data for '{model_name}' in {projects_with_data_count} projects.")
                if importances_per_model:
                    model_importances_df = pd.concat(importances_per_model, ignore_index=True)
                    model_importances_df['model'] = model_name
                    all_models_importances_list.append(model_importances_df)

                if metrics_per_model:
                    model_metrics_df = pd.concat(metrics_per_model, ignore_index=True)
                    model_metrics_df['model'] = model_name
                    all_models_metrics_list.append(model_metrics_df)

        if not all_models_metrics_list:
            print(f"  - No valid data found for any model in experiment {exp_num}. Skipping.")
            continue
        
        final_metrics_df = pd.concat(all_models_metrics_list, ignore_index=True)
        final_importances_df = pd.concat(all_models_importances_list, ignore_index=True) if all_models_importances_list else pd.DataFrame()
        
        metrics_csv_path = output_summary_dir / f"exp{exp_num}_all_models_metrics_comparison.csv"
        importances_csv_path = output_summary_dir / f"exp{exp_num}_all_models_importances_comparison.csv"
        final_metrics_df.to_csv(metrics_csv_path, index=False)
        if not final_importances_df.empty:
            final_importances_df.to_csv(importances_csv_path, index=False)
        print(f"  - Saved aggregated metrics to: {metrics_csv_path.name}")
        if not final_importances_df.empty:
            print(f"  - Saved aggregated importances to: {importances_csv_path.name}")

        # ▼ 追加: モデルごとに平均を算出しつつ、集計に使われたプロジェクト数も付与
        mean_metrics_df = final_metrics_df.groupby(['model', 'Metric'])['Value'].mean().sort_values(ascending=False).reset_index()
        mean_metrics_df.rename(columns={'Value': 'Mean_Value'}, inplace=True)
        # モデルごとの集計対象プロジェクト数（ユニーク）
        project_counts = final_metrics_df.groupby('model')['project'].nunique().reset_index().rename(columns={'project': 'Project_Count'})
        mean_metrics_df = mean_metrics_df.merge(project_counts, on='model', how='left')

        mean_metrics_csv_path = output_summary_dir / f"exp{exp_num}_mean_metrics_by_model.csv"
        mean_metrics_df.to_csv(mean_metrics_csv_path, index=False, float_format='%.6f')
        print(f"  - Saved mean metrics by model to: {mean_metrics_csv_path.name} (with Project_Count)")

        # 3. 上位N件の性能を出力（オプション）
        export_top_n_performance(final_metrics_df, exp_num, output_summary_dir, args.top_n, args.top_metric, args.plot_top)

        # 3b. 陽性日数上位N件の性能を出力（オプション）
        export_top_by_positive_days(
            final_metrics_df,
            exp_num,
            output_summary_dir,
            sorted_common_projects_for_exp,
            valid_models,
            args.top_by_positives,
            args.positives_metric,
            args.plot_positives_top,
        )

        # 4. 可視化（特徴量重要度はモデルごとに、Random はスキップされる場合あり）
        num_projects_in_exp = final_metrics_df['project'].nunique()
        visualize_per_model_importance(
            final_metrics_df,
            final_importances_df if not final_importances_df.empty else pd.DataFrame(columns=['model','feature','importance']),
            exp_num, num_projects_in_exp
        )

    print("\n🎉 Analysis complete!")

if __name__ == '__main__':
    main()
