import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# グラフの日本語表示設定 (必要に応じて)
# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic']


## ------------------------------------------------------
## 設定
## ------------------------------------------------------
# 前のスクリプトで生成された集約結果が格納されているディレクトリを指定
BASE_PATH = Path("./evaluation_summary_comparison")  # 変更
COLOR_PALETTE = {
    "XGBoost": "#1f77b4",       # ブルー
    "RandomForest": "#ff7f0e",  # オレンジ
    "Random": "#2ca02c"         # 緑
}
# 比較対象の実験名（exp0〜exp5）
EXPERIMENTS = [f"exp{i}" for i in range(0, 6)]  # 変更
# exp0..exp5 をプロット上で表示するラベル (順序を保つ)
EXPERIMENT_LABELS = [
    "Random",
    "Kamei",
    "Kamei+Coverage",
    "VCCFINDER",
    "VCCFINDER+Coverage",
    "Coverage",
]

HUE_ORDER = ["Random", "XGBoost", "RandomForest"]


## ------------------------------------------------------
## メイン処理 (改良版)
## ------------------------------------------------------
def _fixed_ylim_for_metric(metric_name: str):
    """Return a fixed y-axis range for the given metric to avoid auto-scaling.

    - MCC: [-1, 1]
    - Others (e.g., AUC, F1, Precision, Recall, Accuracy, G-Mean): [0, 1]
    """
    name = (metric_name or "").strip().upper()
    if "MCC" in name:
        return (-1.0, 1.0)
    # Default: many common metrics are in [0, 1]
    return (0.0, 1.0)

def _collect_projects_from_df(df: pd.DataFrame):
    """Collect project-like identifiers from a DataFrame.

    Heuristically looks for columns commonly used to denote a project/repo.
    Returns a set of unique project names (strings).
    """
    if df is None or df.empty:
        return set()
    candidates = []
    for c in df.columns:
        lc = str(c).lower()
        if lc in {"project", "projects", "project_name", "repo", "repository"}:
            candidates.append(c)
    found = set()
    for c in candidates:
        try:
            for v in df[c].dropna().tolist():
                s = str(v).strip()
                if s:
                    found.add(s)
        except Exception:
            continue
    return found

def main():
    """
    メインの実行関数
    """
    print(f"--- Analyzing trends from: {BASE_PATH} ---")

    # CLI options to drive top-N visualizations
    parser = argparse.ArgumentParser(description='Visualize trends and optionally plot top-N metrics and positives-based top-N.')
    parser.add_argument('--top-n', type=int, default=0, help='各実験・各モデルの上位N件（特定メトリクス）を可視化（0で無効）')
    parser.add_argument('--top-metric', type=str, default='MCC', help='上位N件の対象メトリクス名（例: MCC, AUC_ROC など）')
    parser.add_argument('--top-by-positives', type=int, default=0, help='陽性日数(is_vcc=True)の上位N件（特定メトリクス）を可視化（0で無効）')
    parser.add_argument('--positives-metric', type=str, default='MCC', help='陽性上位の対象メトリクス名（例: MCC）')
    args = parser.parse_args()
    
    # データを格納するための空のリストを作成
    all_data = []

    # 各実験のCSVファイルを読み込み、リストに追加
    used_projects = set()
    for exp_name in EXPERIMENTS:
        # 読み込むファイル名をモデル別平均が記録されたCSVに変更
        file_path = BASE_PATH / f"{exp_name}_mean_metrics_by_model.csv"

        if file_path.exists():
            df = pd.read_csv(file_path)
            # プロジェクト名が含まれていれば収集
            used_projects |= _collect_projects_from_df(df)
            # expN -> フレンドリ名にマップして実験ラベル列を追加
            try:
                idx = int(exp_name.replace('exp', ''))
                exp_label = EXPERIMENT_LABELS[idx]
            except Exception:
                exp_label = exp_name
            df['Experiment'] = exp_label  # 実験名列を追加（マップ済みラベル）
            all_data.append(df)
        else:
            print(f"⚠️ Warning: File not found, skipping: {file_path}")

    if not all_data:
        print("❌ Error: No data files found. Please check BASE_PATH and file names.")
        return

    # 全てのデータフレームを一つに結合
    combined_df = pd.concat(all_data, ignore_index=True)

    # Experiment列をカテゴリ型に変換し、順序を設定（マップ済みラベル順）
    combined_df['Experiment'] = pd.Categorical(combined_df['Experiment'], categories=EXPERIMENT_LABELS, ordered=True)

    # プロットするメトリクスのリストを取得
    metrics_to_plot = combined_df['Metric'].unique()
    print(f"✅ Found metrics to plot: {list(metrics_to_plot)}")

    # --- グラフ作成部分 (seabornでモデル比較グラフに改良) ---
    for metric in metrics_to_plot:
        plt.figure(figsize=(16, 8))
        
        # 対象のメトリクスに関するデータのみをフィルタリング
        metric_df = combined_df[combined_df['Metric'] == metric]

        if metric_df.empty:
            print(f"  - No data for metric '{metric}', skipping plot.")
            continue

        # モデルの表示順序を固定
        metric_df['model'] = pd.Categorical(metric_df['model'], categories=HUE_ORDER, ordered=True)

        ax = sns.barplot(data=metric_df, x='Experiment', y='Mean_Value', hue='model',
                         palette=COLOR_PALETTE, hue_order=HUE_ORDER)
        
        # グラフのタイトルとラベルを設定
        ax.set_title(f'Comparison of Average {metric} Across Experiments', fontsize=18, weight='bold')
        ax.set_xlabel('Experiment', fontsize=14)
        ax.set_ylabel(f'Average Score (Mean {metric})', fontsize=14)

        # X軸のラベルを回転
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        # Y軸の範囲を自動調整し、少し余裕を持たせる
        if not metric_df.empty:
            min_val = metric_df['Mean_Value'].min()
            max_val = metric_df['Mean_Value'].max()
            # ▼ 修正: 負値がある場合は下限をより負側に拡張してクリップを避ける
            if pd.notna(min_val) and min_val < 0:
                y_min = min(min_val * 1.1, -1.0)  # 余白確保しつつ -1.0 以下にしない
            else:
                y_min = 0
            y_max = max(1.0, max_val * 1.1 if pd.notna(max_val) else 1.0)
            ax.set_ylim(y_min, y_max)

        # 棒の上に数値を表示
        for p in ax.patches:
            height = p.get_height()
            if pd.notna(height):
                label = f'{height:.4f}'
                # ▼ 変更: 正負で注記位置を切り替え（負値でも表示）
                if height >= 0:
                    ax.annotate(label,
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom',
                                xytext=(0, 9), textcoords='offset points',
                                fontsize=9, color='black')
                else:
                    ax.annotate(label,
                                (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='top',
                                xytext=(0, -9), textcoords='offset points',
                                fontsize=9, color='black')
        
        ax.legend(title='Model', fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # グラフを画像ファイルとして保存
        output_filename = f'comparison_trend_{metric}.png'
        plt.savefig(output_filename)
        print(f"  📈 Graph saved as '{output_filename}'")
        
        plt.close()

    # -----------------------------
    # Optional: Top-N (by value) aggregated like normal plots, for ALL metrics
    # -----------------------------
    if args.top_n and args.top_n > 0:
        print(f"\n--- Building top-{args.top_n} aggregated plots for ALL metrics ---")
        topn_rows = []
        for i, exp_name in enumerate(EXPERIMENTS):
            all_metrics_path = BASE_PATH / f"{exp_name}_all_models_metrics_comparison.csv"
            if not all_metrics_path.exists():
                print(f"⚠️ Skipping {exp_name}: missing {all_metrics_path.name}")
                continue
            try:
                df_all = pd.read_csv(all_metrics_path)
            except Exception as e:
                print(f"⚠️ Failed to read {all_metrics_path}: {e}")
                continue
            # プロジェクト名が含まれていれば収集
            used_projects |= _collect_projects_from_df(df_all)
            # Determine experiment label
            try:
                idx = int(exp_name.replace('exp', ''))
                exp_label = EXPERIMENT_LABELS[idx]
            except Exception:
                exp_label = exp_name

            metrics_here = sorted(df_all['Metric'].dropna().unique().tolist())
            for metric in metrics_here:
                df_m = df_all[df_all['Metric'] == metric]
                # Per model: sort by Value desc, take top-N, compute mean
                for model in HUE_ORDER:
                    g = df_m[df_m['model'] == model]
                    if g.empty:
                        continue
                    g_top = g.sort_values('Value', ascending=False).head(args.top_n)
                    mean_top = g_top['Value'].mean()
                    topn_rows.append({
                        'Experiment': exp_label,
                        'model': model,
                        'Metric': metric,
                        'Mean_TopN': mean_top
                    })

        if topn_rows:
            topn_df = pd.DataFrame(topn_rows)
            # Plot for each metric similarly to the main section
            for metric in topn_df['Metric'].unique():
                plt.figure(figsize=(16, 8))
                sub = topn_df[topn_df['Metric'] == metric].copy()
                sub['Experiment'] = pd.Categorical(sub['Experiment'], categories=EXPERIMENT_LABELS, ordered=True)
                sub['model'] = pd.Categorical(sub['model'], categories=HUE_ORDER, ordered=True)
                ax = sns.barplot(data=sub, x='Experiment', y='Mean_TopN', hue='model',
                                 palette=COLOR_PALETTE, hue_order=HUE_ORDER)
                ax.set_title(f'Comparison of Average {metric} (Top-{args.top_n}) Across Experiments', fontsize=18, weight='bold')
                ax.set_xlabel('Experiment', fontsize=14)
                ax.set_ylabel(f'Average Score (Mean {metric}, Top-{args.top_n})', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                # Annotate values on bars
                for p in ax.patches:
                    height = p.get_height()
                    if pd.notna(height):
                        label = f'{height:.4f}'
                        if height >= 0:
                            ax.annotate(label,
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 9), textcoords='offset points',
                                        fontsize=9, color='black')
                        else:
                            ax.annotate(label,
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='top',
                                        xytext=(0, -9), textcoords='offset points',
                                        fontsize=9, color='black')
                ax.legend(title='Model', fontsize=11)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                out = f"trend_top{args.top_n}_avg_{metric}.png"
                plt.savefig(out, dpi=150)
                print(f"  📈 Saved: {out}")
                plt.close()

    # -----------------------------
    # Optional: Top-N by positives — aggregated like normal plots, for ALL metrics
    # -----------------------------
    if args.top_by_positives and args.top_by_positives > 0:
        print(f"\n--- Building positives-based top-{args.top_by_positives} aggregated plots for ALL metrics ---")
        pos_rows = []
        for i, exp_name in enumerate(EXPERIMENTS):
            all_pos_path = BASE_PATH / f"{exp_name}_top{args.top_by_positives}_by_positive_days_all_metrics.csv"
            if not all_pos_path.exists():
                print(f"⚠️ Skipping {exp_name}: missing {all_pos_path.name}")
                continue
            try:
                dfp = pd.read_csv(all_pos_path)
            except Exception as e:
                print(f"⚠️ Failed to read {all_pos_path}: {e}")
                continue
            # プロジェクト名が含まれていれば収集
            used_projects |= _collect_projects_from_df(dfp)
            # Determine experiment label
            try:
                idx = int(exp_name.replace('exp', ''))
                exp_label = EXPERIMENT_LABELS[idx]
            except Exception:
                exp_label = exp_name
            metrics_here = sorted(dfp['Metric'].dropna().unique().tolist())
            for metric in metrics_here:
                mdf = dfp[dfp['Metric'] == metric]
                for model in HUE_ORDER:
                    g = mdf[mdf['model'] == model]
                    if g.empty:
                        continue
                    mean_val = g['Value'].mean()
                    pos_rows.append({
                        'Experiment': exp_label,
                        'model': model,
                        'Metric': metric,
                        'Mean_TopPos': mean_val
                    })

        if pos_rows:
            posdf = pd.DataFrame(pos_rows)
            for metric in posdf['Metric'].unique():
                plt.figure(figsize=(16, 8))
                sub = posdf[posdf['Metric'] == metric].copy()
                sub['Experiment'] = pd.Categorical(sub['Experiment'], categories=EXPERIMENT_LABELS, ordered=True)
                sub['model'] = pd.Categorical(sub['model'], categories=HUE_ORDER, ordered=True)
                ax = sns.barplot(data=sub, x='Experiment', y='Mean_TopPos', hue='model',
                                 palette=COLOR_PALETTE, hue_order=HUE_ORDER)
                ax.set_title(f'Comparison of Average {metric} (Top-{args.top_by_positives} by positives) Across Experiments', fontsize=18, weight='bold')
                ax.set_xlabel('Experiment', fontsize=14)
                ax.set_ylabel(f'Average Score (Mean {metric}, Top-{args.top_by_positives})', fontsize=14)
                # 固定の縦軸（メトリクスに応じた既定範囲）で表示し、恣意的な自動調整を避ける
                ymin, ymax = _fixed_ylim_for_metric(metric)
                ax.set_ylim(ymin, ymax)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                for p in ax.patches:
                    height = p.get_height()
                    if pd.notna(height):
                        label = f'{height:.4f}'
                        if height >= 0:
                            ax.annotate(label,
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='bottom',
                                        xytext=(0, 9), textcoords='offset points',
                                        fontsize=9, color='black')
                        else:
                            ax.annotate(label,
                                        (p.get_x() + p.get_width() / 2., height),
                                        ha='center', va='top',
                                        xytext=(0, -9), textcoords='offset points',
                                        fontsize=9, color='black')
                ax.legend(title='Model', fontsize=11)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                out = f"trend_top{args.top_by_positives}_bypositives_avg_{metric}.png"
                plt.savefig(out, dpi=150)
                print(f"  📈 Saved: {out}")
                plt.close()

    # 解析で用いたプロジェクトの一覧を標準出力に表示
    if used_projects:
        print("\nUsed projects (count: {}):".format(len(used_projects)))
        for p in sorted(used_projects):
            print(f" - {p}")
    else:
        print("\nUsed projects: (no project column found in inputs)")

    print("\n🎉 Analysis complete!")

if __name__ == '__main__':
    main()
