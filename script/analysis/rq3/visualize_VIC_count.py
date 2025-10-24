import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 設定 ---
csv_path = "/work/riku-ka/vuljit/rq3_dataset/oss_fuzz_vulns_2025802.csv"          # 入力CSV
date_col = "published"         # 期間判定に使う日付カラム（"published" や "modified" に変更可）
project_col = "package_name"   # 「プロジェクト」と見なすカラム（"repo" にすることも可）
start = "2018-10-12"
end = "2025-06-01"
top_n = None                     # プロジェクト数が多い場合は上位 top_n のみ表示（None にすると全て表示）

# --- 読み込み ---
df = pd.read_csv(csv_path, parse_dates=[date_col], keep_default_na=False)

# --- 追加: introduced_commits の重複除外 ---
# 空文字を NA に変換して扱いやすくする
df['introduced_commits'] = df.get('introduced_commits', pd.Series(dtype='object')).astype(str).str.strip()
df.loc[df['introduced_commits'] == '', 'introduced_commits'] = pd.NA
# introduced_commits が存在する行について、重複するコミットハッシュは最初だけ残す（以降は削除）
dup_mask = df['introduced_commits'].notna() & df['introduced_commits'].duplicated(keep='first')
if dup_mask.any():
    df = df[~dup_mask]

# 日付が空の行は除外（必要に応じて別処理）
df = df[df[date_col].notna()]

# 期間でフィルタ
start_dt = pd.to_datetime(start)
end_dt = pd.to_datetime(end)

# date_col を確実に datetime 型に（失敗したものは NaT）
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# シリーズのタイムゾーンを UTC に統一（tz-aware なら変換、そうでなければローカライズ）
if df[date_col].dt.tz is None:
    df[date_col] = df[date_col].dt.tz_localize('UTC')
else:
    df[date_col] = df[date_col].dt.tz_convert('UTC')

# start/end を UTC-aware にする（既に tz-aware なら変換）
if getattr(start_dt, 'tzinfo', None) is None:
    start_dt = start_dt.tz_localize('UTC')
else:
    start_dt = start_dt.tz_convert('UTC')

if getattr(end_dt, 'tzinfo', None) is None:
    end_dt = end_dt.tz_localize('UTC')
else:
    end_dt = end_dt.tz_convert('UTC')

mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
df_period = df.loc[mask]

# プロジェクトごとにカウント
counts = df_period[project_col].value_counts()

# 必要なら上位のみ
if top_n is not None:
    top_counts = counts.head(top_n)
    others_count = counts.iloc[top_n:].sum()
    if others_count > 0:
        top_counts["(others)"] = others_count
    plot_counts = top_counts
else:
    plot_counts = counts

# --- プロット ---
sns.set(style="whitegrid", context="talk")

# データが空なら何もしない
if plot_counts.empty:
    print("No data to plot for the selected period.")
else:
    # 降順（多い順）にして順位を付ける
    plot_counts = plot_counts.sort_values(ascending=False)
    n = len(plot_counts)

    # 動的な横幅（要素数に応じて）
    fig_width = max(10, min(24, 0.6 * n))
    plt.figure(figsize=(fig_width, 6))

    # 棒グラフ（縦）。x軸には順位ラベルのみ表示してプロジェクト名は隠す
    ax = plot_counts.plot(kind="bar", color=sns.color_palette("tab10", n_colors=min(10, n)))
    plt.title(f"Project vuln counts ({start} — {end}) — names hidden, shown as ranks")
    plt.xlabel("Rank")
    plt.ylabel("Count")

    # x軸ラベルを 1,2,... に置き換え（左から多い順）
    ranks = [str(i + 1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(ranks, rotation=0)

    # 棒の上に数値ラベル（強調）
    # for p in ax.patches:
    #     height = int(p.get_height())
    #     ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2, height),
    #                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig("project_histogram.png", dpi=150)
    plt.show()