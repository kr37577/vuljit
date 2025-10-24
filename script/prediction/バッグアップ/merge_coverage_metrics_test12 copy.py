import os
import sys
import pandas as pd
import numpy as np
import ast
from bisect import bisect_left, bisect_right
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # 進捗表示のためtqdmをインポート
import argparse
# settings.pyから特徴量リストをインポート（ローカル settings に変更）
from settings import KAMEI_FEATURES, VCCFINDER_FEATURES, ALL_PERCENT_FEATURES, ALL_CALCULATION_FEATURES
from datetime import timedelta

# --- ユーティリティ ---

def _parse_changed_files(cell):
    """文字列のリスト表現からPythonリストへ。失敗時は[]。"""
    if pd.isna(cell):
        return []
    try:
        v = ast.literal_eval(cell)
        return v if isinstance(v, list) else []
    except Exception:
        return []

def _norm_path(p):
    """必要なら正規化を追加。ここではトリムのみ。"""
    return p.strip()


# --- ▼▼▼【変更なし】パッチカバレッジ読み込み用の関数 ▼▼▼ ---
def load_patch_coverage_data(patch_coverage_file):
    """
    プロジェクトごとのパッチカバレッジCSVファイルを読み込む。
    """
    try:
        print(f"  [DEBUG] パッチカバレッジファイル読み込み: {patch_coverage_file}")
        df = pd.read_csv(patch_coverage_file)
        print(f"    -> 読み込み成功: {len(df)} 行")
        
        # 'date'列をdatetimeオブジェクトに変換し、'merge_date'として統一
        df['merge_date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.date
        df.dropna(subset=['merge_date'], inplace=True)
        return df
    except FileNotFoundError:
        print(f"  [INFO] パッチカバレッジファイルが見つかりませんでした: {patch_coverage_file}")
        return None
    except Exception as e:
        print(f"  エラー: パッチカバレッジファイルの読み込み中にエラー: {e}", file=sys.stderr)
        return None

# --- ▼▼▼【変更なし】データ読み込みと前処理用の関数 ▼▼▼ ---
def load_and_prepare_data(metrics_file, coverage_file_project, coverage_file_project_total):
    """
    メトリクス、ProjectベースカバレッジのCSVファイルを読み込み、前処理を行う。
    """
    try:
        print(f"  [DEBUG] メトリクスファイル読み込み: {metrics_file}")
        metrics_df = pd.read_csv(metrics_file)
        print(f"    -> 読み込み成功: {len(metrics_df)} 行")

        # Projectベースのカバレッジファイル
        print(f"  [DEBUG] Projectカバレッジファイル読み込み: {coverage_file_project}")
        coverage_project_df = pd.read_csv(coverage_file_project)
        print(f"    -> 読み込み成功: {len(coverage_project_df)} 行")

        print(f"  [DEBUG] Projectカバレッジファイル（合計）読み込み: {coverage_file_project_total}")
        coverage_project_total_df = pd.read_csv(coverage_file_project_total)
        print(f"    -> 読み込み成功: {len(coverage_project_total_df)} 行")

        # commit_hash で一意にフィルタリング（merge_date の算出前に実施）
        if 'commit_hash' in metrics_df.columns:
            before_len = len(metrics_df)
            metrics_df = metrics_df.drop_duplicates(subset=['commit_hash'], keep='first')
            print(f"    -> commit_hash 重複排除: {before_len} -> {len(metrics_df)} 行")
        else:
            print("  警告: metrics_df に 'commit_hash' 列がありません。重複排除をスキップします。", file=sys.stderr)

        # commit_datetime をUTCのISO8601文字列に正規化
        if 'commit_datetime' in metrics_df.columns:
            dt_parsed = pd.to_datetime(
                metrics_df['commit_datetime'].astype(str).str.replace(' ', 'T', regex=False),
                errors='coerce', utc=True
            )
            # 例: 2014-01-10T15:55:12+00:00 の形式にする
            metrics_df['commit_datetime'] = (
                dt_parsed.dt.strftime('%Y-%m-%dT%H:%M:%S%z')
                .str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
            )

        # 日付/時刻データをdatetimeオブジェクトに変換（UTC基準）
        metrics_df['merge_date'] = pd.to_datetime(
            metrics_df['commit_datetime'].str.replace(' ', 'T', regex=False),
            errors='coerce', utc=True
        ).dt.date
        coverage_project_df['merge_date'] = pd.to_datetime(coverage_project_df['date'], format='%Y%m%d', errors='coerce').dt.date
        coverage_project_total_df['merge_date'] = pd.to_datetime(coverage_project_total_df['date'], format='%Y%m%d', errors='coerce').dt.date
        
        # 日付変換に失敗した行を削除
        metrics_df.dropna(subset=['merge_date'], inplace=True)
        coverage_project_df.dropna(subset=['merge_date'], inplace=True)
        coverage_project_total_df.dropna(subset=['merge_date'], inplace=True)

        if 'filename' not in coverage_project_df.columns:
            print(f"エラー: Projectカバレッジファイルに 'filename' 列がありません。", file=sys.stderr)
            return None, None, None
        
        # --- 列名の統一 (Projectベース) ---
        coverage_project_df.rename(columns={
            'functions_covered': 'function_covered', 'functions_count': 'function_count',
            'lines_covered': 'line_covered', 'lines_count': 'line_count',
            'regions_covered': 'region_covered', 'regions_count': 'region_count',
            'branches_covered': 'branch_covered', 'branches_count': 'branch_count',
            'instantiations_covered': 'instantiation_covered', 'instantiations_count': 'instantiation_count'
        }, inplace=True)

        coverage_project_total_df.columns = [col.replace('totals_', '') for col in coverage_project_total_df.columns]
        coverage_project_total_df.columns = [col.replace('_notcovered', '') for col in coverage_project_total_df.columns]
        coverage_project_total_df.rename(columns={
            'functions_covered': 'function_covered', 'functions_count': 'function_count','functions_percent': 'function_percent',
            'lines_covered': 'line_covered', 'lines_count': 'line_count','lines_percent': 'line_percent',
            'regions_covered': 'region_covered', 'regions_count': 'region_count','regions_percent': 'region_percent',
            'branches_covered': 'branch_covered', 'branches_count': 'branch_count','branches_percent': 'branch_percent',
            'instantiations_covered': 'instantiation_covered', 'instantiations_count': 'instantiation_count', 'instantiations_percent': 'instantiation_percent'
            }, inplace=True)

        return metrics_df, coverage_project_df, coverage_project_total_df

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません: {e}", file=sys.stderr)
        return None, None, None
    except Exception as e:
        print(f"エラー: ファイル読み込みまたは前処理中にエラーが発生しました: {e}", file=sys.stderr)
        return None, None, None

# --- ▼▼▼【変更なし】コミットごとのカバレッジ計算関数 ▼▼▼ ---
def calculate_commit_coverage(metrics_df, coverage_project_df, coverage_project_total_df, patch_coverage_df):
    """
    データフレームを受け取り、コミットごとのカバレッジ【絶対値】を計算して結果を返す。
    """
    results = []
    coverage_types = ['function', 'line', 'region', 'branch', 'instantiation']
    
    available_project_dates = sorted(coverage_project_df['merge_date'].unique())
    
    metrics_df.sort_values(by='merge_date', inplace=True)

    print(f"  -> {len(metrics_df)}件のコミットのカバレッジ絶対値を計算中...")
    for _, commit_row in tqdm(metrics_df.iterrows(), total=metrics_df.shape[0], desc="  コミット処理"):
        commit_hash = commit_row['commit_hash']
        commit_date = commit_row['merge_date']
        commit_result = commit_row.to_dict()

        try:
            changed_files_str = commit_row['commit_change_file_path_filetered']
            changed_files = [] if pd.isna(changed_files_str) else ast.literal_eval(changed_files_str)
        except (ValueError, SyntaxError):
            print(f"  警告: commit {commit_hash} のファイルパスを解析できませんでした。スキップします。", file=sys.stderr)
            changed_files = []

        # === Projectベースのカバレッジ計算 ===
        # 前日が存在する時だけ採用
        prev_date = commit_date - timedelta(days=1)
        if prev_date in set(available_project_dates):  # setでO(1)照合。listのままでも可
            target_date = prev_date
        else:
            continue 
         
        daily_coverage_df = coverage_project_df[coverage_project_df['merge_date'] == target_date]
        daily_coverage_total_df = coverage_project_total_df[coverage_project_total_df['merge_date'] == target_date]

        matching_indices = []
        if changed_files and not daily_coverage_df.empty:
            coverage_filenames = daily_coverage_df['filename']
            for path_suffix in changed_files:
                path_components = path_suffix.split('/')
                candidate_indices = coverage_filenames.index.tolist()
                for i in range(1, len(path_components) + 1):
                    sub_path_to_match = '/'.join(path_components[-i:])
                    current_coverage_subset = coverage_filenames.loc[candidate_indices]
                    new_candidate_indices = current_coverage_subset[current_coverage_subset.str.endswith(sub_path_to_match, na=False)].index.tolist()
                    if not new_candidate_indices: break
                    candidate_indices = new_candidate_indices
                    if len(candidate_indices) == 1: break
                matching_indices.extend(candidate_indices)
        relevant_coverage = daily_coverage_df.loc[list(set(matching_indices))]
        
        # project_commit_*系の特徴量を削除
        # for cov_type in coverage_types:
        #     covered_col, count_col = f'{cov_type}_covered', f'{cov_type}_count'
        #     if not relevant_coverage.empty and covered_col in relevant_coverage.columns and count_col in relevant_coverage.columns:
        #         total_covered, total_count = relevant_coverage[covered_col].sum(), relevant_coverage[count_col].sum()
        #         percent = (total_covered / total_count) * 100 if total_count > 0 else 0.0
        #     else:
        #         total_covered, total_count, percent = 0, 0, 0.0
        #     commit_result[f'project_commit_{cov_type}_covered'] = total_covered
        #     commit_result[f'project_commit_{cov_type}_count'] = total_count
        #     commit_result[f'project_commit_{cov_type}_percent'] = percent
        
        project_total_coverage_row = daily_coverage_total_df.iloc[[0]] if not daily_coverage_total_df.empty else None
        for cov_type in coverage_types:
            for metric in ['count', 'covered', 'percent']:
                col_name = f'{cov_type}_{metric}'
                value = 0.0
                if project_total_coverage_row is not None and not project_total_coverage_row.empty and col_name in project_total_coverage_row.columns:
                    value = project_total_coverage_row[col_name].iloc[0]
                commit_result[f'project_total_{cov_type}_{metric}'] = value
        
        # === パッチカバレッジの集計 ===
        if patch_coverage_df is not None and not patch_coverage_df.empty:
            daily_patch_df = patch_coverage_df[patch_coverage_df['merge_date'] == target_date]
            if not daily_patch_df.empty:
                total_added = daily_patch_df['total_added_lines'].sum()
                covered_added = daily_patch_df['covered_added_lines'].sum()
                recalculated_coverage = (covered_added / total_added) * 100 if total_added > 0 else 0.0
                
                commit_result['patch_total_added_lines'] = total_added
                commit_result['patch_covered_added_lines'] = covered_added
                commit_result['patch_coverage_recalculated'] = recalculated_coverage
            else:
                commit_result['patch_total_added_lines'] = 0
                commit_result['patch_covered_added_lines'] = 0
                commit_result['patch_coverage_recalculated'] = 0.0
        else:
            commit_result['patch_total_added_lines'] = 0
            commit_result['patch_covered_added_lines'] = 0
            commit_result['patch_coverage_recalculated'] = 0.0

        results.append(commit_result)
    
    return pd.DataFrame(results)

# --- 台帳作成 ---

def build_vcc_file_ledger(metrics_df,
                          path_col='commit_change_file_path_filetered',
                          is_vcc_col='is_vcc',
                          date_col='merge_date'):
    """
    is_vcc=1 のコミットで変更されたファイルを台帳化。
    返り値: dict[file_path] -> sorted(list[date])
    """
    df = metrics_df.copy()

    # 日付列が無ければ生成
    if date_col not in df.columns and 'commit_datetime' in df.columns:
        df[date_col] = pd.to_datetime(df['commit_datetime'].str.replace(' ', 'T', regex=False),
                                      errors='coerce', utc=True).dt.date

    df = df.dropna(subset=[date_col])

    # is_vcc フィルタ
    df = df[df[is_vcc_col] == 1]

    ledger = {}
    for _, row in df.iterrows():
        d = row[date_col]
        for p in _parse_changed_files(row.get(path_col, None)):
            q = _norm_path(p)
            if not q:
                continue
            ledger.setdefault(q, []).append(d)

    # ソートと重複除去
    for k in list(ledger.keys()):
        ledger[k] = sorted(set(ledger[k]))

    return ledger

# --- 台帳クエリ（コミット→特徴量） ---

def _count_events_before(dates_sorted, t, window_days=None):
    """
    dates_sorted: 昇順のdateリスト
    t: 基準日（含まない）
    window_days=None なら「過去すべて」、数値ならその日数だけ遡る
    戻り値: 件数, 直近日付 or None
    """
    if not dates_sorted:
        return 0, None
    # t 未満のものだけ
    idx = bisect_left(dates_sorted, t)
    if idx == 0:
        return 0, None
    if window_days is None:
        last_date = dates_sorted[idx-1]
        return idx, last_date
    start = t - timedelta(days=window_days)
    left = bisect_right(dates_sorted, start - timedelta(days=0))
    cnt = max(0, idx - left)
    last_date = dates_sorted[idx-1] if cnt > 0 else None
    return cnt, last_date

def add_commit_file_vcc_features(metrics_df, vcc_ledger,
                                 path_col='commit_change_file_path_filetered',
                                 date_col='merge_date',
                                 windows=(90,)):
    """
    各コミットに以下を付与:
      - changed_files_past_vcc_any_ever (0/1)
      - changed_files_past_vcc_count_ever (int)
      - changed_files_past_vcc_file_ratio_ever (float)
      - changed_files_time_since_last_vcc_min (float 日, NaN許容)
      - changed_files_past_vcc_any_{Wd} (0/1) 例: 90d
      - changed_files_past_vcc_count_{Wd} (int)
    """
    df = metrics_df.copy()

    # 日付列が無ければ生成
    if date_col not in df.columns and 'commit_datetime' in df.columns:
        df[date_col] = pd.to_datetime(df['commit_datetime'].str.replace(' ', 'T', regex=False),
                                      errors='coerce', utc=True).dt.date

    df = df.dropna(subset=[date_col])

    # 出力列の器
    df['changed_files_past_vcc_any_ever'] = 0
    df['changed_files_past_vcc_count_ever'] = 0
    df['changed_files_past_vcc_file_ratio_ever'] = 0.0
    df['changed_files_time_since_last_vcc_min'] = np.nan

    for W in windows:
        df[f'changed_files_past_vcc_any_{W}d'] = 0
        df[f'changed_files_past_vcc_count_{W}d'] = 0

    # 反復
    for i, row in df.iterrows():
        t = row[date_col]
        files = [_norm_path(p) for p in _parse_changed_files(row.get(path_col, None))]
        files = [p for p in files if p]

        if not files:
            continue

        ever_hit_files = 0
        ever_cnt_total = 0
        last_dates = []

        window_cnts = {W: 0 for W in windows}
        window_any = {W: 0 for W in windows}

        for fp in files:
            dates = vcc_ledger.get(fp, [])

            # ever
            cnt_ever, last_ever = _count_events_before(dates, t, window_days=None)
            if cnt_ever > 0:
                ever_hit_files += 1
                ever_cnt_total += cnt_ever
                if last_ever:
                    last_dates.append(last_ever)

            # windows
            for W in windows:
                cnt_w, last_w = _count_events_before(dates, t, window_days=W)
                window_cnts[W] += cnt_w
                if cnt_w > 0:
                    window_any[W] = 1

        df.at[i, 'changed_files_past_vcc_any_ever'] = 1 if ever_hit_files > 0 else 0
        df.at[i, 'changed_files_past_vcc_count_ever'] = ever_cnt_total
        df.at[i, 'changed_files_past_vcc_file_ratio_ever'] = (ever_hit_files / len(files)) if files else 0.0

        if last_dates:
            df.at[i, 'changed_files_time_since_last_vcc_min'] = (t - max(last_dates)).days

        for W in windows:
            df.at[i, f'changed_files_past_vcc_any_{W}d'] = window_any[W]
            df.at[i, f'changed_files_past_vcc_count_{W}d'] = window_cnts[W]

    return df


# --- ▼▼▼【修正箇所】集約ロジック変更 ▼▼▼ ---
def process_project_coverage(project_id, directory_name, metrics_base_path, coverage_base_project_path, patch_coverage_base_path, output_base_path):
    """
    単一プロジェクトのカバレッジ計算処理全体を管理する。
    """
    if pd.isna(directory_name) or not directory_name:
        directory_name = project_id
    metrics_file = os.path.join(metrics_base_path, directory_name, f"{directory_name}_commit_metrics_with_tfidf.csv")
    coverage_file_project = os.path.join(coverage_base_project_path, project_id, f"{project_id}_and_date.csv")
    coverage_file_project_total = os.path.join(coverage_base_project_path, project_id, f"{project_id}_total_and_date.csv")
    patch_coverage_file = os.path.join(patch_coverage_base_path, project_id, f"{project_id}_patch_coverage.csv")
    
    print(f"▶️  プロジェクト '{project_id}' の処理を開始します...")

    data = load_and_prepare_data(metrics_file, coverage_file_project, coverage_file_project_total)
    if data[0] is None:
        print(f"⚠️  プロジェクト '{project_id}' は必須ファイルが不足またはエラーのためスキップします。")
        return
    metrics_df, coverage_project_df, coverage_project_total_df = data
    # ▼ 追加：VCC台帳とファイル過去VCC特徴量（30/90/180日とever）
    vcc_ledger = build_vcc_file_ledger(metrics_df,
                                    path_col='commit_change_file_path_filetered',
                                    is_vcc_col='is_vcc',
                                    date_col='merge_date')
    metrics_df = add_commit_file_vcc_features(metrics_df, vcc_ledger,
                                            path_col='commit_change_file_path_filetered',
                                            date_col='merge_date',
                                            windows=(30,90,180))

    patch_coverage_df = load_patch_coverage_data(patch_coverage_file)

    result_df = calculate_commit_coverage(metrics_df, coverage_project_df, coverage_project_total_df, patch_coverage_df)

    if result_df is None or result_df.empty:
        print(f"ℹ️  プロジェクト '{project_id}' で処理できるデータがありませんでした。")
        return

    # --- 日毎の集約処理 ---

    print(f"  -> 日毎のデータ集約を計算中...")
    
    # 1. 日付とコミット時刻でソート
    if 'commit_datetime' in result_df.columns:
        result_df.sort_values(by=['merge_date', 'commit_datetime'], inplace=True)
    else:
        result_df.sort_values(by='merge_date', inplace=True)
        
    # 【★★★ここから修正★★★】
    # 【★★★置き換え★★★】t−1保持と (t−1)−(t−2) の日次Δを totals/patch から直接つくる
    print("  -> カバレッジの変化量(_delta)を計算中...")
    # TODO
    coverage_types = ['function', 'line', 'region', 'branch', 'insta']

    # 1) totals の「日次系列（全日）」を作る → t−1 と t−2
    tot = coverage_project_total_df.copy()
    tot['merge_date'] = pd.to_datetime(tot['merge_date']).dt.date
    tot = tot.sort_values('merge_date').drop_duplicates('merge_date', keep='last')

    # project_total_*_percent 列を用意（元列は function_percent など）
    for t in coverage_types:
        src = f'{t}_percent'
        dst = f'project_total_{t}_percent'
        if src in tot.columns and dst not in tot.columns:
            tot[dst] = pd.to_numeric(tot[src], errors='coerce')

    tot = tot.set_index('merge_date')
    last_prev     = tot[[f'project_total_{t}_percent' for t in coverage_types]].shift(1)   # t−1
    last_prevprev = tot[[f'project_total_{t}_percent' for t in coverage_types]].shift(2)   # t−2
    # (t−1) − (t−2) を作り、列名に _delta を付与して上書き事故を防ぐ
    tot_deltas = (last_prev - last_prevprev)
    tot_deltas.columns = [f'{c}_delta' for c in tot_deltas.columns]                                      # (t−1) − (t−2)

    # 2) patch coverage の t−1 と Δ（ある場合のみ）
    patch_prev = patch_delta = None
    if patch_coverage_df is not None and \
       'merge_date' in patch_coverage_df.columns and \
       'patch_coverage_recalculated' in patch_coverage_df.columns:
        pc = (patch_coverage_df
              .sort_values('merge_date')
              .drop_duplicates('merge_date', keep='last')
              .set_index('merge_date')['patch_coverage_recalculated'])
        patch_prev  = pc.shift(1)                 # t−1
        patch_delta = pc.shift(1) - pc.shift(2)   # (t−1) − (t−2)

    # 3) コミットのあった日だけに引き直す（順序も保証）
    days = (result_df[['merge_date']]
            .drop_duplicates()
            .sort_values('merge_date')
            .set_index('merge_date'))

    tminus1_tbl = days.join(last_prev)
    delta_tbl   = days.join(tot_deltas)
    if patch_prev is not None:
        tminus1_tbl = tminus1_tbl.join(patch_prev.rename('patch_coverage_recalculated'))
    if patch_delta is not None:
        delta_tbl   = delta_tbl.join(patch_delta.rename('patch_coverage_recalculated_delta'))

    # 4) result_df に反映（t−1値は上書き、Δは追加）— NaNはそのまま維持
    result_df = result_df.merge(tminus1_tbl.reset_index(), on='merge_date', how='left', suffixes=('', '_recalc'))
    for t in coverage_types:
        col = f'project_total_{t}_percent'
        rec = f'{col}_recalc'
        if rec in result_df.columns:
            result_df[col] = result_df[rec]
            result_df.drop(columns=[rec], inplace=True)

    result_df = result_df.merge(delta_tbl.reset_index(), on='merge_date', how='left')
    
    # project_commit_*系の特徴量は削除
    # 5) コミット級 Δ は「commit_%(t−1) − total(t−1)」
    # for t in coverage_types:
    #     current_col = f'project_commit_{t}_percent'
    #     base_col    = f'project_total_{t}_percent'            # ここは t−1
    #     delta_col   = f'project_commit_{t}_percent_delta'
    #     if current_col in result_df.columns and base_col in result_df.columns:
    #         result_df[delta_col] = result_df[current_col] - result_df[base_col]
    
    # 2. is_vccのコミット数をカウントするために列を複製
    if 'is_vcc' in result_df.columns:
        result_df['vcc_commit_count'] = result_df['is_vcc']

    # 3. settings.pyとご指示に基づき、日毎の集約ルールを動的に定義
    print("  -> 指示に基づき、集約ルールを定義中...")

    aggregation_rules = {}

    aggregation_rules['commit_hash'] = 'count'
    aggregation_rules['is_vcc'] = 'max' # VCCがあったかのフラグ (0/1)
    
    # 【★★★ここを修正★★★】
    # VCCコミットの総数をカウントするルールを追加
    aggregation_rules['vcc_commit_count'] = 'sum' 

    # Kamei特徴量のルールを定義
    kamei_rules = {
        'sum': [
            "subsystems_changed", "directories_changed", "files_changed", "lines_added", "lines_deleted", 
            "ndev", "nuc","total_prev_loc"
        ],
        'mean': [
            "entropy", "age", "exp", "rexp", "sexp", "is_bug_fix"
        ]
    }
    for rule, features in kamei_rules.items():
        for feature in features:
            if feature in KAMEI_FEATURES:
                aggregation_rules[feature] = rule

    # VCCFinder特徴量のルールを定義 (変更なし)
    for feature in VCCFINDER_FEATURES:
        aggregation_rules[feature] = 'sum'
        
    # ▼ 追加：過去VCC特徴量の集約ルール
    bool_feats = [
        'changed_files_past_vcc_any_ever',
        'changed_files_past_vcc_any_30d',
        'changed_files_past_vcc_any_90d',
        'changed_files_past_vcc_any_180d',
    ]
    count_feats = [
        'changed_files_past_vcc_count_ever',
        'changed_files_past_vcc_count_30d',
        'changed_files_past_vcc_count_90d',
        'changed_files_past_vcc_count_180d',
    ]
    other_feats = {
        'changed_files_past_vcc_file_ratio_ever': 'mean',
        'changed_files_time_since_last_vcc_min': 'min',
    }

    for f in bool_feats:
        if f in result_df.columns: aggregation_rules[f] = 'max'
    for f in count_feats:
        if f in result_df.columns: aggregation_rules[f] = 'sum'
    for f, rule in other_feats.items():
        if f in result_df.columns: aggregation_rules[f] = rule
    

    # カバレッジ特徴量のルールを定義 (変更なし)
    ALL_COVERAGE_FEATURES = ALL_PERCENT_FEATURES + ALL_CALCULATION_FEATURES
    for feature in ALL_COVERAGE_FEATURES:
         # project_commit_*_delta はコミットごとの変化量なのでsumで集計
        if feature.startswith('project_commit_') and feature.endswith('_delta'):
            aggregation_rules[feature] = 'sum'
        # それ以外のカバレッジ関連特徴量（project_total, patch_coverageなど）はlastで集計
        else:
            aggregation_rules[feature] = 'last'
    
    # t−1 の total％ を必ず残す
    for t in coverage_types:
        aggregation_rules[f'project_total_{t}_percent'] = 'last'

    # (t−1)−(t−2) の Δ も残す
    for t in coverage_types:
        dcol = f'project_total_{t}_percent_delta'
        if dcol in result_df.columns:
            aggregation_rules[dcol] = 'last'

    # patch coverage の t−1 と Δ も残す
    if 'patch_coverage_recalculated' in result_df.columns:
        aggregation_rules['patch_coverage_recalculated'] = 'last'
    if 'patch_coverage_recalculated_delta' in result_df.columns:
        aggregation_rules['patch_coverage_recalculated_delta'] = 'last'
    
    # 4. 実際に集約処理を実行
    valid_aggregation_rules = {k: v for k, v in aggregation_rules.items() if k in result_df.columns}
    daily_aggregated_df = result_df.groupby('merge_date').agg(valid_aggregation_rules).reset_index()

    if 'commit_hash' in valid_aggregation_rules:
        daily_aggregated_df.rename(columns={'commit_hash': 'daily_commit_count'}, inplace=True)

    # 5. 結果をCSVファイルに保存
    output_dir = os.path.join(output_base_path, project_id)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{project_id}_daily_aggregated_metrics.csv") 
    daily_aggregated_df.to_csv(output_path, index=False)
    print(f"成功: {project_id} の日毎集約データを {output_path} に保存しました。")

def main():
    """
    メイン関数：コマンドライン引数で指定された単一プロジェクトの処理を実行する。
    """
    parser = argparse.ArgumentParser(description='単一プロジェクトのコミットカバレッジから日次集約データを計算します。')
    parser.add_argument('project_id', type=str, help='処理対象のプロジェクトID (例: apache-httpd)')
    parser.add_argument('directory_name', type=str, help='プロジェクトに対応するディレクトリ名 (例: httpd)')
    # ルートディレクトリ類（ENV で上書き可能）
    parser.add_argument('--metrics', default=os.environ.get('VULJIT_METRICS_DIR'), help='metrics_base_path (default: $VULJIT_METRICS_DIR)')
    parser.add_argument('--coverage', default=os.environ.get('VULJIT_COVERAGE_AGG_DIR'), help='coverage_base_project_path (default: $VULJIT_COVERAGE_AGG_DIR)')
    parser.add_argument('--patch-coverage', dest='patch', default=os.environ.get('VULJIT_PATCH_COV_DIR'), help='patch_coverage_base_path (default: $VULJIT_PATCH_COV_DIR)')
    parser.add_argument('--out', default=os.environ.get('VULJIT_BASE_DATA_DIR'), help='output_base_path (default: $VULJIT_BASE_DATA_DIR)')
    args = parser.parse_args()

    # パス設定（引数優先、未指定時は repo 相対デフォルト）
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, '..'))
    metrics_base_path = args.metrics or os.path.join(repo_root, 'data', 'metrics_output')
    coverage_base_project_path = args.coverage or os.path.join(repo_root, 'outputs', 'metrics', 'coverage_aggregate')
    patch_coverage_base_path = args.patch or os.path.join(repo_root, 'outputs', 'metrics', 'patch_coverage')
    output_base_path = args.out or os.path.join(repo_root, 'data')

    # 単一プロジェクトの処理を呼び出し
    process_project_coverage(
        args.project_id,
        args.directory_name,
        metrics_base_path,
        coverage_base_project_path,
        patch_coverage_base_path,
        output_base_path
    )

    print(f"\n--- プロジェクト '{args.project_id}' の処理が正常に完了しました ---")

if __name__ == '__main__':
    main()
