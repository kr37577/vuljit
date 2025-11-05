# data_preparation.py

import pandas as pd
from typing import Optional, Tuple, List

def preprocess_dataframe_for_within_project(
    df: pd.DataFrame, df_original_for_cols: pd.DataFrame
) -> Optional[Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]]:
    """
    単一プロジェクトのDataFrameの前処理を行う。
    元のコードの `preprocess_dataframe_for_within_project` のロジックをここに配置します。
    """
    print("\n--- データ前処理開始 ---")
    
    # Backward compatibility: allow y_is_vcc / label_date columns emitted by aggregation pipeline
    if 'is_vcc' not in df.columns and 'y_is_vcc' in df.columns:
        df = df.copy()
        df['is_vcc'] = df['y_is_vcc']
    if 'merge_date' not in df.columns and 'label_date' in df.columns:
        df = df.copy()
        df['merge_date'] = df['label_date']
        if 'merge_date' not in df_original_for_cols.columns and 'label_date' in df_original_for_cols.columns:
            df_original_for_cols = df_original_for_cols.copy()
            df_original_for_cols['merge_date'] = df_original_for_cols['label_date']

    required_cols = ['is_vcc', 'merge_date']
    if not all(col in df.columns for col in required_cols):
        print("エラー: 必須カラム ('is_vcc', 'merge_date') のいずれかがCSVに存在しません。")
        return None

    df_processed = df.copy()
    
    # 1. commit_datetime を datetime 型に変換し、時系列でソート
    #    'YYYY-MM-DD HH:MM:SS+ZZ:ZZ' 形式を自動で解釈
    #    不正な値（変換できない値）を含む行は削除 (NaT)
    df_processed['merge_date'] = pd.to_datetime(df_processed['merge_date'], errors='coerce', utc=True)
    df_processed.dropna(subset=['merge_date'], inplace=True)
    # 【重要】indexを保持して並び替え（reset_indexは行わない）→ 後段の予測確率アラインに必須
    df_processed = df_processed.sort_values(by='merge_date', ascending=True)
    print(f"  データを時系列（merge_date）でソートしました。データ数: {len(df_processed)}")

    # 'is_vcc'列を0/1の数値データに変換する処理
    is_vcc_series = df_processed['is_vcc'].copy()
    if pd.api.types.is_bool_dtype(is_vcc_series):
        is_vcc_series = is_vcc_series.astype(int)
    elif pd.api.types.is_numeric_dtype(is_vcc_series):
        is_vcc_series = is_vcc_series.fillna(-1).astype(int).replace({-1: pd.NA})
    elif is_vcc_series.dtype == object:
        map_to_int = {'True': 1, 'true': 1, 'TRUE': 1,
                      'False': 0, 'false': 0, 'FALSE': 0, True: 1, False: 0}
        is_vcc_series = pd.to_numeric(is_vcc_series.replace(map_to_int), errors='coerce')
    
    df_processed['is_vcc_processed'] = is_vcc_series
    df_processed.dropna(subset=['is_vcc_processed'], inplace=True)
    if df_processed.empty:
        print("is_vccの処理後、有効なデータが残りませんでした。")
        return None
    df_processed['is_vcc_processed'] = df_processed['is_vcc_processed'].astype(int)

    # df_processed['repo_path'] = df_processed['repo_path'].fillna('UNKNOWN_PROJECT')

    # 特徴量として使用しない列を定義
    non_feature_cols = [
        'commit_hash', 'repo_path', 'commit_datetime', 'is_vcc', 'is_vcc_processed',
        'commit_change_file_path_filetered', 'merge_date','vcc_commit_count',
        'label_date', 'y_is_vcc'
    ]
    feature_columns = [col for col in df_original_for_cols.columns if col not in non_feature_cols]

    if not feature_columns:
        print("特徴量カラムが見つかりませんでした。")
        return None

    # 特徴量(X), 目的変数(y)を抽出
    X = df_processed[feature_columns].copy()
    y = df_processed['is_vcc_processed'].astype(int)
    # repo_paths = df_processed['repo_path']

    # 特徴量を数値型に変換し、NaNを0で埋める
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype(int)
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)

    print(f"前処理完了。特徴量カラム数: {len(feature_columns)}")
    print("--- データ前処理終了 ---\n")
    return X, y, None, feature_columns
