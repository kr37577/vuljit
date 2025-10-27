import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf_for_commits(project_name: str, metrics_base_dir: str) -> None:
    """
    コミットメトリクスCSVを読み込み、コミットメッセージのTF-IDFを計算して
    新しい列として追加し、別のファイルに保存します。

    Args:
        project_name (str): CSVファイルが格納されているプロジェクト名。
        metrics_base_dir (str): メトリクスファイルを格納するベースディレクトリ。
    """

    # --- 1. ファイルパスの準備 ---
    project_dir = Path(metrics_base_dir) / project_name
    input_filepath = project_dir / f"{project_name}_commit_metrics_with_vulnerability_label.csv"
    output_filepath = project_dir / f"{project_name}_commit_metrics_with_tfidf.csv"

    # --- 2. CSVファイルの読み込み ---
    try:
        df = pd.read_csv(input_filepath)
        print(f"'{input_filepath}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {input_filepath}")
        return

    # --- 3. TF-IDFの計算 ---
    # commit_message列の欠損値(NaN)を空文字に置き換える
    commit_messages = df['commit_message'].fillna('')

    # TfidfVectorizerの初期化
    # - stop_words='english': a, the, is などの一般的な単語を除外
    # - max_features=10: 出現頻度が高い上位10単語のみを対象
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        min_df=0.0,
        analyzer="word",
        tokenizer=None,
        lowercase=True,  
        preprocessor=None,
        stop_words="english",
        max_features=10,
        use_idf=True
    )

    # TF-IDF行列を計算
    tfidf_matrix = vectorizer.fit_transform(commit_messages)

    # --- 4. 結果をDataFrameに変換 ---
    # 特徴量名（単語）を取得
    feature_names = vectorizer.get_feature_names_out()
    print(f"抽出されたトップ10単語: {feature_names}")
    
    # 新しい列名を生成 (VCC_w1, VCC_w2, ...)
    tfidf_col_names = [f'VCC_w{i+1}' for i in range(len(feature_names))]

    # TF-IDF行列をDataFrameに変換
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_col_names)

    # --- 5. 元のDataFrameと結合 ---
    # インデックスをリセットして、安全に結合できるようにする
    df.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([df, tfidf_df], axis=1)

    # --- 6. 新しいCSVファイルとして保存 ---
    project_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_filepath, index=False)
    print(f"TF-IDFメトリクスを追加した新しいファイルを保存しました: {output_filepath}")
    
    # --- 7. 結果の確認 ---
    print("\n--- 処理後のデータフレーム（先頭5行） ---")
    print(final_df.head())
    print("\n--- 追加された列 ---")
    print(final_df[tfidf_col_names].head())


if __name__ == '__main__':
    # コマンドライン引数の解析
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[3]
    default_metrics_dir = os.environ.get(
        "VULJIT_METRICS_DIR",
        os.path.join(repo_root, "datasets", "metric_inputs"),
    )

    parser = argparse.ArgumentParser(description='Calculate TF-IDF for commit messages.')
    parser.add_argument('-p', '--project_name', type=str, required=True, help='Name of the project to process.')
    parser.add_argument('-m', '--metrics-dir', default=default_metrics_dir,
                        help=f'Base directory where metrics files are stored (default: {default_metrics_dir})')
    args = parser.parse_args()

    if args.project_name == 'your_project_name_here':
        print("エラー: `project_name`変数を実際のプロジェクト名に設定してください。")
    else:
        calculate_tfidf_for_commits(args.project_name, args.metrics_dir)
