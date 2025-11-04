import argparse
import os
from pathlib import Path

import pandas as pd
from typing import Optional, Tuple, List

from sklearn.feature_extraction.text import TfidfVectorizer


def add_commit_tfidf(
    df: pd.DataFrame,
    *,
    max_features: int = 10,
    stop_words: Optional[str] = "english",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    コミットメッセージに TF-IDF 特徴量列を追加した DataFrame を返す。
    """
    if "commit_message" not in df.columns:
        raise ValueError("DataFrame に 'commit_message' 列が存在しません。")

    commit_messages = df["commit_message"].fillna("")
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        min_df=0.0,
        analyzer="word",
        tokenizer=None,
        lowercase=True,
        preprocessor=None,
        stop_words=stop_words,
        max_features=max_features,
        use_idf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(commit_messages)
    feature_names = list(vectorizer.get_feature_names_out())
    tfidf_col_names = [f"VCC_w{i+1}" for i in range(len(feature_names))]

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_col_names, index=df.index)

    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
    return df_with_tfidf, feature_names


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

    final_df, feature_names = add_commit_tfidf(df)
    print(f"抽出されたトップ{len(feature_names)}単語: {feature_names}")

    # --- 6. 新しいCSVファイルとして保存 ---
    project_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_filepath, index=False)
    print(f"TF-IDFメトリクスを追加した新しいファイルを保存しました: {output_filepath}")
    
    # --- 7. 結果の確認 ---
    print("\n--- 処理後のデータフレーム（先頭5行） ---")
    print(final_df.head())
    print("\n--- 追加された列 ---")
    added_cols = [col for col in final_df.columns if col.startswith("VCC_w")]
    print(final_df[added_cols].head())


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
