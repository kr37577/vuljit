import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# --- 設定項目 (デフォルト値。必要であれば引数で上書き可能) ---
# これは全プロジェクト共通の脆弱性情報ファイルです。パスは固定的か、
# もし頻繁に変わる場合は引数で渡すことも考えられます。
DEFAULT_VULNERABILITIES_FILE_PATH = 'datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv'

INTRODUCED_COMMITS_COLUMN = 'introduced_commits'  # 脆弱性情報CSV内の導入コミット列名
COMMIT_HASH_COLUMN = 'commit_hash'  # コミットメトリクスCSV内のコミットハッシュ列名
NEW_LABEL_COLUMN = 'is_vcc'  # 追加するラベル列名

def load_vulnerabilities(vulnerabilities_path: str) -> pd.DataFrame:
    """脆弱性CSVを読み込むヘルパー。"""
    vulnerabilities_file = Path(vulnerabilities_path)
    if not vulnerabilities_file.exists():
        raise FileNotFoundError(f"脆弱性情報ファイルが見つかりません: {vulnerabilities_file}")

    df = pd.read_csv(vulnerabilities_file)
    if INTRODUCED_COMMITS_COLUMN not in df.columns:
        raise ValueError(
            f"脆弱性CSV '{vulnerabilities_file}' に '{INTRODUCED_COMMITS_COLUMN}' 列が見つかりません。"
        )
    return df


def add_vcc_labels(
    df_metrics: pd.DataFrame,
    package_name: str,
    vulnerabilities: Optional[pd.DataFrame] = None,
    vulnerabilities_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    コミットメトリクス DataFrame に脆弱性導入ラベル列を追加して返す。
    """
    if COMMIT_HASH_COLUMN not in df_metrics.columns:
        raise ValueError(f"コミットメトリクスに '{COMMIT_HASH_COLUMN}' 列がありません。")

    if vulnerabilities is None:
        vuln_path = vulnerabilities_path or DEFAULT_VULNERABILITIES_FILE_PATH
        vulnerabilities = load_vulnerabilities(vuln_path)

    package_vulns = vulnerabilities[vulnerabilities['package_name'] == package_name].copy()
    unique_introduced_commits: set[str] = set()

    if not package_vulns.empty:
        package_vulns[INTRODUCED_COMMITS_COLUMN] = package_vulns[INTRODUCED_COMMITS_COLUMN].fillna('')
        introduced_commits_series = package_vulns[INTRODUCED_COMMITS_COLUMN].astype(str).str.split(' ').explode()
        unique_introduced_commits = set(
            introduced_commits_series[introduced_commits_series.str.strip() != ''].dropna().astype(str).unique()
        )

    df_labeled = df_metrics.copy()
    df_labeled[COMMIT_HASH_COLUMN] = df_labeled[COMMIT_HASH_COLUMN].astype(str)

    if unique_introduced_commits:
        df_labeled[NEW_LABEL_COLUMN] = df_labeled[COMMIT_HASH_COLUMN].isin(unique_introduced_commits)
    else:
        df_labeled[NEW_LABEL_COLUMN] = False

    return df_labeled


def main():
    parser = argparse.ArgumentParser(description="指定されたパッケージのコミットメトリクスに脆弱性情報を元にラベル付けします。")
    parser.add_argument("package_name", help="処理対象のパッケージ名またはリポジトリ名（例: 'arrow', 'harfbuzz'）。Bashスクリプト内の 'local_repo_dir_name' に対応します。")
    parser.add_argument("metrics_file_path", help="パッケージのコミットメトリクスCSVファイルへのパス（例: output/arrow/arrow_commit_metrics_output_per_commit.csv）。")
    parser.add_argument("-o", "--output_dir", required=True, help="ラベル付けされた出力CSVファイルを保存するディレクトリ（例: output/arrow/）。")
    parser.add_argument("--vuln_file", default=DEFAULT_VULNERABILITIES_FILE_PATH,
                        help=f"メインの脆弱性情報CSVファイルへのパス。デフォルト: {DEFAULT_VULNERABILITIES_FILE_PATH}")
    
    args = parser.parse_args()

    package_name = args.package_name
    metrics_file_path = args.metrics_file_path
    output_dir = args.output_dir
    vulnerabilities_file_path = args.vuln_file

    # 出力ファイルパスを構築
    # 例: output/arrow/arrow_commit_metrics_with_vulnerability_label.csv
    output_file_name = f"{package_name}_commit_metrics_with_vulnerability_label.csv"
    output_file_path = os.path.join(output_dir, output_file_name)

    try:
        print(f"パッケージ '{package_name}' の処理を開始します...")
        print(f"ベースとなる脆弱性情報を '{vulnerabilities_file_path}' から読み込んでいます...")
        df_vuln_base = load_vulnerabilities(vulnerabilities_file_path)

        print(f"コミットメトリクスを '{metrics_file_path}' から読み込んでいます...")
        df_metrics = pd.read_csv(metrics_file_path)

        df_labeled = add_vcc_labels(
            df_metrics,
            package_name=package_name,
            vulnerabilities=df_vuln_base,
        )

        os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリが存在することを確認
        df_labeled.to_csv(output_file_path, index=False)
        print(f"処理が完了しました。結果は '{output_file_path}' に保存されました。")

        num_labeled_commits = df_labeled[NEW_LABEL_COLUMN].sum()
        print(f"{num_labeled_commits} 件のコミットに脆弱性導入ラベル ('{NEW_LABEL_COLUMN}') が付けられました。")

    except Exception as e:
        print(f"予期せぬエラーが発生しました ({package_name}): {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
