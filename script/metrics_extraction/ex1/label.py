import pandas as pd
import sys
import os
import argparse

# --- 設定項目 (デフォルト値。必要であれば引数で上書き可能) ---
# これは全プロジェクト共通の脆弱性情報ファイルです。パスは固定的か、
# もし頻繁に変わる場合は引数で渡すことも考えられます。
DEFAULT_VULNERABILITIES_FILE_PATH = '/work/riku-ka/fuzz_introspector/rq3_dataset/oss_fuzz_vulns_2025802.csv'

INTRODUCED_COMMITS_COLUMN = 'introduced_commits'  # 脆弱性情報CSV内の導入コミット列名
COMMIT_HASH_COLUMN = 'commit_hash'  # コミットメトリクスCSV内のコミットハッシュ列名
NEW_LABEL_COLUMN = 'is_vcc'  # 追加するラベル列名

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
        try:
            df_vuln_base = pd.read_csv(vulnerabilities_file_path)
        except FileNotFoundError:
            print(f"エラー: 脆弱性情報ファイル '{vulnerabilities_file_path}' が見つかりません。処理を終了します。")
            sys.exit(1)

        if INTRODUCED_COMMITS_COLUMN not in df_vuln_base.columns:
            print(
                f"エラー: '{vulnerabilities_file_path}' に '{INTRODUCED_COMMITS_COLUMN}' 列が見つかりません。処理を終了します。")
            sys.exit(1)

        print(f"パッケージ '{package_name}' の脆弱情報をフィルタリングしています...")
        # 1. 脆弱性導入コミットのリスト作成 (特定の package_name のみに限定)
        current_package_df_vuln = df_vuln_base[df_vuln_base['package_name'] == package_name].copy()

        unique_introduced_commits = set()

        if not current_package_df_vuln.empty:
            current_package_df_vuln.loc[:, INTRODUCED_COMMITS_COLUMN] = current_package_df_vuln[
                INTRODUCED_COMMITS_COLUMN].fillna('')
            
            introduced_commits_series = current_package_df_vuln[INTRODUCED_COMMITS_COLUMN].astype(
                str).str.split(' ').explode()
            
            unique_introduced_commits = set(
                introduced_commits_series[introduced_commits_series.str.strip() != ''].dropna().unique())
        
        if not unique_introduced_commits:
            print(
                f"警告: '{vulnerabilities_file_path}' から '{package_name}' の有効な脆弱性導入コミットハッシュが見つかりませんでした。")
        else:
            print(
                f"'{package_name}' の重複しない脆弱性導入コミットハッシュは {len(unique_introduced_commits)} 件です。")
            # print(f"サンプルハッシュ: {list(unique_introduced_commits)[:5]}") # 確認用

        # 2. コミットメトリクスへのラベル付け
        try:
            print(f"コミットメトリクスを '{metrics_file_path}' から読み込んでいます...")
            df_metrics = pd.read_csv(metrics_file_path)
        except FileNotFoundError:
            print(f"エラー: コミットメトリクスファイル '{metrics_file_path}' が見つかりません。'{package_name}' パッケージの処理を終了します。")
            sys.exit(1)

        if COMMIT_HASH_COLUMN not in df_metrics.columns:
            print(f"エラー: '{metrics_file_path}' に '{COMMIT_HASH_COLUMN}' 列が見つかりません。'{package_name}' パッケージの処理を終了します。")
            sys.exit(1)

        df_metrics[COMMIT_HASH_COLUMN] = df_metrics[COMMIT_HASH_COLUMN].astype(str)
        
        # 比較のために unique_introduced_commits の要素も文字列型に変換 (多くは既に文字列だが念のため)
        unique_introduced_commits_str = {str(commit) for commit in unique_introduced_commits}

        df_metrics[NEW_LABEL_COLUMN] = df_metrics[COMMIT_HASH_COLUMN].isin(
            unique_introduced_commits_str)

        # 3. 結果の保存
        os.makedirs(output_dir, exist_ok=True) # 出力ディレクトリが存在することを確認
        df_metrics.to_csv(output_file_path, index=False)
        print(f"処理が完了しました。結果は '{output_file_path}' に保存されました。")

        num_labeled_commits = df_metrics[NEW_LABEL_COLUMN].sum()
        print(f"{num_labeled_commits} 件のコミットに脆弱性導入ラベル ('{NEW_LABEL_COLUMN}') が付けられました。")

    except Exception as e:
        print(f"予期せぬエラーが発生しました ({package_name}): {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()