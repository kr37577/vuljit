import pandas as pd
import argparse
import re

def normalize_string(s: str) -> str:
    """文字列を正規化（小文字化し、記号を削除）する"""
    return re.sub(r'[\-_\.]', '', s.lower())

def generate_mapping(github_list_path: str, ossfuzz_list_path: str, output_prefix: str):
    """
    2つのプロジェクトリストを読み込み、複数の戦略でマッピングを試みる。

    Args:
        github_list_path (str): 'owner/repository' 形式のリストファイルパス。
        ossfuzz_list_path (str): OSS-FuzzのプロジェクトIDリストのファイルパス。
        output_prefix (str): 出力ファイル名のプレフィックス。
    """
    try:
        # --- 1. データの読み込みと準備 ---
        print("INFO: データを読み込んでいます...")
        # GitHubリストを読み込み、'directory_name'と正規化名を作成
        github_df = pd.read_csv(github_list_path, header=None, names=['full_repo_path'])
        github_df['directory_name'] = github_df['full_repo_path'].str.split('/').str[-1].str.strip('/')
        github_df.dropna(subset=['directory_name'], inplace=True)
        github_df['normalized_dir'] = github_df['directory_name'].apply(normalize_string)
        
        # OSS-Fuzzリストを読み込み、正規化名を作成
        ossfuzz_df = pd.read_csv(ossfuzz_list_path, header=None, names=['project_id'])
        ossfuzz_df.dropna(inplace=True)
        ossfuzz_df['normalized_id'] = ossfuzz_df['project_id'].apply(normalize_string)

        # マッピング結果を格納するリスト
        successful_matches = []
        
        # マッチング対象のリスト（処理が進むにつれて減っていく）
        remaining_ossfuzz = ossfuzz_df.copy()
        
        print("INFO: マッピング処理を開始します...")

        # --- 2. マッチング戦略の実行 ---
        
        # 戦略1: project_id と directory_name の完全一致 (大文字小文字無視)
        print("INFO: [PASS 1/2] 完全一致を試みています...")
        merged_exact = pd.merge(
            remaining_ossfuzz, 
            github_df, 
            left_on=remaining_ossfuzz['project_id'].str.lower(),
            right_on=github_df['directory_name'].str.lower(),
            how='inner'
        ).drop(columns=['key_0'])
        
        if not merged_exact.empty:
            successful_matches.append(merged_exact[['project_id', 'directory_name']])
            remaining_ossfuzz = remaining_ossfuzz[~remaining_ossfuzz['project_id'].isin(merged_exact['project_id'])]

        # 戦略2: 正規化された名前での完全一致
        print("INFO: [PASS 2/2] 正規化後の一致を試みています...")
        merged_normalized = pd.merge(
            remaining_ossfuzz,
            github_df,
            left_on='normalized_id',
            right_on='normalized_dir',
            how='inner'
        )
        if not merged_normalized.empty:
            successful_matches.append(merged_normalized[['project_id', 'directory_name']])
            remaining_ossfuzz = remaining_ossfuzz[~remaining_ossfuzz['project_id'].isin(merged_normalized['project_id'])]

        # --- 3. 結果の出力 ---
        print("INFO: 結果をファイルに出力しています...")
        
        # 成功したマッピング
        if successful_matches:
            final_successful = pd.concat(successful_matches).drop_duplicates(subset=['project_id'])
            output_path = f"{output_prefix}_successful.csv"
            final_successful.to_csv(output_path, index=False)
            print(f"SUCCESS: {len(final_successful)}件の成功したマッピングを '{output_path}' に保存しました。")
        
        # マッチしなかったもの
        if not remaining_ossfuzz.empty:
            output_path = f"{output_prefix}_unmatched.csv"
            remaining_ossfuzz[['project_id']].to_csv(output_path, index=False)
            print(f"INFO: {len(remaining_ossfuzz)}件のマッチしなかったIDを '{output_path}' に保存しました。")

    except FileNotFoundError as e:
        print(f"ERROR: ファイルが見つかりません: {e.filename}")
    except Exception as e:
        print(f"ERROR: 予期せぬエラーが発生しました: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="GitHubリポジトリリストとOSS-FuzzプロジェクトIDリストをマッピングします。",
        epilog="使用例: python map_projects.py github_list.txt ossfuzz_ids.txt -o project_mapping"
    )
    parser.add_argument("github_list", help="GitHubリポジトリリストのファイルパス ('owner/repo'形式)。")
    parser.add_argument("ossfuzz_list", help="OSS-FuzzプロジェクトIDリストのファイルパス。")
    parser.add_argument("-o", "--output_prefix", default="project_mapping", help="出力ファイル名のプレフィックス。")

    args = parser.parse_args()
    
    generate_mapping(args.github_list, args.ossfuzz_list, args.output_prefix)