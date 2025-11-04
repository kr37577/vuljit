import pandas as pd
import argparse
import os
import sys
from urllib.parse import urlparse

def generate_mapping_from_csv(input_filepath: str, output_filepath: str):
    """
    脆弱性情報などが含まれるCSVファイルを読み込み、
    'package_name' と 'repo' からプロジェクトの対応表CSVを生成します。

    Args:
        input_filepath (str): 入力するCSVファイルのパス。
        output_filepath (str): 生成する対応表CSVの保存パス。
    """
    try:
        print(f"INFO: 入力ファイル '{input_filepath}' を読み込んでいます...")
        df = pd.read_csv(input_filepath, low_memory=False)

        # 必須カラムの存在チェック
        if 'package_name' not in df.columns or 'repo' not in df.columns:
            print("ERROR: CSVファイルに 'package_name' または 'repo' カラムが見つかりません。", file=sys.stderr)
            return

        # --- 1. package_name と repo のユニークな組み合わせを抽出 ---
        # 欠損データは除外する
        projects_df = df[['package_name', 'repo']].dropna().drop_duplicates().copy()
        print(f"INFO: {len(projects_df)} 件のユニークなプロジェクト/リポジトリの組み合わせが見つかりました。")

        # --- 2. リポジトリURLからディレクトリ名を推測する関数 ---
        def get_directory_name(repo_url: str) -> str:
            """
            リポジトリURLからディレクトリ名を取得する。
            例: 'https://github.com/file/file.git' -> 'file'
            """
            if not isinstance(repo_url, str):
                return None
            # URLのパス部分の最後の要素を取得
            path = urlparse(repo_url).path
            dir_name = os.path.basename(path)
            # '.git' という拡張子があれば削除
            if dir_name.endswith('.git'):
                dir_name = dir_name[:-4]
            return dir_name

        # --- 3. 各リポジトリURLに関数を適用し、新しいカラムを作成 ---
        projects_df['directory_name'] = projects_df['repo'].apply(get_directory_name)

        # --- 4. 出力用にデータフレームを整形 ---
        # カラム名を対応表のフォーマットに合わせる
        output_df = projects_df[['package_name', 'directory_name']].copy()
        output_df.rename(columns={'package_name': 'project_id'}, inplace=True)

        # ディレクトリ名が取得できなかった行や、最終的な重複を削除
        output_df.dropna(inplace=True)
        output_df.drop_duplicates(subset=['project_id', 'directory_name'], inplace=True)
        
        # プロジェクトIDでソートして見やすくする
        output_df.sort_values(by='project_id', inplace=True)

        # --- 5. 結果をCSVファイルに保存 ---
        output_df.to_csv(output_filepath, index=False)
        
        print(f"\nSUCCESS: 対応表が '{output_filepath}' に保存されました。")
        print("\n--- 生成された対応表のプレビュー ---")
        print(output_df.head().to_string())
        print("---------------------------------")


    except FileNotFoundError:
        print(f"ERROR: ファイルが見つかりません: {input_filepath}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: 予期せぬエラーが発生しました: {e}", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="OSS-FuzzのCSVなどから、プロジェクト名とディレクトリ名の対応表を自動生成します。",
        epilog="使用例: python create_mapping.py your_input_data.csv -o project_mapping.csv"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="入力する脆弱性情報のCSVファイルパス。"
    )
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=os.path.join(here, "project_mapping.csv"),
        help="出力する対応表ファイルの名前 (既定: mapping/project_mapping.csv)"
    )

    args = parser.parse_args()

    generate_mapping_from_csv(input_filepath=args.input_file, output_filepath=args.output)
