import os
import pandas as pd
import argparse

def filter_projects(mapping_file, project_list_file, output_file):
    """
    マッピングファイル内のproject_idを、指定されたテキストファイルのリストに基づいてフィルタリングします。

    Args:
        mapping_file (str): project_idを含むCSVファイルへのパス。
        project_list_file (str): フィルタリングに使用するプロジェクトIDのリストを含むテキストファイルへのパス。
        output_file (str): フィルタリングされた結果を保存するCSVファイルへのパス。
    """
    try:
        # フィルタリング対象のプロジェクトIDをテキストファイルから読み込み、セットに格納
        with open(project_list_file, 'r') as f:
            # 各行の改行文字や空白を削除
            allowed_projects = {line.strip() for line in f}
        print(f"'{project_list_file}' から {len(allowed_projects)} 件のプロジェクトIDを読み込みました。")

        # マッピングファイルをDataFrameとして読み込み
        mapping_df = pd.read_csv(mapping_file)
        print(f"'{mapping_file}' から {len(mapping_df)} 件のレコードを読み込みました。")

        # project_idが許可リストに含まれる行のみをフィルタリング
        filtered_df = mapping_df[mapping_df['project_id'].isin(allowed_projects)]
        print(f"フィルタリング後のレコード数: {len(filtered_df)} 件")

        # 結果を新しいCSVファイルに保存（インデックスは含めない）
        filtered_df.to_csv(output_file, index=False)
        print(f"フィルタリングされたマッピングを '{output_file}' に保存しました。")

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == '__main__':
    # コマンドライン引数を設定
    parser = argparse.ArgumentParser(description='プロジェクトマッピングファイルをテキストファイルのリストでフィルタリングします。')
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('-m', '--mapping', default=os.path.join(here, 'project_mapping.csv'), help='入力マッピングCSVファイル')
    parser.add_argument('-p', '--projects', default=os.path.join(here, 'c_cpp_projects.txt'), help='プロジェクトIDリストのテキストファイル')
    parser.add_argument('-o', '--output', default=os.path.join(here, 'filtered_project_mapping.csv'), help='出力CSVファイル')

    args = parser.parse_args()

    # 関数を実行
    filter_projects(args.mapping, args.projects, args.output)
