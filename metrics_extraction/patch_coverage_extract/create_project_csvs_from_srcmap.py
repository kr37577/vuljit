import os
import json
import csv
from pathlib import Path
from collections import defaultdict
import argparse

# --- 設定 (ENV/引数で上書き可能) ---
DEFAULT_OUTPUT_PREFIX = 'revisions'


def main():
    parser = argparse.ArgumentParser(description='Create per-project revision CSVs from srcmap JSONs.')
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent  # vuljit/metrics_extraction
    default_src_root = os.environ.get('VULJIT_SRCDOWN_DIR', str(repo_root / 'data' / 'srcmap_json'))
    default_out = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_out = os.path.join(default_out, 'patch_coverage', 'csv_results')

    parser.add_argument('--root', default=default_src_root, help='Root directory containing <project>/json/<date>.json')
    parser.add_argument('--out', default=default_out, help='Output directory for per-project revisions_*.csv')
    parser.add_argument('--prefix', default=DEFAULT_OUTPUT_PREFIX, help='Output filename prefix (default: revisions)')
    args = parser.parse_args()
    """
    指定されたディレクトリ構造を探索し、プロジェクトごとに
    関連する全リポジトリのリビジョン情報をCSVに出力します。
    結果は指定されたディレクトリに保存されます。
    """
    # プロジェクトごとにデータを格納する辞書
    # 例: {'arrow': [['20200117', 'arrow', 'https://...', 'e40517...'], ...]}
    project_data = defaultdict(list)

    # Pathオブジェクトでパスを定義
    root_path = Path(args.root)
    output_path = Path(args.out)

    # ルートディレクトリが存在しない場合はエラー
    if not root_path.is_dir():
        print(f"エラー: 探索対象のディレクトリ '{root_path}' が見つかりません。")
        return

    # --- 結果を保存するディレクトリを作成 ---
    try:
        output_path.mkdir(exist_ok=True)
        print(f"結果は '{output_path}' ディレクトリに保存されます。")
    except OSError as e:
        print(f"エラー: 出力ディレクトリ '{OUTPUT_DIRECTORY}' の作成に失敗しました: {e}")
        return

    print(f"'{root_path}' ディレクトリを探索しています...")

    # .jsonファイルを再帰的に探索
    for json_path in root_path.glob('*/json/*.json'):
        try:
            # パスからプロジェクト名と日付を取得
            project_name = json_path.parent.parent.name
            date = json_path.stem

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSON内のすべてのリポジトリ情報をループ
            for src_path, info in data.items():
                if isinstance(info, dict) and 'rev' in info and 'url' in info:
                    # リポジトリ名、URL、リビジョンを取得
                    repo_name = src_path.replace('/src/', '')
                    url = info['url']
                    revision = info['rev']
                    
                    # プロジェクトデータに追加
                    project_data[project_name].append([date, repo_name, url, revision])
                else:
                    print(f"警告: ファイル '{json_path}' のキー '{src_path}' に 'rev' または 'url' が見つかりません。")

        except json.JSONDecodeError:
            print(f"エラー: ファイル '{json_path}' は不正なJSON形式です。")
        except Exception as e:
            print(f"予期せぬエラーが発生しました ({json_path}): {e}")

    # データがなければ終了
    if not project_data:
        print("処理対象のデータが見つかりませんでした。")
        return

    print("\nCSVファイルを生成しています...")

    # プロジェクトごとにCSVファイルへ書き込む
    for project_name, records in project_data.items():
        # 出力ファイル名を生成し、出力ディレクトリのパスと結合
        file_name = f"{args.prefix}_{project_name}.csv"
        output_csv_path = output_path / file_name

        try:
            # 日付(records[0])、リポジトリ名(records[1])でソート
            records.sort(key=lambda x: (x[0], x[1]))

            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # ヘッダー行を書き込み
                writer.writerow(['date', 'repo_name', 'url', 'revision'])
                # データ行を書き込み
                writer.writerows(records)

            print(f"  - '{output_csv_path}' を生成しました ({len(records)}件)。")

        except IOError as e:
            print(f"エラー: ファイル '{output_csv_path}' の書き込みに失敗しました: {e}")

    print(f"\n処理が完了しました。{len(project_data)}個のプロジェクトについてCSVファイルを生成しました。")


if __name__ == '__main__':
    main()
