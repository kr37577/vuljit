import os
import csv
from collections import defaultdict
from urllib.parse import urlparse
from pathlib import Path
import argparse


def process_and_sort_repos_from_file(csv_filepath):
    """
    CSVファイルを処理し、リポジトリごとのintroduced_commits数をカウントし、
    多い順にソートして変換されたリポジトリ名のリストを返します。

    Args:
        csv_filepath (str): CSVファイルのパス。

    Returns:
        list: カウントが多い順にソートされた、変換済みのリポジトリ名リスト。
              エラーが発生した場合は空のリストを返します。
    """
    repo_commits_map = defaultdict(set)

    try:
        with open(csv_filepath, 'r', encoding='utf-8', newline='') as csv_file:
            reader = csv.reader(csv_file)
            try:
                header = next(reader)  # ヘッダー行を読み飛ばす
            except StopIteration:
                print(f"エラー: CSVファイル '{csv_filepath}' が空か、ヘッダー行がありません。")
                return []

            try:
                repo_col_index = header.index("repo")
                introduced_commits_col_index = header.index(
                    "introduced_commits")
            except ValueError:
                print(
                    f"エラー: CSVヘッダーに 'repo' または 'introduced_commits' 列が見つかりません。ファイル: {csv_filepath}")
                return []

            for i, row in enumerate(reader, start=1):  # start=1 でヘッダー後の行番号
                if not row:  # 空行をスキップ
                    print(f"警告: {csv_filepath} の {i+1}行目は空行です。スキップします。")
                    continue
                try:
                    repo_url = row[repo_col_index]
                    introduced_commit = row[introduced_commits_col_index]

                    if introduced_commit:  # introduced_commitが空でない場合のみ処理
                        repo_commits_map[repo_url].add(introduced_commit)
                except IndexError:
                    print(f"警告: {csv_filepath} の {i+1}行目で列数が不足しています: {row}")
                    continue
    except FileNotFoundError:
        print(f"エラー: CSVファイル '{csv_filepath}' が見つかりません。")
        return []
    except IOError as e:
        print(f"エラー: CSVファイル '{csv_filepath}' の読み込み中にエラーが発生しました: {e}")
        return []
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return []

    # introduced_commitsのユニーク数でソート
    sorted_repos_with_counts = sorted(
        [(repo, len(commits)) for repo, commits in repo_commits_map.items()],
        key=lambda item: item[1],
        reverse=True
    )

    transformed_repo_names = []
    transformed_repo_commits_count = 0
    for repo_url, count in sorted_repos_with_counts:
        repo_url = repo_url.strip()  # 前後の空白を削除
        if repo_url.startswith("https://github.com/") or 'github.com' in repo_url:
            transformed_name = repo_url.replace(
                "https://github.com/", "").replace(".git", "").replace("git://github.com/", "")
            # 特殊ケース: ghostpdl のみ owner を ArtifexSoftware に固定
            if transformed_name.endswith("/ghostpdl") or transformed_name == "ghostpdl":
                transformed_name = "ArtifexSoftware/ghostpdl"
            # リクエストの形式: owner/name
            transformed_repo_names.append(transformed_name)
            transformed_repo_commits_count += count
            # カウントも表示する場合の形式: owner/name -> count
            # transformed_repo_names.append(f"{transformed_name} -> {count}")
            # 変換できない場合は元のURLとカウントを表示
            # transformed_repo_names.append(repo_url) # リポジトリ名のみの場合
            # transformed_repo_names.append(f"{repo_url} (変換不可) -> {count}")

        elif repo_url.startswith("git://"):
            try:
                parsed_url = urlparse(repo_url)
                # パス部分から先頭のスラッシュと末尾の.gitを削除
                path = parsed_url.path.lstrip('/').replace('.git', '')
                if path:
                    # 特殊ケース: ghostpdl のみ owner を ArtifexSoftware に固定
                    if path.endswith("/ghostpdl") or path == "ghostpdl":
                        path = "ArtifexSoftware/ghostpdl"
                    transformed_repo_names.append(path)
                    transformed_repo_commits_count += count
                else:
                    print(f"警告: URLからリポジトリ名を抽出できませんでした: {repo_url}")
            except Exception as e:
                print(f"警告: URLの解析中にエラーが発生しました: {repo_url}, エラー: {e}")

    print(f"処理完了: {len(transformed_repo_names)} 個のリポジトリ名が変換されました。")
    print(f"合計: {transformed_repo_commits_count} "
          "個のintroduced_commitsがカウントされました。")
    return transformed_repo_names


def save_list_to_file(data_list, filename):
    """
    文字列のリストをテキストファイルに保存します。各要素は新しい行に書き込まれます。

    Args:
        data_list (list): 保存する文字列のリスト。
        filename (str): 保存先のファイル名。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(f"{item}\n")
        print(f"データは正常に '{filename}' に保存されました。")
    except IOError:
        print(f"エラー: ファイル '{filename}' への書き込み中に問題が発生しました。")


# --- メイン処理 ---
if __name__ == "__main__":
    # 引数/環境変数/相対パスに統一
    parser = argparse.ArgumentParser(description='OSV由来CSVからowner/name形式のGitHubリポジトリ一覧を抽出')
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # vuljit/
    default_csv = os.environ.get('VULJIT_VUL_CSV', str(repo_root / 'rq3_dataset' / 'oss_fuzz_vulns_2025802.csv'))
    default_out = os.environ.get('VULJIT_PROJECTS_TXT', str(repo_root / 'metrics_extraction' / 'projects.txt'))
    parser.add_argument('--csv', default=default_csv, help='入力CSVパス（repo,introduced_commits列を含む）')
    parser.add_argument('--out', default=default_out, help='出力テキストファイルパス')
    args = parser.parse_args()

    sorted_transformed_repos = process_and_sort_repos_from_file(args.csv)

    if sorted_transformed_repos:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_list_to_file(sorted_transformed_repos, str(out_path))
        print(f"\n'{args.csv}' の処理結果 ({out_path} に保存):")
        for repo_info in sorted_transformed_repos:
            print(repo_info)
    else:
        print(f"処理するデータがなかったか、エラーにより処理が中断されました。'{args.out}' は作成されません。")
