import os
import pandas as pd
from pathlib import Path
import git
import argparse

"""
Args and env defaults are used instead of hard-coded absolute paths.
"""

# 4. 差分抽出の対象とするファイルの拡張子リスト
CODE_FILE_EXTENSIONS = [
    '.c', '.cc', '.cpp', '.cxx', '.c++',
    '.h', '.hh','hpp', '.hxx', '.h++'
]
# ----------------

def get_repo_dir_name_from_url(url: str) -> str:
    """
    GitのURLから、'.git'を除いたリポジトリ名を抽出する。
    """
    if not isinstance(url, str) or not url:
        return ""
    return url.split('/')[-1].replace('.git', '')

def get_changed_files(repo_path: Path, old_revision: str, new_revision: str, extensions: list) -> list:
    """
    2つのリビジョン間の差分をとり、指定された拡張子に一致するファイルパスのリストを返す。
    """
    changed_files = []
    try:
        repo = git.Repo(str(repo_path))
        # 2つのコミットオブジェクトを取得
        commit1 = repo.commit(old_revision)
        commit2 = repo.commit(new_revision)

        # 差分を取得
        diff_index = commit1.diff(commit2)

        # 拡張子が CODE_FILE_EXTENSIONSで終わるファイルのみを抽出
        target_extensions = tuple(extensions)
        for diff_item in diff_index:
            # 変更前と変更後のパスを取得。新規追加や削除も考慮。
            file_path = diff_item.b_path or diff_item.a_path
            if file_path and file_path.lower().endswith(target_extensions):
                changed_files.append(file_path)

        return changed_files

    except git.exc.NoSuchPathError:
        print(f"  - エラー: リポジトリパスが見つかりません: {repo_path}")
        return []
    except git.exc.BadName as e:
        print(f"  - エラー: 不正なリビジョン: {e}")
        return []
    except Exception as e:
        print(f"  - 警告: 差分取得中に予期せぬエラーが発生しました: {e}")
        return []


def save_patches(repo_path: Path, old_revision: str, new_revision: str, output_dir: Path, extensions: list) -> int:
    """
    2つのリビジョン間の差分をファイルごとのパッチファイルとして保存する。
    保存されたパッチファイルの数を返す。
    """
    try:
        repo = git.Repo(str(repo_path))
        commit1 = repo.commit(old_revision)
        commit2 = repo.commit(new_revision)
        diff_index = commit1.diff(commit2, create_patch=True) # パッチを生成するために create_patch=True を指定

        target_extensions = tuple(extensions)
        saved_patch_count = 0

        for diff_item in diff_index:
            file_path_str = diff_item.b_path or diff_item.a_path
            if file_path_str and file_path_str.lower().endswith(target_extensions):
                # パッチ内容を取得 (バイナリデータの場合があるのでデコードする)
                try:
                    # diffがNoneの場合や空の場合はスキップ
                    if diff_item.diff is None:
                        continue
                    patch_content = diff_item.diff.decode('utf-8')
                    if not patch_content.strip():
                        continue
                except (UnicodeDecodeError, AttributeError):
                    # デコードできないバイナリファイルの差分などはスキップ
                    continue

                # 保存先パスを作成 (例: {output_dir}/src/foo.c.patch)
                # ファイルパスに / が含まれている場合を考慮
                patch_file_path = output_dir.joinpath(file_path_str + ".patch")

                # 保存先ディレクトリが存在しない場合は作成
                patch_file_path.parent.mkdir(parents=True, exist_ok=True)

                # パッチをファイルに書き込む
                with open(patch_file_path, 'w', encoding='utf-8') as f:
                    f.write(patch_content)

                saved_patch_count += 1

        return saved_patch_count

    except git.exc.NoSuchPathError:
        print(f"  - エラー (save_patches): リポジトリパスが見つかりません: {repo_path}")
        return 0
    except git.exc.BadName as e:
        print(f"  - エラー (save_patches): 不正なリビジョン: {e}")
        return 0
    except Exception as e:
        print(f"  - 警告 (save_patches): パッチ保存中に予期せぬエラーが発生しました: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Create daily diffs and patch files per project and date.')
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    default_src = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_src = os.path.join(default_src, 'patch_coverage', 'revision_with_commit_date')
    default_repos = os.environ.get('VULJIT_CLONED_REPOS_DIR', str(repo_root / 'data' / 'intermediate' / 'cloned_repos'))
    default_out = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_out = os.path.join(default_out, 'patch_coverage', 'daily_diffs')

    parser.add_argument('--src', default=default_src, help='Directory containing revisions_with_commit_date_*.csv')
    parser.add_argument('--repos', default=default_repos, help='Directory containing cloned repositories')
    parser.add_argument('--out', default=default_out, help='Output directory for daily diffs and patches')
    args = parser.parse_args()
    """
    メイン処理：コミット日付きCSVを読み込み、日毎の差分ファイルリストを生成する。
    """
    input_path = Path(args.src)
    repos_path = Path(args.repos)
    output_path = Path(args.out)

    if not input_path.is_dir():
        print(f"エラー: 入力ディレクトリ '{input_path}' が見つかりません。")
        return
    if not repos_path.is_dir():
        print(f"エラー: リポジトリディレクトリ '{repos_path}' が見つかりません。")
        return
    
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")

    csv_files = list(input_path.glob('revisions_with_commit_date_*.csv'))
    if not csv_files:
        print(f"'{input_path}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return

    print(f"{len(csv_files)}個のプロジェクトファイルを処理します...")

    for csv_file in csv_files:
        project_name = csv_file.stem.replace('revisions_with_commit_date_', '')
        print(f"\n▶ プロジェクト '{project_name}' の処理を開始...")

        try:
            df = pd.read_csv(csv_file)
            
            # 必要な列が存在するか確認
            required_cols = ['date', 'url', 'revision']
            if not all(col in df.columns for col in required_cols):
                print(f"  - 警告: '{csv_file.name}' に必要な列がありません。スキップします。")
                continue
            
            # 日付でソートされていることを保証する (重要)
            df = df.sort_values(by='date').reset_index(drop=True)

            if len(df) < 2:
                print("  - 差分を取得するにはデータが2行未満です。スキップします。")
                continue

            # プロジェクトごとの出力ディレクトリを作成
            project_output_path = output_path / project_name
            project_output_path.mkdir(exist_ok=True)
            
            print(f"  - {len(df)-1} 日分の差分を処理します。")

            # 2行目からループを開始し、前の行との差分を取得
            for i in range(1, len(df)):
                previous_row = df.iloc[i-1]
                current_row = df.iloc[i]

                # Gitリポジトリのローカルパスを取得
                repo_url = current_row['url']
                repo_dir_name = get_repo_dir_name_from_url(repo_url)
                repo_local_path = repos_path / repo_dir_name

                # 差分のあるファイルリストを取得
                changed_files = get_changed_files(
                    repo_local_path,
                    previous_row['revision'],
                    current_row['revision'],
                    CODE_FILE_EXTENSIONS
                )

                if changed_files:
                    # 今日の日付でCSVファイルを作成
                    output_csv_name = f"{current_row['date']}.csv"
                    output_csv_path = project_output_path / output_csv_name
                    
                    # DataFrameを作成してCSVに保存
                    diff_df = pd.DataFrame(changed_files, columns=['changed_file_path'])
                    diff_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
                    print(f"    - {current_row['date']}: {len(changed_files)}個の変更ファイルをリスト '{output_csv_path.name}' に保存しました。")

                    # パッチファイルを保存するディレクトリを作成
                    patch_output_dir = project_output_path / f"{current_row['date']}_patches"
                    patch_output_dir.mkdir(exist_ok=True)

                    # パッチファイルを保存
                    saved_patch_count = save_patches(
                        repo_local_path,
                        previous_row['revision'],
                        current_row['revision'],
                        patch_output_dir,
                        CODE_FILE_EXTENSIONS
                    )
                    if saved_patch_count > 0:
                        print(f"    - {current_row['date']}: {saved_patch_count}個のパッチファイルを '{patch_output_dir.name}/' に保存しました。")

                else:
                    print(f"    - {current_row['date']}: 対象拡張子の変更ファイルはありませんでした。")

            print(f"✔ プロジェクト '{project_name}' の処理が完了しました。")

        except Exception as e:
            print(f"  - エラー: '{csv_file.name}' の処理中に予期せぬエラーが発生しました: {e}")

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    main()
