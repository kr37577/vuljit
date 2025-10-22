import os
import pandas as pd
from pathlib import Path
import git
import argparse

CODE_FILE_EXTENSIONS = [
    '.c', '.cc', '.cpp', '.cxx', '.c++',  # C/C++ ソースファイル
    '.h', '.hh', '.hpp', '.hxx', '.h++'   # C/C++ ヘッダーファイル
]

# --- 設定 (ENV/引数で上書き可能) ---

def get_repo_dir_name_from_url(url: str) -> str:
    """
    GitのURLから、'.git'を除いたリポジトリ名を抽出する。
    例: 'https://github.com/AFLplusplus/AFLplusplus.git' -> 'AFLplusplus'
    """
    if not isinstance(url, str) or not url:
        return ""
    # URLの最後の部分を取得し、'.git'を削除
    return url.split('/')[-1].replace('.git', '')


def get_commit_date(repo_path: Path, revision: str) -> str:
    """
    指定されたリポジトリとリビジョンから、GitPythonを使用して
    コミットのauthor dateをISO 8601形式で取得する。
    """
    try:
        # リポジトリオブジェクトを初期化
        repo = git.Repo(str(repo_path))
        # 指定されたリビジョン（コミットハッシュ）からコミットオブジェクトを取得
        commit = repo.commit(revision)
        # コミットのauthor dateをdatetimeオブジェクトとして取得し、ISO 8601形式の文字列に変換
        return commit.authored_datetime.isoformat()
    except git.exc.NoSuchPathError:
        # 指定されたパスにリポジトリが存在しない場合
        return "REPO_NOT_FOUND"
    except git.exc.BadName:
        # 指定されたリビジョンがリポジトリ内に存在しない場合
        return "REVISION_NOT_FOUND"
    except Exception as e:
        # その他の予期せぬエラー
        print(f"  - 警告: {repo_path} で予期せぬエラー: {e}")
        return "UNKNOWN_ERROR"


def main():
    parser = argparse.ArgumentParser(description='Append commit date to revision CSVs created from srcmap.')
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    default_src = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_src = os.path.join(default_src, 'patch_coverage', 'csv_results')
    default_repos = os.environ.get('VULJIT_CLONED_REPOS_DIR', str(repo_root / 'data' / 'intermediate' / 'cloned_repos'))
    default_out = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_out = os.path.join(default_out, 'patch_coverage', 'revision_with_commit_date')

    parser.add_argument('--src', default=default_src, help='Directory containing revisions_*.csv files')
    parser.add_argument('--repos', default=default_repos, help='Directory containing cloned repositories')
    parser.add_argument('--out', default=default_out, help='Output directory for revisions_with_commit_date_*.csv')
    args = parser.parse_args()
    """
    メイン処理：各プロジェクトのCSVを読み込み、リビジョンのコミット日を追記した新しいCSVを生成する。
    """
    source_path = Path(args.src)
    repos_path = Path(args.repos)
    output_path = Path(args.out)

    # --- ディレクトリの存在チェックと作成 ---
    if not source_path.is_dir():
        print(f"エラー: 入力ディレクトリ '{source_path}' が見つかりません。")
        return
    if not repos_path.is_dir():
        print(f"エラー: リポジトリディレクトリ '{repos_path}' が見つかりません。")
        return
    
    output_path.mkdir(exist_ok=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")

    # --- CSVファイルの処理 ---
    csv_files = list(source_path.glob('revisions_*.csv'))
    if not csv_files:
        print(f"'{source_path}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return

    print(f"{len(csv_files)}個のプロジェクトファイルを処理します...")

    for csv_file in csv_files:
        project_name = csv_file.stem.replace('revisions_', '')
        print(f"\n▶️  プロジェクト '{project_name}' の処理を開始...")

        try:
            df = pd.read_csv(csv_file)
            
            # 'url' と 'revision' 列が存在するかチェック
            if 'url' not in df.columns or 'revision' not in df.columns:
                print(f"  - 警告: '{csv_file}' に 'url' または 'revision' 列がありません。スキップします。")
                continue

            # コミット日を取得して新しい列として追加
            # urlからローカルのリポジトリ名を生成してパスを構築する
            df['commit_date'] = df.apply(
                lambda row: get_commit_date(
                    repos_path / get_repo_dir_name_from_url(row['url']), 
                    row['revision']
                ),
                axis=1
            )
            
            # 結果を新しいCSVファイルに保存
            output_csv_path = output_path / f"revisions_with_commit_date_{project_name}.csv"
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            
            print(f" 成功: '{output_csv_path}' を生成しました ({len(df)}件)。")

        except Exception as e:
            print(f"  -  エラー: '{csv_file}' の処理中に予期せぬエラーが発生しました: {e}")

    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    main()
