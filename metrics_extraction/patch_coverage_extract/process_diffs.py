import os
import pandas as pd
from pathlib import Path
import git

# (CODE_FILE_EXTENSIONS はこのスクリプトでは未使用のためコメントアウト)
# CODE_FILE_EXTENSIONS = [
#     '.c', '.cc', '.cpp', '.cxx', '.c++',
#     '.h', '.hh', '.hpp', '.hxx', '.h++'
# ]

# --- 設定 ---
SOURCE_CSV_DIRECTORY = '/work/riku-ka/prediction/csv_results'
CLONED_REPOS_DIRECTORY = '/work/riku-ka/metrics_culculator/cloned_repos'
OUTPUT_DIRECTORY = '/work/riku-ka/patch_coverage_culculater/revision_with_commit_date'
# ----------------

def get_repo_dir_name_from_url(url: str) -> str:
    """
    GitのURLから、'.git'を除いたリポジトリ名を抽出する。
    """
    if not isinstance(url, str) or not url:
        return ""
    return url.split('/')[-1].replace('.git', '')


def get_commit_date(repo_path: Path, revision: str) -> str:
    """
    指定されたリポジトリとリビジョンから、コミットのauthor dateを取得する。
    """
    try:
        repo = git.Repo(str(repo_path))
        commit = repo.commit(revision)
        return commit.authored_datetime.isoformat()
    except git.exc.NoSuchPathError:
        return "REPO_NOT_FOUND"
    except git.exc.BadName:
        return "REVISION_NOT_FOUND"
    except Exception as e:
        print(f"  - 警告: {repo_path} で予期せぬエラー: {e}")
        return "UNKNOWN_ERROR"


def main():
    """
    メイン処理：各プロジェクトのCSVを読み込み、フィルタリングしてから
    リビジョンのコミット日を追記した新しいCSVを生成する。
    """
    source_path = Path(SOURCE_CSV_DIRECTORY)
    repos_path = Path(CLONED_REPOS_DIRECTORY)
    output_path = Path(OUTPUT_DIRECTORY)

    if not source_path.is_dir():
        print(f"エラー: 入力ディレクトリ '{source_path}' が見つかりません。")
        return
    if not repos_path.is_dir():
        print(f"エラー: リポジトリディレクトリ '{repos_path}' が見つかりません。")
        return
    
    output_path.mkdir(exist_ok=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")

    csv_files = list(source_path.glob('revisions_*.csv'))
    if not csv_files:
        print(f"'{source_path}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return

    print(f"{len(csv_files)}個のプロジェクトファイルを処理します...")

    for csv_file in csv_files:
        # ファイル名から基準となるプロジェクト名を取得 (例: revisions_afl.csv -> afl)
        project_name = csv_file.stem.replace('revisions_', '')
        print(f"\n プロジェクト '{project_name}' の処理を開始 (ファイル: {csv_file.name})...")

        try:
            df = pd.read_csv(csv_file)
            
            if 'url' not in df.columns or 'revision' not in df.columns:
                print(f"  - 警告: '{csv_file.name}' に 'url' または 'revision' 列がありません。スキップします。")
                continue

            # ======================================================================
            # ▼▼▼ 変更点：プロジェクト名でのフィルタリング処理を追加 ▼▼▼
            # ======================================================================
            if 'repo_name' not in df.columns:
                print(f"  - 警告: フィルタリングに必要な 'repo_name' 列がありません。'{csv_file.name}' をスキップします。")
                continue

            original_row_count = len(df)
            # 'repo_name' 列がファイル名から取得したプロジェクト名と一致する行のみを抽出
            df = df[df['repo_name'] == project_name].copy()
            
            print(f"  - フィルタリング実行: {original_row_count}行 -> {len(df)}行 (プロジェクト名: '{project_name}')")

            # フィルタリングの結果、処理対象のデータがなくなった場合は次のファイルへ
            if df.empty:
                print("  - フィルタリングの結果、対象データが0件になりました。スキップします。")
                continue
            # ======================================================================
            # ▲▲▲ 変更点ここまで ▲▲▲
            # ======================================================================

            # コミット日を取得して新しい列として追加
            print("  - コミット日の取得を開始...")
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
            
            print(f" 成功: '{output_csv_path}' を生成しました。")

        except Exception as e:
            print(f"  -  エラー: '{csv_file.name}' の処理中に予期せぬエラーが発生しました: {e}")

    print("\nすべての処理が完了しました。")


if __name__ == '__main__':
    main()