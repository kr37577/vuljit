import os
import pandas as pd
from pathlib import Path
import git
import argparse
from typing import Dict, Optional

# --- 設定 (ENV/引数で上書き可能) ---

def get_repo_dir_name_from_url(url: str) -> str:
    """
    GitのURLから、'.git'を除いたリポジトリ名を抽出する。
    例: 'https://github.com/AFLplusplus/AFLplusplus.git' -> 'AFLplusplus'
    """
    if not isinstance(url, str) or not url:
        return ""
    return url.split('/')[-1].replace('.git', '')


def get_commit_date(repo_path: Path, revision: str) -> str:
    """
    指定されたリポジトリとリビジョンから、GitPythonを使用して
    コミットのauthor dateをISO 8601形式で取得する。
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


def append_commit_dates(source_dir: Path | str,
                        repos_dir: Path | str,
                        out_dir: Path | str) -> Dict[str, int]:
    """
    revisions_<project>.csv にコミット日時を追加し revisions_with_commit_date_<project>.csv を生成する。
    生成件数をプロジェクト名→レコード数で返す。
    """
    source_path = Path(source_dir)
    repos_path = Path(repos_dir)
    output_path = Path(out_dir)

    if not source_path.is_dir():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {source_path}")
    if not repos_path.is_dir():
        raise FileNotFoundError(f"リポジトリディレクトリが見つかりません: {repos_path}")

    output_path.mkdir(exist_ok=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")

    csv_files = list(source_path.glob('revisions_*.csv'))
    if not csv_files:
        print(f"'{source_path}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return {}

    print(f"{len(csv_files)}個のプロジェクトファイルを処理します...")
    stats: Dict[str, int] = {}

    for csv_file in csv_files:
        project_name = csv_file.stem.replace('revisions_', '')
        print(f"\n▶️  プロジェクト '{project_name}' の処理を開始...")

        try:
            df = pd.read_csv(csv_file)

            if 'url' not in df.columns or 'revision' not in df.columns:
                print(f"  - 警告: '{csv_file}' に 'url' または 'revision' 列がありません。スキップします。")
                continue

            df['commit_date'] = df.apply(
                lambda row: get_commit_date(
                    repos_path / get_repo_dir_name_from_url(row['url']),
                    row['revision']
                ),
                axis=1
            )

            # commit_date を UTC ISO8601 フォーマットに整形する
            commit_series = df['commit_date'].astype(str).str.replace(' ', 'T', regex=False)
            dt_parsed = pd.to_datetime(commit_series, errors='coerce', utc=True)
            formatted = dt_parsed.dt.strftime('%Y-%m-%dT%H:%M:%S%z')
            formatted = formatted.str.replace(r'([+-]\d{2})(\d{2})$', r'\1:\2', regex=True)
            df.loc[~dt_parsed.isna(), 'commit_date'] = formatted[~dt_parsed.isna()]

            output_csv_path = output_path / f"revisions_with_commit_date_{project_name}.csv"
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

            stats[project_name] = len(df)
            print(f" 成功: '{output_csv_path}' を生成しました ({len(df)}件)。")

        except Exception as e:
            print(f"  -  エラー: '{csv_file}' の処理中に予期せぬエラーが発生しました: {e}")

    print("\nすべての処理が完了しました。")
    return stats


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Append commit date to revision CSVs created from srcmap.')
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent.parent
    default_src = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_src = os.path.join(default_src, 'patch_coverage', 'csv_results')
    default_repos = os.environ.get('VULJIT_CLONED_REPOS_DIR', str(repo_root / 'data' / 'intermediate' / 'cloned_repos'))
    default_out = os.environ.get('VULJIT_PATCH_COVERAGE_INPUTS_DIR')
    if not default_out:
        default_out = str(repo_root / 'datasets' / 'derived_artifacts' / 'patch_coverage_inputs')

    parser.add_argument('--src', default=default_src, help='Directory containing revisions_*.csv files')
    parser.add_argument('--repos', default=default_repos, help='Directory containing cloned repositories')
    parser.add_argument('--out', default=default_out, help='Output directory for revisions_with_commit_date_*.csv')
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        append_commit_dates(args.src, args.repos, args.out)
    except FileNotFoundError as e:
        print(f"エラー: {e}")


if __name__ == '__main__':
    main()
