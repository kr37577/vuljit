import argparse
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from google.cloud import storage

from calculate_patch_coverage_per_project import (
    DEFAULT_OUTPUT_BASE_DIRECTORY,
    compute_patch_coverage_for_patch_text,
)
from prepare_patch_coverage_inputs import (
    load_canonical_repo_map,
    load_repo_name_overrides,
)

# create_daily_diff.py の設定と揃える
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_MAIN_REPO_MAP = (
    REPO_ROOT / "datasets" / "derived_artifacts" / "oss_fuzz_metadata" / "c_cpp_vulnerability_summary.csv"
)

DEFAULT_INPUT_CSV_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_INPUTS_DIR",
        REPO_ROOT / "datasets" / "derived_artifacts" / "patch_coverage_inputs",
    )
)
DEFAULT_CLONED_REPOS_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_CLONED_REPOS_DIR",
        REPO_ROOT / "datasets" / "raw" / "cloned_c_cpp_projects",
    )
)

# 差分対象の拡張子リスト（create_daily_diff.py と同一）
CODE_FILE_EXTENSIONS: Tuple[str, ...] = (
    '.c', '.cc', '.cpp', '.cxx', '.c++',
    '.h', '.hh', '.hpp', '.hxx', '.h++'
)

_WORKER_STORAGE_CLIENT: Optional[storage.Client] = None


def _initialize_worker_storage_client() -> None:
    global _WORKER_STORAGE_CLIENT
    if _WORKER_STORAGE_CLIENT is not None:
        return
    try:
        _WORKER_STORAGE_CLIENT = storage.Client.create_anonymous_client()
    except Exception as e:
        _WORKER_STORAGE_CLIENT = None
        print(f"  - 警告: ワーカー初期化時に GCS クライアント生成に失敗しました: {e}")


def get_repo_dir_name_from_url(url: str) -> str:
    if not isinstance(url, str) or not url:
        return ""
    return url.split('/')[-1].replace('.git', '')


def get_changed_files(repo_path: Path, old_revision: str, new_revision: str, extensions: Iterable[str]) -> List[str]:
    try:
        if not repo_path.is_dir():
            print(f"  - エラー: リポジトリパスが見つかりません: {repo_path}")
            return []

        cmd = [
            'git', '-C', str(repo_path), 'diff', '--name-only',
            str(old_revision), str(new_revision)
        ]
        env = os.environ.copy()
        env.setdefault('GIT_OPTIONAL_LOCKS', '0')
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, env=env)
        names = [line.strip() for line in out.decode('utf-8', errors='ignore').splitlines() if line.strip()]

        target_extensions = tuple(ext.lower() for ext in extensions)
        files = [p for p in names if p.lower().endswith(target_extensions)]
        return files
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"  - 警告: 差分取得中に予期せぬエラーが発生しました: {e}")
        return []


def get_patch_text(repo_path: Path, old_revision: str, new_revision: str, rel_path: str) -> Optional[str]:
    try:
        cmd = [
            'git', '-C', str(repo_path), 'diff',
            str(old_revision), str(new_revision), '--', rel_path
        ]
        env = os.environ.copy()
        env.setdefault('GIT_OPTIONAL_LOCKS', '0')
        patch = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, env=env)
        patch_text = patch.decode('utf-8', errors='ignore')
        if not patch_text.strip():
            return None

        lines = patch_text.splitlines()
        start_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if line.startswith('@@'):
                start_idx = i
                break

        if start_idx is None:
            print(f"    - 警告: 差分にhunkが含まれていないため全体を使用します: {rel_path}")
            return patch_text if patch_text.endswith("\n") else patch_text + "\n"

        return "\n".join(lines[start_idx:]) + "\n"
    except subprocess.CalledProcessError:
        return None
    except Exception as e:
        print(f"  - 警告: パッチ生成中に予期せぬエラーが発生しました ({rel_path}): {e}")
        return None


def _compute_coverage_worker(args: Tuple[str, str, str, str, Optional[str]]) -> Optional[dict]:
    project_name, date, file_path, patch_text, parsing_out = args
    parsing_root = Path(parsing_out) if parsing_out else None
    return compute_patch_coverage_for_patch_text(
        project_name=project_name,
        date=date,
        file_path_str=file_path,
        patch_text=patch_text,
        parsing_output_root=parsing_root,
        storage_client=_WORKER_STORAGE_CLIENT,
    )


def _normalize_projects(values: Sequence[str]) -> List[str]:
    tokens: List[str] = []
    for val in values:
        if not val:
            continue
        for token in re.split(r"[,\s]+", val.strip()):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def process_project(
    project_name: str,
    input_dir: Path,
    repos_dir: Path,
    output_dir: Path,
    parsing_dir: Optional[Path],
    workers: int,
    canonical_repo_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    csv_file = input_dir / f"revisions_with_commit_date_{project_name}.csv"
    if not csv_file.is_file():
        print(f"エラー: 入力CSVが見つかりません: {csv_file}")
        return

    canonical_entry = (canonical_repo_map or {}).get(project_name) if canonical_repo_map else None
    canonical_repo_name = (canonical_entry or {}).get("repo_name")
    canonical_repo_dir = (canonical_entry or {}).get("repo_dir_name")

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"エラー: '{csv_file}' の読み込みに失敗しました: {e}")
        return

    required_cols = {'date', 'url', 'revision', 'repo_name'}
    if not required_cols.issubset(df.columns):
        print(f"エラー: '{csv_file}' に必要な列 {required_cols} がありません。")
        return

    if 'project' not in df.columns:
        df['project'] = project_name
        print("  - 注意: 'project' 列が存在しないため CSV 名から補完しました。")
    else:
        missing = df['project'].isna().sum()
        if missing:
            df.loc[df['project'].isna(), 'project'] = project_name
            print(f"  - 注意: 'project' 列に {missing} 件の欠損があり補完しました。")

    original_len = len(df)
    df = df[df['project'] == project_name].copy()
    if original_len != len(df):
        sample_repo = ", ".join(sorted(set(df['repo_name']))[:5]) if not df.empty else "なし"
        print(f"  - フィルタ: {original_len}行 -> {len(df)}行 (project='{project_name}', repo例: {sample_repo})")
    if df.empty:
        print(f"  - 対象プロジェクト '{project_name}' の行がCSVに存在しません ({csv_file}).")
        return

    if canonical_repo_name:
        filtered_df = df[df['repo_name'].astype(str).str.strip() == canonical_repo_name].copy()
        removed_count = len(df) - len(filtered_df)
        if removed_count:
            print(f"  - フィルタ: canonical repo '{canonical_repo_name}' 以外の {removed_count} 行を除外しました。")
        if filtered_df.empty:
            print(f"  - 警告: canonical repo '{canonical_repo_name}' の行が存在しないためスキップします。")
            return
        if len(filtered_df) < 2:
            print(f"  - 警告: canonical repo '{canonical_repo_name}' の行が2件未満です。差分を計算できません。")
            return
        df = filtered_df

    df = df.sort_values(by='date', kind='mergesort').reset_index(drop=True)
    if len(df) < 2:
        print(f"  - データが少ないため差分を計算できません ({csv_file}).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{project_name}_patch_coverage.csv"

    parsing_dir_path = None
    if parsing_dir is not None:
        parsing_dir.mkdir(parents=True, exist_ok=True)
        parsing_dir_path = parsing_dir
    parsing_out_str = str(parsing_dir_path) if parsing_dir_path else None

    skipped_dates: set[str] = set()
    if output_file_path.exists():
        try:
            existing_df = pd.read_csv(output_file_path)
            if 'date' in existing_df.columns:
                skipped_dates = set(existing_df['date'].astype(str))
            print(f"✔ 既存の出力から {len(skipped_dates)} 件の日付をスキップ対象として読み込みました。")
        except pd.errors.EmptyDataError:
            print("⚠️ 既存の出力ファイルは空でした。")
        except Exception as e:
            print(f"⚠️ 既存の出力を読み込む際にエラーが発生しました: {e}")

    print(f"\n▶ プロジェクト '{project_name}' の処理を開始します。")

    workers = max(1, workers)
    storage_client_for_sequential: Optional[storage.Client] = None
    parallel_executor: Optional[ProcessPoolExecutor] = None

    try:
        if workers == 1:
            try:
                storage_client_for_sequential = storage.Client.create_anonymous_client()
            except Exception as e:
                print(f"  - 警告: GCSクライアントの初期化に失敗しました（逐次モード）: {e}")
        else:
            parallel_executor = ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker_storage_client,
            )

        processed_any = False

        for i in range(1, len(df)):
            previous_row = df.iloc[i - 1]
            current_row = df.iloc[i]
            date_str = str(current_row['date'])

            if date_str in skipped_dates:
                print(f"  - スキップ: 日付 '{date_str}' は既に処理済みです。")
                continue

            repo_name = str(current_row.get('repo_name') or '').strip()
            repo_dir_name = repo_name or get_repo_dir_name_from_url(str(current_row['url']))
            repo_local_name = canonical_repo_dir or repo_dir_name
            repo_local_path = repos_dir / repo_local_name

            print(f"  - 日付: {date_str}")

            changed_files = get_changed_files(
                repo_local_path,
                str(previous_row['revision']),
                str(current_row['revision']),
                CODE_FILE_EXTENSIONS,
            )

            if not changed_files:
                print("    - 差分対象のファイルが見つかりませんでした。")
                continue

            patch_records: List[Tuple[str, str]] = []
            for rel_path in changed_files:
                patch_text = get_patch_text(
                    repo_local_path,
                    str(previous_row['revision']),
                    str(current_row['revision']),
                    rel_path,
                )
                if patch_text:
                    patch_records.append((rel_path, patch_text))

            if not patch_records:
                print("    - パッチを生成できるファイルがありませんでした。")
                continue

            daily_results: List[Optional[dict]]
            if workers == 1:
                daily_results = []
                for file_path, patch_text in patch_records:
                    result = compute_patch_coverage_for_patch_text(
                        project_name=project_name,
                        date=date_str,
                        file_path_str=file_path,
                        patch_text=patch_text,
                        parsing_output_root=parsing_dir_path,
                        storage_client=storage_client_for_sequential,
                    )
                    daily_results.append(result)
            else:
                args_list = [
                    (project_name, date_str, file_path, patch_text, parsing_out_str)
                    for file_path, patch_text in patch_records
                ]
                daily_results = list(parallel_executor.map(_compute_coverage_worker, args_list))

            filtered_results = [r for r in daily_results if r]
            for r in filtered_results:
                total_added = r['total_added_lines']
                if total_added > 0:
                    print(f"    - {r['file_path']}: 追加 {total_added}行, うちカバー {r['covered_added_lines']}行 (カバレッジ: {r['patch_coverage']:.2f}%)")

            if filtered_results:
                processed_any = True
                df_result = pd.DataFrame(filtered_results)
                header = not output_file_path.exists() or output_file_path.stat().st_size == 0
                df_result.to_csv(output_file_path, mode='a', header=header, index=False, encoding='utf-8-sig')
                print(f"  ✔ 日付 '{date_str}' の結果を '{output_file_path}' に追記しました。")

        if processed_any or skipped_dates:
            print(f"\n✔ プロジェクト '{project_name}' の処理が完了しました。結果は '{output_file_path}' に保存されています。")
        else:
            print(f"プロジェクト '{project_name}' で処理対象のデータが見つかりませんでした。")
    finally:
        if parallel_executor is not None:
            parallel_executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="revisions_with_commit_date CSV を直接処理してパッチカバレッジを計算するインメモリパイプライン。"
    )
    parser.add_argument(
        "-p",
        "--project",
        dest="projects",
        action="append",
        required=True,
        help="処理対象のプロジェクト名（複数指定可、カンマ区切り可）",
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        help="revisions_with_commit_date_<project>.csv が存在するディレクトリ",
    )
    parser.add_argument(
        "--repos",
        dest="repos_dir",
        help="差分取得に使用する git クローン済みリポジトリのディレクトリ",
    )
    parser.add_argument(
        "--coverage-out",
        dest="coverage_out",
        help="パッチカバレッジCSVの出力先ベースディレクトリ",
    )
    parser.add_argument(
        "--parsing-out",
        dest="parsing_out",
        help="HTML解析結果(JSON)を保存するディレクトリ（指定しない場合は保存しません）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="カバレッジ計算を行う並列プロセス数 (既定: 4、1で逐次実行)",
    )
    parser.add_argument(
        "--main-repo-map",
        # default=str(DEFAULT_MAIN_REPO_MAP),
        default=None,
        help="プロジェクト -> canonical repo URL のCSV。'none' を指定すると無効化します。",
    )
    parser.add_argument(
        "--repo-name-overrides",
        help="canonical repo 名 -> ローカルディレクトリ名のマッピング (JSON またはCSV)",
    )
    parser.add_argument(
        "--dry-run-missing-repos",
        action="store_true",
        help="パッチカバレッジを計算せず、canonical repo が欠けているプロジェクトを確認します。",
    )
    args = parser.parse_args()

    projects = _normalize_projects(args.projects or [])
    if not projects:
        print("エラー: --project で少なくとも1つのプロジェクトを指定してください。")
        raise SystemExit(1)

    input_dir = Path(args.input_dir or DEFAULT_INPUT_CSV_DIRECTORY)
    if not input_dir.is_dir():
        print(f"エラー: 入力ディレクトリ '{input_dir}' が見つかりません。")
        raise SystemExit(1)

    repos_dir = Path(args.repos_dir or DEFAULT_CLONED_REPOS_DIRECTORY)
    if not repos_dir.is_dir():
        print(f"エラー: リポジトリディレクトリ '{repos_dir}' が見つかりません。")
        raise SystemExit(1)

    coverage_out_dir = Path(args.coverage_out or DEFAULT_OUTPUT_BASE_DIRECTORY)
    parsing_out_dir = Path(args.parsing_out) if args.parsing_out else None

    canonical_map_path = args.main_repo_map
    if canonical_map_path and canonical_map_path.lower() == "none":
        canonical_map_path = None

    repo_overrides = load_repo_name_overrides(args.repo_name_overrides)
    canonical_repo_map: Dict[str, Dict[str, str]] = {}
    if canonical_map_path:
        canonical_repo_map = load_canonical_repo_map(canonical_map_path, repo_overrides)
        if not canonical_repo_map:
            print("⚠️  canonical repo map を読み込めませんでした。フィルタリングを無効化します。")
    elif args.repo_name_overrides:
        print("⚠️  canonical repo map が無効のため、--repo-name-overrides は無視されます。")

    if args.dry_run_missing_repos:
        if not canonical_repo_map:
            print("canonical repo map が読み込めないため、欠損チェックを実行できません。")
            return
        missing = []
        for project, entry in canonical_repo_map.items():
            repo_dir = repos_dir / entry['repo_dir_name']
            if not repo_dir.is_dir():
                missing.append((project, entry['repo_dir_name'], entry['main_repo']))
        if not missing:
            print("✔ すべての canonical repo がローカルに存在します。")
        else:
            print("欠損している canonical repo:")
            for project, dir_name, repo_url in missing:
                print(f"  - {project}: {dir_name} ({repo_url})")
        return

    for project in projects:
        process_project(
            project_name=project,
            input_dir=input_dir,
            repos_dir=repos_dir,
            output_dir=coverage_out_dir / project,
            parsing_dir=parsing_out_dir,
            workers=args.workers,
            canonical_repo_map=canonical_repo_map,
        )

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    main()
