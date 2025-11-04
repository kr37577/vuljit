import os
import re
import argparse
import subprocess
import pandas as pd
from pathlib import Path
import git
from typing import Optional, Dict, Tuple, List
from multiprocessing import Pool

# --- 設定 ---

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_INPUT_CSV_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_INPUTS_DIR",
        REPO_ROOT / "datasets" / "derived_artifacts" / "patch_coverage_inputs",
    )
)
DEFAULT_CLONED_REPOS_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_CLONED_REPOS_DIR",
        REPO_ROOT / "datasets" / "intermediate" / "cloned_repos",
    )
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_DIFFS_DIR",
        REPO_ROOT / "data" / "intermediate" / "patch_coverage" / "daily_diffs",
    )
)

# 4. 差分抽出の対象とするファイルの拡張子リスト
CODE_FILE_EXTENSIONS = [
    '.c', '.cc', '.cpp', '.cxx', '.c++',
    '.h', '.hh','hpp', '.hxx', '.h++'
]
# 事前にタプルへ（endswith の高速化）
TARGET_EXTS = tuple(CODE_FILE_EXTENSIONS)

# Repoオブジェクトのシンプルなキャッシュ（同一リポでの再オープンを避ける）
_REPO_CACHE: Dict[str, git.Repo] = {}

def _get_repo(repo_path: Path) -> git.Repo:
    key = str(repo_path)
    repo = _REPO_CACHE.get(key)
    if repo is None:
        repo = git.Repo(key)
        _REPO_CACHE[key] = repo
    return repo
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
    create_daily_diff.py と同様に、拡張子判定は大小文字非依存で行う。
    """
    # 高速化版: `git diff --name-only`で全ファイル名を取得し、Python側で拡張子フィルタ
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

        # create_daily_diff.py と同じロジック：小文字化して endswith 判定
        target_extensions = tuple(extensions)
        files = [p for p in names if p.lower().endswith(target_extensions)]
        return files
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"  - 警告: 差分取得中に予期せぬエラーが発生しました: {e}")
        return []


def save_patches(repo_path: Path, old_revision: str, new_revision: str, output_dir: Path, extensions: list) -> int:
    """
    2つのリビジョン間の差分をファイルごとのパッチファイルとして保存する。
    保存されたパッチファイルの数を返す。
    """
    # 高速化版: `git diff <old> <new> -- <file>`で各ファイルのパッチを取得
    try:
        if not repo_path.is_dir():
            print(f"  - エラー (save_patches): リポジトリパスが見つかりません: {repo_path}")
            return 0

        changed_files = get_changed_files(repo_path, old_revision, new_revision, extensions)
        if not changed_files:
            return 0

        env = os.environ.copy()
        env.setdefault('GIT_OPTIONAL_LOCKS', '0')
        saved_patch_count = 0
        for rel_path in changed_files:
            try:
                cmd = [
                    'git', '-C', str(repo_path), 'diff',
                    str(old_revision), str(new_revision), '--', rel_path
                ]
                patch = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, env=env)
                patch_text = patch.decode('utf-8', errors='ignore')
                if not patch_text.strip():
                    continue

                # ヘッダ除去: diff --git / index / --- / +++ などを除外し、@@ からのハンクのみを残す
                # 先頭から最初のハンク開始（行頭 @@）を探し、それ以前の行を捨てる
                lines = patch_text.splitlines()
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('@@'):
                        start_idx = i
                        break
                # ハンクが見つからない場合はスキップ
                if not lines or not (start_idx < len(lines) and lines[start_idx].startswith('@@')):
                    continue
                body = "\n".join(lines[start_idx:]) + "\n"

                patch_file_path = output_dir.joinpath(rel_path + ".patch")
                patch_file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(patch_file_path, 'w', encoding='utf-8') as f:
                    f.write(body)
                saved_patch_count += 1
            except subprocess.CalledProcessError:
                continue

        return saved_patch_count
    except Exception as e:
        print(f"  - 警告 (save_patches): パッチ保存中に予期せぬエラーが発生しました: {e}")
        return 0


def _outputs_exist(output_csv_path: Path, patch_output_dir: Path) -> bool:
    try:
        if output_csv_path.is_file() and patch_output_dir.is_dir():
            return any(patch_output_dir.rglob("*.patch"))
        return False
    except Exception:
        return False


def _process_one(task: Tuple[int, Path, str, str, str, Path]) -> Tuple[int, int, int]:
    i, repo_local_path, prev_rev, curr_rev, date_str, project_output_path = task
    try:
        output_csv_path = project_output_path / f"{date_str}.csv"
        patch_output_dir = project_output_path / f"{date_str}_patches"

        # 既存出力が揃っていればスキップ
        if _outputs_exist(output_csv_path, patch_output_dir):
            return (i, 0, 0)

        changed_files = get_changed_files(
            repo_local_path, prev_rev, curr_rev, CODE_FILE_EXTENSIONS
        )

        saved_patch_count = 0
        if changed_files:
            # pandasを経由せず軽量に出力
            tmp_csv = output_csv_path.with_suffix('.csv.tmp')
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            # 互換性のためBOM付UTF-8で出力（utf-8-sig）
            with open(tmp_csv, 'w', encoding='utf-8-sig') as f:
                f.write('changed_file_path\n')
                for p in changed_files:
                    if ',' in p or '"' in p:
                        f.write('"' + p.replace('"', '""') + '"\n')
                    else:
                        f.write(p + '\n')
            os.replace(tmp_csv, output_csv_path)

            patch_output_dir.mkdir(exist_ok=True)
            saved_patch_count = save_patches(
                repo_local_path, prev_rev, curr_rev, patch_output_dir, CODE_FILE_EXTENSIONS
            )

        return (i, len(changed_files), saved_patch_count)
    except Exception:
        return (i, 0, 0)


def main():
    # 引数でプロジェクトや各ディレクトリを指定可能に（未指定時は従来の定数を使用）
    ap = argparse.ArgumentParser(description="日毎の差分抽出（途中セーブ対応・安全な並列化）")
    ap.add_argument("-p", "--project", action="append", help="処理するプロジェクト名（繰り返し可、カンマ区切りも可）")
    ap.add_argument("--input", dest="input_dir", help="コミット日付きCSVのディレクトリ")
    ap.add_argument("--repos", dest="repos_dir", help="gitクローンされたリポジトリの親ディレクトリ")
    ap.add_argument("--out", dest="out_dir", help="日毎の差分出力ディレクトリ")
    ap.add_argument("--workers", type=int, default=12, help="並列ワーカー数 (既定: 12)")
    ap.add_argument("--progress-interval", type=int, default=10, help="進捗ファイルを更新する間隔 (件数)")
    args = ap.parse_args()
    """
    メイン処理：コミット日付きCSVを読み込み、日毎の差分ファイルリストを生成する。
    """
    input_path = Path(args.input_dir or DEFAULT_INPUT_CSV_DIRECTORY)
    repos_path = Path(args.repos_dir or DEFAULT_CLONED_REPOS_DIRECTORY)
    output_path = Path(args.out_dir or DEFAULT_OUTPUT_DIRECTORY)

    if not input_path.is_dir():
        print(f"エラー: 入力ディレクトリ '{input_path}' が見つかりません。")
        return
    if not repos_path.is_dir():
        print(f"エラー: リポジトリディレクトリ '{repos_path}' が見つかりません。")
        return
    
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")

    csv_files = sorted(input_path.glob('revisions_with_commit_date_*.csv'))

    # プロジェクト指定がある場合はフィルタ
    selected: Optional[set[str]] = None
    if args.project:
        selected = set()
        for val in args.project:
            if not val:
                continue
            for token in re.split(r"[,\s]+", val.strip()):
                token = token.strip()
                if token:
                    selected.add(token)
        if selected:
            csv_files = [p for p in csv_files if p.stem.replace('revisions_with_commit_date_', '') in selected]
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
            required_cols = ['date', 'url', 'revision', 'repo_name']
            if not all(col in df.columns for col in required_cols):
                print(f"  - 警告: '{csv_file.name}' に必要な列がありません。スキップします。")
                continue
            
            # 対象プロジェクトの行に限定
            original_len = len(df)
            df = df[df['repo_name'] == project_name].copy()
            if df.empty:
                print(f"  - 警告: プロジェクト '{project_name}' に一致する行がありません。スキップします。")
                continue
            if original_len != len(df):
                print(f"  - フィルタ: {original_len}行 -> {len(df)}行 (repo_name='{project_name}')")

            # 日付でソートされていることを保証する (重要)
            df = df.sort_values(by='date').reset_index(drop=True)

            if len(df) < 2:
                print("  - 差分を取得するにはデータが2行未満です。スキップします。")
                continue

            # プロジェクトごとの出力ディレクトリを作成
            project_output_path = output_path / project_name
            project_output_path.mkdir(exist_ok=True)
            
            print(f"  - {len(df)-1} 日分の差分を処理します。")

            # --- 途中セーブ/再開機能（内部処理は変更しない）---
            progress_file = project_output_path / ".progress"

            def _load_progress(path: Path) -> Optional[int]:
                try:
                    if path.is_file():
                        txt = path.read_text(encoding='utf-8').strip()
                        if txt.isdigit():
                            idx = int(txt)
                            if idx < 1:
                                return 1
                            next_idx = idx + 1
                            if next_idx >= len(df):
                                return len(df)
                            return next_idx
                    return None
                except Exception:
                    return None

            def _infer_progress_from_outputs() -> Optional[int]:
                # 既に出力済みの日付(YYYYMMDD.csv)やパッチディレクトリから、最後に処理した行インデックスを推測
                try:
                    done_dates = set()
                    for p in project_output_path.glob("*.csv"):
                        if p.name[:8].isdigit():
                            done_dates.add(p.name[:8])
                    for d in project_output_path.iterdir():
                        if d.is_dir() and d.name.endswith("_patches") and d.name[:8].isdigit():
                            done_dates.add(d.name[:8])
                    if not done_dates:
                        return None
                    last_idx = None
                    for i in range(1, len(df)):
                        date_str = str(df.iloc[i]['date'])
                        if str(date_str) in done_dates:
                            last_idx = i
                    if last_idx is None:
                        return None
                    next_idx = last_idx + 1
                    if next_idx >= len(df):
                        return len(df)
                    return next_idx
                except Exception:
                    return None

            start_index = _load_progress(progress_file)
            if start_index is None:
                start_index = _infer_progress_from_outputs()
            if start_index is None:
                start_index = 1

            if start_index > 1:
                print(f"  - 再開ポイントを検出: 直近の処理インデックス={start_index} (日付={df.iloc[start_index]['date']}) から再開します。")

            # 2行目相当(start_index)から日次ペアを並列処理
            tasks: List[Tuple[int, Path, str, str, str, Path]] = []
            for i in range(start_index, len(df)):
                previous_row = df.iloc[i-1]
                current_row = df.iloc[i]
                repo_dir_name = get_repo_dir_name_from_url(str(current_row['url']))
                repo_local_path = repos_path / repo_dir_name
                date_str = str(current_row['date'])
                tasks.append((i, repo_local_path, str(previous_row['revision']), str(current_row['revision']), date_str, project_output_path))

            workers = max(1, args.workers)
            prog_interval = max(1, args.progress_interval)
            max_done = start_index - 1
            done_count = 0

            if tasks:
                with Pool(processes=workers) as pool:
                    for (i_done, n_changed, n_patches) in pool.imap_unordered(_process_one, tasks):
                        done_count += 1
                        max_done = max(max_done, i_done)
                        if (done_count % prog_interval) == 0:
                            try:
                                with open(progress_file, 'w', encoding='utf-8') as pf:
                                    pf.write(str(max_done))
                                    pf.flush()
                                    os.fsync(pf.fileno())
                            except Exception:
                                pass

                # 最終進捗を記録
                try:
                    with open(progress_file, 'w', encoding='utf-8') as pf:
                        pf.write(str(max_done))
                        pf.flush()
                        os.fsync(pf.fileno())
                except Exception:
                    pass

            print(f"✔ プロジェクト '{project_name}' の処理が完了しました。")
            # 完了後に進捗ファイルを残す/消すは任意だが、ここでは残しておく（再実行時のスキップに利用）

        except Exception as e:
            print(f"  - エラー: '{csv_file.name}' の処理中に予期せぬエラーが発生しました: {e}")

    print("\nすべての処理が完了しました。")


if __name__ == "__main__":
    main()
