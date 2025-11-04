import pandas as pd
from pathlib import Path
import re
from typing import Dict, Any, Optional, Tuple, Iterable
from google.cloud import storage
from bs4 import BeautifulSoup
import os
import json
import argparse  # 【変更点】コマンドライン引数を扱うために追加
from concurrent.futures import ProcessPoolExecutor

# --- 設定 ---

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# 1. create_daily_diff.py の出力ディレクトリ
DEFAULT_DAILY_DIFFS_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_DIFFS_DIR",
        REPO_ROOT / "data" / "intermediate" / "patch_coverage" / "daily_diffs",
    )
)

# 2. OSS-Fuzz カバレッジレポートのGCSバケット名
COVERAGE_BUCKET_NAME = os.environ.get("VULJIT_COVERAGE_BUCKET", "oss-fuzz-coverage")

# 3. パッチカバレッジ結果を出力する親ディレクトリ
DEFAULT_OUTPUT_BASE_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_RESULTS_DIR",
        REPO_ROOT / "datasets" / "derived_artifacts" / "patch_coverage_metrics",
    )
)

# 4. HTML解析結果を保存する親ディレクトリ
DEFAULT_PARSING_RESULTS_DIRECTORY = Path(
    os.environ.get(
        "VULJIT_PATCH_COVERAGE_PARSING_DIR",
        REPO_ROOT
        / "data"
        / "intermediate"
        / "patch_coverage"
        / "parsing_results",
    )
)


# ----------------

# キャッシュと事前コンパイル（出力は不変、速度のみ改善）
_HUNK_RE = re.compile(r"\+([0-9]+)")
_BLOB_LIST_CACHE: Dict[Tuple[str, str], list] = {}
_BLOB_RESOLVE_CACHE: Dict[Tuple[str, str, str], Optional[str]] = {}
_REPORT_CACHE: Dict[str, Dict[int, Dict[str, Any]]] = {}

# 並列ワーカーで共有する匿名クライアント（初期化コスト削減用）
_WORKER_STORAGE_CLIENT: Optional[storage.Client] = None


def _initialize_worker_storage_client() -> None:
    """
    ProcessPoolExecutor の initializer から呼び出し、
    各ワーカーで一度だけ匿名クライアントを生成して共有する。
    """
    global _WORKER_STORAGE_CLIENT
    if _WORKER_STORAGE_CLIENT is not None:
        return
    try:
        _WORKER_STORAGE_CLIENT = storage.Client.create_anonymous_client()
    except Exception as e:
        _WORKER_STORAGE_CLIENT = None
        print(f"    - 警告: ワーカー初期化時に GCS クライアント生成に失敗しました: {e}")


def _extract_added_lines_from_iter(lines: Iterable[str]) -> Dict[int, str]:
    added_lines_content: Dict[int, str] = {}
    new_file_line_num = 0
    for line in lines:
        if line.startswith('@@'):
            match = _HUNK_RE.search(line)
            if match:
                new_file_line_num = int(match.group(1)) - 1
        elif line.startswith('+') and not line.startswith('+++'):
            new_file_line_num += 1
            added_lines_content[new_file_line_num] = line[1:].strip()
        elif not line.startswith('-'):
            new_file_line_num += 1
    return added_lines_content


def extract_added_lines_from_patch_text(patch_text: str) -> Dict[int, str]:
    """
    パッチ文字列を解析し、追加された行の「行番号」と「内容」の辞書を返す。
    """
    if not patch_text:
        return {}
    return _extract_added_lines_from_iter(patch_text.splitlines())

def get_added_lines_with_content_from_patch(patch_file: Path) -> Dict[int, str]:
    """
    パッチファイルを解析し、追加された行の「行番号」と「内容」の辞書を返す。
    """
    added_lines_content = {}
    try:
        with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
            added_lines_content = _extract_added_lines_from_iter(f.readlines())
    except FileNotFoundError:
        print(f"    - 警告: パッチファイルが見つかりません: {patch_file}")
    except Exception as e:
        print(f"    - 警告: パッチファイル解析中にエラー: {patch_file}, {e}")
    return added_lines_content

def _list_blobs_for_date(storage_client: storage.Client, project_name: str, date: str) -> list:
    key = (project_name, date)
    if key in _BLOB_LIST_CACHE:
        return _BLOB_LIST_CACHE[key]
    prefix = f"{project_name}/reports/{date}/linux/"
    try:
        blobs_iter = storage_client.list_blobs(COVERAGE_BUCKET_NAME, prefix=prefix)
        names = [b.name for b in blobs_iter]
    except Exception:
        names = []
    _BLOB_LIST_CACHE[key] = names
    return names

def find_report_blob_path(storage_client: storage.Client, project_name: str, date: str, target_file_path: str) -> Optional[str]:
    """
    GCSバケット内で指定されたファイルパスに一致するカバレッジレポートのBlobパスを検索する。
    """
    try:
        cache_key = (project_name, date, target_file_path)
        if cache_key in _BLOB_RESOLVE_CACHE:
            return _BLOB_RESOLVE_CACHE[cache_key]
        blobs = _list_blobs_for_date(storage_client, project_name, date)
        candidate_blobs = [name for name in blobs if name.endswith(target_file_path + '.html')]

        if not candidate_blobs:
            _BLOB_RESOLVE_CACHE[cache_key] = None
            return None
        if len(candidate_blobs) == 1:
            _BLOB_RESOLVE_CACHE[cache_key] = candidate_blobs[0]
            return candidate_blobs[0]

        target_parts = target_file_path.split('/')
        best_candidate = None
        max_match_count = -1

        for candidate_path in candidate_blobs:
            candidate_parts = candidate_path.replace('.html', '').split('/')
            match_count = sum(1 for i in range(1, min(len(target_parts), len(candidate_parts)) + 1) if target_parts[-i] == candidate_parts[-i])
            
            if match_count > max_match_count:
                max_match_count = match_count
                best_candidate = candidate_path
            elif match_count == max_match_count and (best_candidate is None or len(candidate_path) < len(best_candidate)):
                best_candidate = candidate_path
        _BLOB_RESOLVE_CACHE[cache_key] = best_candidate
        return best_candidate
    except Exception as e:
        print(f"    - 警告: GCSでのBlob検索中にエラーが発生しました: {e}")
        return None

def extract_code_data_from_gcs_report(storage_client: storage.Client, blob_path: str) -> Dict[int, Dict[str, Any]]:
    """
    GCSからカバレッジレポートHTMLを取得し、全行の「行番号」「内容」「実行回数」を抽出する。
    """
    cached = _REPORT_CACHE.get(blob_path)
    if cached is not None:
        return cached
    code_data: Dict[int, Dict[str, Any]] = {}
    try:
        bucket = storage_client.bucket(COVERAGE_BUCKET_NAME)
        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"    - 警告: 指定されたBlobが見つかりません: {blob_path}")
            return code_data

        html_content = blob.download_as_bytes()
        soup = BeautifulSoup(html_content, 'html.parser')

        for row in soup.find_all('tr'):
            line_number_tag = row.find('td', class_='line-number')
            code_tag = row.find('td', class_='code')
            if not (line_number_tag and code_tag):
                continue
            
            try:
                line_number = int(line_number_tag.get_text(strip=True))
                
                if line_number in code_data:
                    continue

                content = code_tag.pre.get_text() if code_tag.pre else code_tag.get_text()
                
                execute_count = None
                count_tag = row.find('td', class_='covered-line') or row.find('td', class_='uncovered-line')
                if count_tag and (count_text := count_tag.get_text(strip=True)):
                    count_text = count_text.lower()
                    multiplier = {'k': 1000, 'm': 1000000}.get(count_text[-1], 1)
                    execute_count = int(float(count_text.rstrip('km')) * multiplier)
                
                code_data[line_number] = {'content': content.strip(), 'execute': execute_count}
            except (ValueError, TypeError, AttributeError):
                continue
    except Exception as e:
        print(f"    - 警告: GCSからのレポート取得または解析中にエラー: {blob_path}, {e}")
    # 正常終了時はキャッシュに保存
    _REPORT_CACHE[blob_path] = code_data
    return code_data


def compute_patch_coverage_for_patch_text(
    project_name: str,
    date: str,
    file_path_str: str,
    patch_text: str,
    parsing_output_root: Optional[Path] = None,
    storage_client: Optional[storage.Client] = None,
) -> Optional[Dict[str, Any]]:
    """
    パッチ文字列を直接受け取り、単一ファイルのパッチカバレッジ結果を計算して返す。
    """
    try:
        added_lines_dict = extract_added_lines_from_patch_text(patch_text)
        if not added_lines_dict:
            return None

        total_added = len(added_lines_dict)
        covered_added_lines_count = 0

        report_blob_path = None
        report_data_dict: Dict[int, Dict[str, Any]] = {}
        try:
            client = storage_client or storage.Client.create_anonymous_client()
            report_blob_path = find_report_blob_path(client, project_name, date, file_path_str)
            if report_blob_path:
                report_data_dict = extract_code_data_from_gcs_report(client, report_blob_path)
        except Exception as e:
            print(f"    - 警告: パッチカバレッジ計算中のGCS処理でエラー: {file_path_str}, {e}")

        if report_data_dict and parsing_output_root is not None:
            try:
                parsing_output_file = Path(parsing_output_root) / project_name / date / (file_path_str + '.json')
                parsing_output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(parsing_output_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data_dict, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"    - 警告: 解析結果のJSON保存中にエラー: {file_path_str}, {e}")

        if report_data_dict:
            for line_num, line_content in added_lines_dict.items():
                if line_num in report_data_dict and line_content == report_data_dict[line_num]['content']:
                    exec_count = report_data_dict[line_num]['execute']
                    if exec_count is not None and exec_count > 0:
                        covered_added_lines_count += 1

        coverage = (covered_added_lines_count / total_added) * 100 if total_added > 0 else 0

        return {
            'project': project_name,
            'date': date,
            'file_path': file_path_str,
            'total_added_lines': total_added,
            'covered_added_lines': covered_added_lines_count,
            'patch_coverage': coverage,
        }
    except Exception as e:
        print(f"    - エラー: パッチカバレッジ計算中に予期せぬエラーが発生しました: {file_path_str}, {e}")
        return None


def _process_one_file(args: Tuple[str, str, str, str, str]) -> Optional[Dict[str, Any]]:
    """
    並列ワーカー用: 単一ファイルのパッチカバレッジを計算して結果辞書を返す。
    入力: (project_name, date, patch_dir_str, file_path_str)
    出力: dict または None（追加行がない等でスキップ）
    ロジックは既存の逐次処理と同一。
    """
    project_name, date, patch_dir_str, parsing_root_str, file_path_str = args
    try:
        patch_path = Path(patch_dir_str) / (file_path_str + ".patch")
        try:
            patch_text = patch_path.read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            print(f"    - 警告: パッチファイルが見つかりません: {patch_path}")
            return None
        except Exception as e:
            print(f"    - 警告: パッチファイル読み込み中にエラー: {patch_path}, {e}")
            return None

        parsing_root_path: Optional[Path] = Path(parsing_root_str) if parsing_root_str else None
        storage_client = _WORKER_STORAGE_CLIENT
        return compute_patch_coverage_for_patch_text(
            project_name=project_name,
            date=date,
            file_path_str=file_path_str,
            patch_text=patch_text,
            parsing_output_root=parsing_root_path,
            storage_client=storage_client,
        )
    except Exception as e:
        print(f"    - エラー: 並列処理中にエラー: {file_path_str}, {e}")
        return None

def main():
    """
    メイン処理：日毎の差分を読み込み、パッチカバレッジを計算してプロジェクト毎にCSV出力する。
    HTML解析結果も別途JSONで保存する。
    """
    # 【変更点】コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(description="指定された単一プロジェクトのパッチカバレッジを計算します。")
    parser.add_argument('-p', '--project', required=True, type=str, help="処理対象のプロジェクト名を指定します。")
    parser.add_argument('--diffs', default=None, help='create_daily_diff.py の出力ディレクトリ')
    parser.add_argument('--out', default=None, help='パッチカバレッジCSVの出力先ベースディレクトリ')
    parser.add_argument('--parsing-out', default=None, help='HTML解析結果(JSON)の保存ディレクトリ')
    parser.add_argument('--workers', type=int, default=3, help='並列プロセス数')
    args = parser.parse_args()
    target_project_name = args.project

    base_dir = Path(args.diffs or DEFAULT_DAILY_DIFFS_DIRECTORY)
    if not base_dir.is_dir():
        print(f"エラー: 入力ディレクトリ '{base_dir}' が見つかりません。")
        return

    try:
        storage_client = storage.Client.create_anonymous_client()
    except Exception as e:
        print(f"GCSクライアントの初期化に失敗しました: {e}")
        return

    # 【変更点】指定されたプロジェクトのディレクトリが存在するかチェック
    project_dir = base_dir / target_project_name
    if not project_dir.is_dir():
        print(f"エラー: 指定されたプロジェクト '{target_project_name}' のディレクトリが見つかりません: {project_dir}")
        return

    # 【変更点】複数プロジェクトを対象としたループを削除し、単一プロジェクトの処理に
    project_name = project_dir.name
    print(f"\n▶ プロジェクト '{project_name}' の処理を開始...")
    
    # 出力ファイルパスを先に定義
    output_root = Path(args.out or DEFAULT_OUTPUT_BASE_DIRECTORY)
    output_project_dir = output_root / project_name
    output_project_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_project_dir / f"{project_name}_patch_coverage.csv"

    parsing_root = Path(args.parsing_out or DEFAULT_PARSING_RESULTS_DIRECTORY)
    parsing_root.mkdir(parents=True, exist_ok=True)
    
    # --- ★★★ 途中再開機能 ★★★ ---
    processed_dates = set()
    if output_file_path.exists():
        try:
            # 既存の出力ファイルを読み込み、処理済みの日付を取得
            existing_df = pd.read_csv(output_file_path)
            if 'date' in existing_df.columns:
                # 日付を文字列に変換してセットに追加
                processed_dates = set(existing_df['date'].astype(str))
            print(f"✔ 既存の出力ファイルを発見。{len(processed_dates)}件の処理済み日付を読み込みました。")
        except pd.errors.EmptyDataError:
            print(f"⚠️ 既存の出力ファイルは空です。最初から処理を開始します。")
        except Exception as e:
            print(f"⚠️ 既存の出力ファイルの読み込みに失敗しました: {e}。最初から処理を開始します。")
    # ------------------------------------

    diff_csv_files = sorted(list(project_dir.glob('*.csv')))
    
    if not diff_csv_files:
        print(f"プロジェクト '{project_name}' 内に処理対象のCSVファイルが見つかりませんでした。")
        return

    processed_something = False
    for diff_csv in diff_csv_files:
        date = diff_csv.stem
        
        # ★★★ 処理済みの日付はスキップ ★★★
        if date in processed_dates:
            print(f"  - スキップ: 日付 '{date}' は既に処理済みです。")
            continue

        print(f"  - 日付: {date}")
        patch_dir = project_dir / f"{date}_patches"
        if not patch_dir.is_dir():
            continue

        daily_results = []
        try:
            changed_files_df = pd.read_csv(diff_csv)
            if 'changed_file_path' not in changed_files_df.columns:
                print(f"  - 警告: '{diff_csv.name}' に 'changed_file_path' 列がありません。スキップします。")
                continue

            file_paths = changed_files_df['changed_file_path'].astype(str).tolist()
            parsing_root_str = str(parsing_root)
            args_list = [(project_name, date, str(patch_dir), parsing_root_str, fp) for fp in file_paths]

            # 入力順を維持するため map を使用
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_initialize_worker_storage_client,
            ) as ex:
                results = list(ex.map(_process_one_file, args_list))

            # Noneを除外
            daily_results = [r for r in results if r]

            # 既存のログ出力と同等のメッセージを順序通りに表示
            for r in daily_results:
                total_added = r['total_added_lines']
                if total_added > 0:
                    print(f"    - {r['file_path']}: 追加 {total_added}行, うちカバー {r['covered_added_lines']}行 (カバレッジ: {r['patch_coverage']:.2f}%)")

        except Exception as e:
            print(f"  - エラー: CSV処理中にエラーが発生しました: {diff_csv}, {e}")

        # 日付ごとの処理が完了したら、結果をCSVに追記
        if daily_results:
            processed_something = True
            daily_results_df = pd.DataFrame(daily_results)
            # ファイルが存在しない初回書き込み時のみヘッダーを出力し、以降は追記モード
            header = not output_file_path.exists() or output_file_path.stat().st_size == 0
            daily_results_df.to_csv(output_file_path, mode='a', header=header, index=False, encoding='utf-8-sig')
            print(f"  ✔ 日付 '{date}' の結果を '{output_file_path}' に追記しました。")


    # プロジェクトの処理完了後、最終メッセージを表示
    if processed_something or len(processed_dates) > 0:
        print(f"\n✔ プロジェクト '{project_name}' の全カバレッジ結果を '{output_file_path}' に保存しました。")
    else:
        print(f"プロジェクト '{project_name}' で処理対象のデータが見つかりませんでした。")
    
    print("\nすべての処理が完了しました。")
    
if __name__ == "__main__":
    main()
