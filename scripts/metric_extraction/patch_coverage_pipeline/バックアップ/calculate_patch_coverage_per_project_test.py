import pandas as pd
from pathlib import Path
import re
from typing import Set, Dict, Any
from google.cloud import storage
from bs4 import BeautifulSoup
import os
import json
import argparse # 【変更点】コマンドライン引数を扱うために追加

# --- 設定 ---

DAILY_DIFFS_DIRECTORY = os.environ.get('VULJIT_DAILY_DIFFS_DIR')  # optional env override
COVERAGE_BUCKET_NAME = os.environ.get('VULJIT_COVERAGE_BUCKET', 'oss-fuzz-coverage')
OUTPUT_BASE_DIRECTORY = os.environ.get('VULJIT_PATCH_COVERAGE_OUT')
PARSING_RESULTS_DIRECTORY = os.environ.get('VULJIT_PARSING_RESULTS_DIR')


# ----------------

def get_added_lines_with_content_from_patch(patch_file: Path) -> Dict[int, str]:
    """
    パッチファイルを解析し、追加された行の「行番号」と「内容」の辞書を返す。
    """
    added_lines_content = {}
    try:
        with open(patch_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        new_file_line_num = 0
        for line in lines:
            if line.startswith('@@'):
                match = re.search(r'\+([0-9]+)', line)
                if match:
                    new_file_line_num = int(match.group(1)) - 1
            elif line.startswith('+') and not line.startswith('+++'):
                new_file_line_num += 1
                added_lines_content[new_file_line_num] = line[1:].strip()
            elif not line.startswith('-'):
                new_file_line_num += 1
    except FileNotFoundError:
        print(f"    - 警告: パッチファイルが見つかりません: {patch_file}")
    except Exception as e:
        print(f"    - 警告: パッチファイル解析中にエラー: {patch_file}, {e}")
    return added_lines_content

def find_report_blob_path(storage_client: storage.Client, project_name: str, date: str, target_file_path: str) -> str | None:
    """
    GCSバケット内で指定されたファイルパスに一致するカバレッジレポートのBlobパスを検索する。
    """
    try:
        prefix = f"{project_name}/reports/{date}/linux/"
        blobs_iterator = storage_client.list_blobs(COVERAGE_BUCKET_NAME, prefix=prefix)
        candidate_blobs = [blob.name for blob in blobs_iterator if blob.name.endswith(target_file_path + '.html')]

        if not candidate_blobs:
            return None
        if len(candidate_blobs) == 1:
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
        return best_candidate
    except Exception as e:
        print(f"    - 警告: GCSでのBlob検索中にエラーが発生しました: {e}")
        return None

def extract_code_data_from_gcs_report(storage_client: storage.Client, blob_path: str) -> Dict[int, Dict[str, Any]]:
    """
    GCSからカバレッジレポートHTMLを取得し、全行の「行番号」「内容」「実行回数」を抽出する。
    """
    code_data = {}
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
    return code_data

def main():
    """
    メイン処理：日毎の差分を読み込み、パッチカバレッジを計算してプロジェクト毎にCSV出力する。
    HTML解析結果も別途JSONで保存する。
    """
    # 【変更点】コマンドライン引数のパーサーを設定
    parser = argparse.ArgumentParser(description="指定された単一プロジェクトのパッチカバレッジを計算します。")
    parser.add_argument('-p', '--project', required=True, type=str, help="処理対象のプロジェクト名を指定します。")
    parser.add_argument('--diffs', default=DAILY_DIFFS_DIRECTORY, help='Daily diffs root directory')
    parser.add_argument('--out', default=OUTPUT_BASE_DIRECTORY, help='Output root directory for patch coverage CSV')
    parser.add_argument('--parsing-out', default=PARSING_RESULTS_DIRECTORY, help='Output root directory for HTML parsing cache')
    args = parser.parse_args()
    target_project_name = args.project

    # Resolve defaults if env/args not provided
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent
    base_dir = Path(args.diffs) if args.diffs else Path(os.path.join(os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate')), 'patch_coverage', 'daily_diffs'))
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
    output_root = Path(args.out) if args.out else Path(os.path.join(os.environ.get('VULJIT_OUTPUTS_DIR', str(repo_root / 'outputs')), 'metrics', 'patch_coverage'))
    output_project_dir = output_root / project_name
    output_project_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_project_dir / f"{project_name}_patch_coverage.csv"
    
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
            for index, row in changed_files_df.iterrows():
                file_path_str = row['changed_file_path']
                added_lines_dict = get_added_lines_with_content_from_patch(patch_dir / (file_path_str + ".patch"))
                if not added_lines_dict:
                    continue

                report_blob_path = find_report_blob_path(storage_client, project_name, date, file_path_str)
                total_added = len(added_lines_dict)
                covered_added_lines_count = 0
                
                if report_blob_path:
                    report_data_dict = extract_code_data_from_gcs_report(storage_client, report_blob_path)
                    
                    if report_data_dict:
                        # HTML解析結果をJSONファイルで保存する処理
                        try:
                            parsing_root = Path(args.parsing_out) if args.parsing_out else Path(os.path.join(os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate')), 'patch_coverage', 'parsing_results'))
                            parsing_output_file = parsing_root / project_name / date / (file_path_str + '.json')
                            parsing_output_file.parent.mkdir(parents=True, exist_ok=True)
                            with open(parsing_output_file, 'w', encoding='utf-8') as f:
                                json.dump(report_data_dict, f, ensure_ascii=False, indent=4)
                        except Exception as e:
                            print(f"    - 警告: 解析結果のJSON保存中にエラー: {parsing_output_file}, {e}")

                        # カバレッジ計算処理
                        for line_num, line_content in added_lines_dict.items():
                            if line_num in report_data_dict and line_content == report_data_dict[line_num]['content']:
                                exec_count = report_data_dict[line_num]['execute']
                                if exec_count is not None and exec_count > 0:
                                    covered_added_lines_count += 1
                
                coverage = (covered_added_lines_count / total_added) * 100 if total_added > 0 else 0
                if total_added > 0:
                    print(f"    - {file_path_str}: 追加 {total_added}行, うちカバー {covered_added_lines_count}行 (カバレッジ: {coverage:.2f}%)")

                daily_results.append({
                    'project': project_name,
                    'date': date,
                    'file_path': file_path_str,
                    'total_added_lines': total_added,
                    'covered_added_lines': covered_added_lines_count,
                    'patch_coverage': coverage
                })
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
