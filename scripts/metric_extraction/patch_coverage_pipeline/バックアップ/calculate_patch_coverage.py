import pandas as pd
from pathlib import Path
import re
from typing import Set, Dict, Any
from google.cloud import storage
from bs4 import BeautifulSoup
import os
import json

# --- 設定 ---

# 1. create_daily_diff.py の出力ディレクトリ
DAILY_DIFFS_DIRECTORY = '/work/riku-ka/patch_coverage_culculater/daily_diffs_test'

# 2. OSS-Fuzz カバレッジレポートのGCSバケット名
COVERAGE_BUCKET_NAME = 'oss-fuzz-coverage'

# 3. パッチカバレッジ結果を出力する親ディレクトリ
OUTPUT_BASE_DIRECTORY = '/work/riku-ka/patch_coverage_culculater/patch_coverage_results'

# 4. HTML解析結果を保存する親ディレクトリ
PARSING_RESULTS_DIRECTORY = '/work/riku-ka/patch_coverage_culculater/parsing_results'


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
                
                # --- 【重要・変更】ここから ---
                # 既に行番号が辞書に登録されている場合はスキップする。
                # これにより、expansion-view内の重複した行情報が無視され、最初に出現した情報が保持される。
                if line_number in code_data:
                    continue
                # --- 【重要・変更】ここまで ---

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
    base_dir = Path(DAILY_DIFFS_DIRECTORY)
    if not base_dir.is_dir():
        print(f"エラー: 入力ディレクトリ '{base_dir}' が見つかりません。")
        return

    try:
        storage_client = storage.Client.create_anonymous_client()
    except Exception as e:
        print(f"GCSクライアントの初期化に失敗しました: {e}")
        return

    project_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    print(f"{len(project_dirs)}個のプロジェクトを処理します...")

    for project_dir in project_dirs:
        project_name = project_dir.name
        print(f"\n▶ プロジェクト '{project_name}' の処理を開始...")
        
        project_results = []
        diff_csv_files = sorted(list(project_dir.glob('*.csv')))
        
        for diff_csv in diff_csv_files:
            date = diff_csv.stem
            print(f"  - 日付: {date}")
            patch_dir = project_dir / f"{date}_patches"
            if not patch_dir.is_dir():
                continue

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
                                parsing_output_file = Path(PARSING_RESULTS_DIRECTORY) / project_name / date / (file_path_str + '.json')
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

                    project_results.append({
                        'project': project_name,
                        'date': date,
                        'file_path': file_path_str,
                        'total_added_lines': total_added,
                        'covered_added_lines': covered_added_lines_count,
                        'patch_coverage': coverage
                    })
            except Exception as e:
                print(f"  - エラー: CSV処理中にエラーが発生しました: {diff_csv}, {e}")

        # プロジェクトの処理完了後、結果をCSVに保存
        if project_results:
            output_project_dir = Path(OUTPUT_BASE_DIRECTORY) / project_name
            output_project_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_project_dir / f"{project_name}_patch_coverage.csv"
            
            results_df = pd.DataFrame(project_results)
            results_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"✔ プロジェクト '{project_name}' のカバレッジ結果を '{output_file_path}' に保存しました。")
        else:
            print(f"プロジェクト '{project_name}' で処理対象のデータが見つかりませんでした。")
    
    print("\nすべての処理が完了しました。")

if __name__ == "__main__":
    main()