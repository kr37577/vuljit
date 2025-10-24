import os
import json
import csv
import glob
import re  # 正規表現を扱うためにimport
from pathlib import Path
import argparse


def process_coverage_data(project_base_dir_str, output_root: str | None = None):
    """
    指定されたプロジェクトディレクトリ内のカバレッジJSONファイルを処理し、
    ファイル単位の詳細とサマリーの合計をそれぞれ別のCSVファイルに出力します。

    Args:
        project_base_dir_str (str): ルートとなるプロジェクトディレクトリのパス。
                                    この下の階層に日付形式のディレクトリが含まれていることを期待します。
    """
    project_base_dir = Path(project_base_dir_str)
    if not project_base_dir.is_dir():
        print(
            f"Error: Project directory '{project_base_dir_str}' not found or is not a directory.")
        return

    project_name = project_base_dir.name
    if output_root:
        output_dir = Path(output_root) / project_name
    else:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / "output_project_0802" / project_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-file CSV ---
    output_per_file_csv_filename = f"{project_name}_and_date.csv"
    output_per_file_csv_path = output_dir / output_per_file_csv_filename
    headers_per_file = [
        "date", "filename",
        "branches_count", "branches_covered", "branches_notcovered", "branches_percent",
        "functions_count", "functions_covered", "functions_percent",
        "instantiations_count", "instantiations_covered", "instantiations_percent",
        "lines_count", "lines_covered", "lines_percent",
        "regions_count", "regions_covered", "regions_notcovered", "regions_percent"
    ]
    all_rows_per_file = []

    # --- Total CSV ---
    output_total_csv_filename = f"{project_name}_total_and_date.csv"
    output_total_csv_path = output_dir / output_total_csv_filename
    headers_total = [
        "date",
        "totals_branches_count", "totals_branches_covered", "totals_branches_notcovered", "totals_branches_percent",
        "totals_functions_count", "totals_functions_covered", "totals_functions_percent",
        "totals_instantiations_count", "totals_instantiations_covered", "totals_instantiations_percent",
        "totals_lines_count", "totals_lines_covered", "totals_lines_percent",
        "totals_regions_count", "totals_regions_covered", "totals_regions_notcovered", "totals_regions_percent"
    ]
    all_rows_total = []

    search_pattern = str(project_base_dir / "**" / "*.json")

    print(f"Searching for JSON files with pattern: {search_pattern}")
    found_files = list(glob.glob(search_pattern, recursive=True))

    # outputディレクトリ内のファイルは処理対象から除外
    found_files = [f for f in found_files if not Path(
        f).resolve().is_relative_to(output_dir.resolve())]

    if not found_files:
        print(
            f"No JSON files found in '{project_base_dir_str}'.")
    else:
        print(f"Found {len(found_files)} JSON files to process.")

    # --- 修正箇所 ---
    # YYYY-MM-DD や YYYYMMDD 形式にマッチする正規表現パターン
    date_pattern = re.compile(r'^\d{4}-?\d{2}-?\d{2}$')
    # --- 修正箇所ここまで ---

    for json_file_path_str in found_files:
        json_file_path = Path(json_file_path_str)
        try:
            relative_path = json_file_path.relative_to(project_base_dir)
            relative_path_parts = relative_path.parts

            # --- 修正箇所 ---
            # パスの中から日付形式のディレクトリ名を探す
            date_str = None
            for part in relative_path_parts:
                if date_pattern.match(part):
                    date_str = part
                    break  # 日付が見つかったらループを抜ける

            if not date_str:
                print(
                    f"Warning: Could not find a date-like directory in path {json_file_path_str}. Skipping.")
                continue
            # --- 修正箇所ここまで ---

            with open(json_file_path, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

            data_block = None
            # 従来の summary.json 形式 (dataキーでラップされている)
            if "data" in coverage_data and isinstance(coverage_data["data"], list) and coverage_data["data"]:
                data_block = coverage_data["data"][0]
            # fuzzer_stats のような、dataキーでラップされていないJSONに対応
            elif "totals" in coverage_data and "files" in coverage_data:
                data_block = coverage_data
            else:
                print(
                    f"Warning: 'data' array or 'totals'/'files' keys are missing or empty in {json_file_path_str}. Skipping.")
                continue

            # --- Process per-file data ---
            if "files" in data_block:
                for file_coverage_info in data_block.get("files", []):
                    filename = file_coverage_info.get("filename", "N/A")
                    summary_details = file_coverage_info.get("summary", {})

                    row_per_file = {
                        "date": date_str,
                        "filename": filename,
                        "branches_count": summary_details.get("branches", {}).get("count", 0),
                        "branches_covered": summary_details.get("branches", {}).get("covered", 0),
                        "branches_notcovered": summary_details.get("branches", {}).get("notcovered", 0),
                        "branches_percent": summary_details.get("branches", {}).get("percent", 0.0),
                        "functions_count": summary_details.get("functions", {}).get("count", 0),
                        "functions_covered": summary_details.get("functions", {}).get("covered", 0),
                        "functions_percent": summary_details.get("functions", {}).get("percent", 0.0),
                        "instantiations_count": summary_details.get("instantiations", {}).get("count", 0),
                        "instantiations_covered": summary_details.get("instantiations", {}).get("covered", 0),
                        "instantiations_percent": summary_details.get("instantiations", {}).get("percent", 0.0),
                        "lines_count": summary_details.get("lines", {}).get("count", 0),
                        "lines_covered": summary_details.get("lines", {}).get("covered", 0),
                        "lines_percent": summary_details.get("lines", {}).get("percent", 0.0),
                        "regions_count": summary_details.get("regions", {}).get("count", 0),
                        "regions_covered": summary_details.get("regions", {}).get("covered", 0),
                        "regions_notcovered": summary_details.get("regions", {}).get("notcovered", 0),
                        "regions_percent": summary_details.get("regions", {}).get("percent", 0.0),
                    }
                    all_rows_per_file.append(row_per_file)
            else:
                print(
                    f"Warning: 'files' key is missing in data block of {json_file_path_str} for per-file data.")

            # --- Process total data ---
            if "totals" in data_block:
                totals_summary = data_block.get("totals", {})
                row_total = {
                    "date": date_str,
                    "totals_branches_count": totals_summary.get("branches", {}).get("count", 0),
                    "totals_branches_covered": totals_summary.get("branches", {}).get("covered", 0),
                    "totals_branches_notcovered": totals_summary.get("branches", {}).get("notcovered", 0),
                    "totals_branches_percent": totals_summary.get("branches", {}).get("percent", 0.0),
                    "totals_functions_count": totals_summary.get("functions", {}).get("count", 0),
                    "totals_functions_covered": totals_summary.get("functions", {}).get("covered", 0),
                    "totals_functions_percent": totals_summary.get("functions", {}).get("percent", 0.0),
                    "totals_instantiations_count": totals_summary.get("instantiations", {}).get("count", 0),
                    "totals_instantiations_covered": totals_summary.get("instantiations", {}).get("covered", 0),
                    "totals_instantiations_percent": totals_summary.get("instantiations", {}).get("percent", 0.0),
                    "totals_lines_count": totals_summary.get("lines", {}).get("count", 0),
                    "totals_lines_covered": totals_summary.get("lines", {}).get("covered", 0),
                    "totals_lines_percent": totals_summary.get("lines", {}).get("percent", 0.0),
                    "totals_regions_count": totals_summary.get("regions", {}).get("count", 0),
                    "totals_regions_covered": totals_summary.get("regions", {}).get("covered", 0),
                    "totals_regions_notcovered": totals_summary.get("regions", {}).get("notcovered", 0),
                    "totals_regions_percent": totals_summary.get("regions", {}).get("percent", 0.0),
                }
                all_rows_total.append(row_total)
            else:
                print(
                    f"Warning: 'totals' key is missing in data block of {json_file_path_str}. Skipping totals for this file.")

        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from {json_file_path_str}. Skipping.")
        except Exception as e:
            print(
                f"An unexpected error occurred while processing {json_file_path_str}: {e}. Skipping.")

    # --- Write per-file CSV ---
    if not all_rows_per_file:
        print(
            f"No per-file data found to write to CSV for project '{project_name}'.")
        try:
            with open(output_per_file_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers_per_file)
                writer.writeheader()
            print(
                f"Empty per-file CSV with headers created at: {output_per_file_csv_path}")
        except IOError:
            print(
                f"Error writing empty per-file CSV file to {output_per_file_csv_path}.")
    else:
        try:
            # 日付とファイル名でソートしてから書き込む
            all_rows_per_file.sort(key=lambda r: (r['date'], r['filename']))
            with open(output_per_file_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=headers_per_file)
                writer.writeheader()
                for data_row in all_rows_per_file:
                    writer.writerow(data_row)
            print(
                f"Successfully created per-file CSV: {output_per_file_csv_path}")
        except IOError:
            print(
                f"Error writing per-file CSV file to {output_per_file_csv_path}.")
        except Exception as e:
            print(
                f"An unexpected error occurred while writing per-file CSV: {e}.")

    # --- Write total CSV ---
    if not all_rows_total:
        print(
            f"No total data found to write to CSV for project '{project_name}'.")
        try:
            with open(output_total_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers_total)
                writer.writeheader()
            print(
                f"Empty total CSV file with headers created at: {output_total_csv_path}")
        except IOError:
            print(
                f"Error writing empty total CSV file to {output_total_csv_path}.")
    else:
        try:
            # 日付でソートしてから書き込む
            all_rows_total.sort(key=lambda r: r['date'])
            with open(output_total_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers_total)
                writer.writeheader()
                for data_row in all_rows_total:
                    writer.writerow(data_row)
            print(f"Successfully created total CSV: {output_total_csv_path}")
        except IOError:
            print(f"Error writing total CSV file to {output_total_csv_path}.")
        except Exception as e:
            print(
                f"An unexpected error occurred while writing total CSV: {e}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Process coverage JSONs for a project and emit per-file/total CSVs.')
    ap.add_argument('project_dir', help='Path to project directory that contains date subdirs with JSONs')
    ap.add_argument('--out', dest='out_root', default=None, help='Output root directory (default: script_dir/output_project_0802)')
    args = ap.parse_args()
    process_coverage_data(args.project_dir, output_root=args.out_root)
