import os
import argparse
import csv
from google.cloud import storage
from google.cloud.storage import Client, Bucket
from google.api_core import exceptions as google_exceptions
from datetime import datetime, timedelta
import concurrent.futures
from typing import Set, List


def read_packages_from_csv(csv_path: str, column_index: int = 0) -> Set[str]:
    """CSVファイルからパッケージ名のリストを読み込み、重複を除外して返す。"""
    packages = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header row if necessary (optional)
            # next(reader, None)
            for row in reader:
                if len(row) > column_index:
                    package_name = row[column_index].strip()
                    if package_name:
                        packages.add(package_name)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
    return packages


def download_single_file(storage_client: Client, bucket_name: str, blob_name: str, destination_file_name: str) -> bool:
    """GCSから単一のファイルをダウンロードする。"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Ensure the local directory exists
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

        print(
            f"Attempting to download gs://{bucket_name}/{blob_name} to {destination_file_name}...")
        blob.download_to_filename(destination_file_name)
        print(f"Successfully downloaded to {destination_file_name}")
        return True
    except google_exceptions.NotFound:
        # Don't print error if file simply doesn't exist for a given subdir/filename combo
        # print(f"Info: File not found at gs://{bucket_name}/{blob_name}")
        return False
    except Exception as e:
        print(f"Error downloading gs://{bucket_name}/{blob_name}: {e}")
        return False


def download_reports(package_names: Set[str], start_date_str: str, end_date_str: str, download_dir: str, max_workers: int) -> None:
    """
    指定された期間とパッケージリストに基づき、GCSから [date].json ファイルを並列ダウンロードする。

    Args:
        package_names: ダウンロード対象のパッケージ名のセット。
        start_date_str: ダウンロード開始日の日付文字列 (YYYYMMDD)。
        end_date_str: ダウンロード終了日の日付文字列 (YYYYMMDD)。
        download_dir: ダウンロードしたファイルを保存するルートディレクトリ。
        max_workers: 並列ダウンロードに使用する最大ワーカー数。
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYYMMDD.")
        return

    if start_date > end_date:
        print("Error: Start date must be before or equal to end date.")
        return

    # 匿名クライアントを作成
    try:
        storage_client: Client = storage.Client.create_anonymous_client()
        bucket_name = "oss-fuzz-coverage"
        # bucket object is not serializable/sharable between processes directly,
        # but client and bucket_name are fine for threads.

        download_tasks = []

        for package_name in package_names:
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                # Construct the blob name for the JSON file
                json_filename = f"{date_str}.json"
                blob_name = f"{package_name}/srcmap/{json_filename}"

                # Construct local path preserving structure
                destination_file_name = os.path.join(
                    download_dir, package_name, "json", json_filename)

                # Add task details to the list
                download_tasks.append(
                    (bucket_name, blob_name, destination_file_name))

                current_date += timedelta(days=1)

        # Execute downloads in parallel
        if download_tasks:
            print(
                f"\nStarting parallel download of {len(download_tasks)} json files using up to {max_workers} workers...")
            # We pass the client itself to the workers in a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create futures, passing bucket_name instead of bucket object
                futures = [executor.submit(download_single_file, storage_client, bucket, blob, dest)
                           for bucket, blob, dest in download_tasks]

                # Wait for completion (optional: add progress tracking here)
                for future in concurrent.futures.as_completed(futures):
                    # You can check future.result() or future.exception() if needed
                    pass
            print("All download tasks submitted.")
        else:
            print("No files found matching the criteria to download.")

    except Exception as e:
        print(f"An overall error occurred: {e}")


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
    default_download_dir = os.environ.get(
        "VULJIT_SRCDOWN_DIR",
        os.path.join(repo_root, "datasets", "raw", "srcmap_json"),
    )
    default_csv_path = os.environ.get(
        "VULJIT_VUL_CSV",
        os.path.join(
            repo_root,
            "datasets",
            "derived_artifacts",
            "vulnerability_reports",
            "oss_fuzz_vulnerabilities.csv",
        ),
    )

    default_start_date = os.environ.get("VULJIT_START_DATE", "20160101")
    default_end_date = os.environ.get("VULJIT_END_DATE", "20250802")

    parser = argparse.ArgumentParser(
        description="Download [date].json files for multiple projects over a date range from GCS oss-fuzz-coverage bucket.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=default_csv_path,
        help=f"Path to CSV file containing project names in the specified column (default: {default_csv_path})")
    parser.add_argument(
        "start_date",
        nargs="?",
        default=default_start_date,
        help=f"Start date string (YYYYMMDD) (default: {default_start_date})")
    parser.add_argument(
        "end_date",
        nargs="?",
        default=default_end_date,
        help=f"End date string (YYYYMMDD) (default: {default_end_date})")
    # Removed --filenames argument
    # parser.add_argument("--filenames", nargs='+', required=True,
    #                     help="List of filenames to download from each report subdirectory (e.g., index.html fuzz_report.html).")
    parser.add_argument("-d", "--dir", default=default_download_dir,
                        help=f"Root directory to save the downloaded files (default: {default_download_dir})")
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel download workers (default: number of CPU cores or 4)")
    parser.add_argument("--csv-column", type=int, default=2,  # Changed default from 0 to 2
                        help="0-based index of the column containing package names in the CSV file (default: 2)")

    args = parser.parse_args()

    # Read package names from CSV using the specified or default column index
    package_names = read_packages_from_csv(args.csv_file, args.csv_column)

    if not package_names:
        print("No package names found in the CSV file. Exiting.")
    else:
        print(f"Found {len(package_names)} unique package names.")
        # Start the download process, removed args.filenames
        download_reports(package_names, args.start_date,
                         args.end_date, args.dir, args.workers)

    print("\nScript finished.")
