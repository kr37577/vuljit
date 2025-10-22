# repo_commit_processor.py の変更案

import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from typing import List, Dict, Any, Set, Optional

import git  # For git.Repo and git.exc.GitCommandError
import argparse
import sys

from get_feature_commit_func import calculate_commit_metrics
from vccfinder_metrics_calculator import VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(module)s:%(lineno)d - %(message)s'
)

BASE_METRIC_KEYS = [
    "subsystems_changed", "directories_changed", "files_changed", "total_lines_changed",
    "lines_added", "lines_deleted", "total_prev_loc", "is_bug_fix", "past_bug_fixes",
    "entropy", "ndev", "age", "nuc", "exp", "rexp", "sexp",
    # "mean_days_since_creation", "mean_past_changes", "past_different_authors",
    # "author_past_contributions", "author_past_contributions_ratio",
    # "author_30days_past_contributions", "author_30days_past_contributions_ratio",
    # "author_workload", "days_after_creation", "touched_files", "number_of_hunks",
]
VCC_METRIC_KEYS = VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER
COMMIT_DATA_KEYS = [
    "commit_hash", "commit_datetime", "commit_message",
    "commit_change_file_path", "commit_change_file_path_filetered", "commit_patch", "commit_code_patch", "file_text",
]
WRAPPER_ADDED_KEYS = ["repo_path", "processing_error"]
ALL_CSV_FIELDNAMES = list(dict.fromkeys(
    WRAPPER_ADDED_KEYS + COMMIT_DATA_KEYS + BASE_METRIC_KEYS + VCC_METRIC_KEYS
))


def get_processed_commit_hashes(csv_filepath: str) -> Set[str]:
    processed_hashes: Set[str] = set()
    if not os.path.exists(csv_filepath):
        return processed_hashes
    try:
        with open(csv_filepath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'commit_hash' not in reader.fieldnames:  # CSVに'commit_hash'列がない場合をハンドル
                logging.warning(
                    f"CSV file {csv_filepath} does not have 'commit_hash' column. Cannot determine processed commits.")
                return processed_hashes
            for row in reader:
                if row.get('commit_hash'):
                    processed_hashes.add(row['commit_hash'])
    except Exception as e:
        logging.error(
            f"Error reading processed commit hashes from {csv_filepath}: {e}")
    return processed_hashes


def get_repository_commit_hashes(repo_path: str) -> List[str]:
    try:
        repo = git.Repo(repo_path)
        commit_hashes = [
            # 古い順にコミットハッシュを取得
            commit.hexsha for commit in repo.iter_commits(reverse=True)]
        if not commit_hashes:
            logging.warning(f"No commits found in repository: {repo_path}")
        return commit_hashes
    except git.exc.InvalidGitRepositoryError:
        logging.error(f"Invalid Git repository at {repo_path}")
    except git.exc.NoSuchPathError:
        logging.error(f"Repository path does not exist: {repo_path}")
    except Exception as e:
        logging.error(f"Error accessing repository {repo_path}: {e}")
    return []


def process_single_commit_wrapper(repo_path: str, commit_hash: str) -> Dict[str, Any]:
    logging.debug(
        f"Attempting to process commit {commit_hash} in repository {repo_path}")
    result_metrics: Dict[str, Any] = {
        key: None for key in ALL_CSV_FIELDNAMES}  # 全てのCSVフィールドで辞書を初期化
    result_metrics['repo_path'] = repo_path
    result_metrics['commit_hash'] = commit_hash
    result_metrics['processing_error'] = None

    try:
        calculated_data = calculate_commit_metrics(repo_path, commit_hash)
        if calculated_data:
            for key in ALL_CSV_FIELDNAMES:
                if key in calculated_data:
                    result_metrics[key] = calculated_data[key]

        if result_metrics.get('commit_datetime') is None or result_metrics.get('commit_message') is None:
            try:
                repo = git.Repo(repo_path)
                commit_obj = repo.commit(commit_hash)
                if result_metrics.get('commit_datetime') is None:
                    result_metrics['commit_datetime'] = commit_obj.committed_datetime.isoformat(
                    )
                if result_metrics.get('commit_message') is None:
                    result_metrics['commit_message'] = commit_obj.message
            except Exception as e_git_info:
                logging.warning(
                    f"Could not fetch additional commit info (datetime/message) for {commit_hash} in {repo_path}: {e_git_info}")
                result_metrics['processing_error'] = result_metrics.get(
                    'processing_error', '') + f"; GitInfoError: {e_git_info}"

    except Exception as e:
        logging.error(
            f"Unhandled error in process_single_commit_wrapper for {commit_hash} in {repo_path}: {e}", exc_info=True)
        result_metrics['processing_error'] = str(e)  # エラー情報を記録

    for key in ['commit_change_file_path']:
        if key in result_metrics and isinstance(result_metrics[key], list):
            result_metrics[key] = ';'.join(result_metrics[key])
    return result_metrics


def append_metrics_to_csv(metrics_data_list: List[Dict[str, Any]], csv_filepath: str, write_header: bool):
    if not metrics_data_list:
        return
    try:
        with open(csv_filepath, mode='a', newline='', encoding='utf-8') as f:  # ファイルを追記モードでオープン
            writer = csv.DictWriter(
                # extrasaction='ignore'で余分なキーを無視
                f, fieldnames=ALL_CSV_FIELDNAMES, extrasaction='ignore')
            if write_header:
                writer.writeheader()  # ヘッダーを書き込む
                logging.info(f"CSV header written to {csv_filepath}")
            writer.writerows(metrics_data_list)
        logging.info(
            f"Appended {len(metrics_data_list)} records to {csv_filepath}")
    except IOError as e:
        logging.error(f"IOError writing to CSV {csv_filepath}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error writing to CSV {csv_filepath}: {e}")

# process_repository 関数のシグネチャとロジックを変更


def process_repository(
    repo_path: str,
    # 名前変更: processed_hashes -> processed_hashes_set
    processed_hashes_set: Set[str],
    max_workers: Optional[int],
    output_csv_file: str,
    initial_header_needs_writing: bool
) -> bool:  # 戻り値を「更新後のヘッダー書き込みが必要か否か」のbool値に変更
    """
    Processes new commits in a single repository in parallel.
    Metrics for each commit are written to the CSV file as they are processed.
    """
    logging.info(f"Starting processing for repository: {repo_path}")
    all_commit_hashes = get_repository_commit_hashes(repo_path)

    new_commits_to_process = [
        ch for ch in all_commit_hashes if ch not in processed_hashes_set]

    if not new_commits_to_process:
        logging.info(f"No new commits to process in {repo_path}")
        return initial_header_needs_writing  # ヘッダー書き込み状態は変わらない

    logging.info(
        f"Found {len(new_commits_to_process)} new commits to process in {repo_path}")

    current_header_needs_writing = initial_header_needs_writing
    processed_count_in_repo = 0

    # ProcessPoolExecutorで並列処理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_commit_wrapper, repo_path, commit_hash): commit_hash
            for commit_hash in new_commits_to_process
        }

        for i, future in enumerate(as_completed(futures)):  # 完了した順に結果を取得
            commit_hash = futures[future]
            try:
                metrics = future.result()  # 個々のコミットのメトリクスを取得
                if metrics:
                    # メトリクスを即座にCSVに書き込む
                    append_metrics_to_csv(
                        # 単一メトリクスをリストで渡す
                        [metrics], output_csv_file, current_header_needs_writing)
                    if current_header_needs_writing:  # ヘッダーが書かれたらフラグを更新
                        current_header_needs_writing = False
                    # 正常処理時のみセットに追加
                    if metrics.get('commit_hash') and metrics.get('processing_error') is None:
                        processed_hashes_set.add(metrics['commit_hash'])
                    processed_count_in_repo += 1
                logging.debug(
                    f"Successfully processed and saved commit {commit_hash} from {repo_path} ({i+1}/{len(new_commits_to_process)})")
            except Exception as e:
                logging.error(
                    f"Future for commit {commit_hash} in {repo_path} generated an exception: {e}", exc_info=True)
                error_metric = {key: None for key in ALL_CSV_FIELDNAMES}
                error_metric['repo_path'] = repo_path
                error_metric['commit_hash'] = commit_hash
                error_metric['processing_error'] = f"FutureError: {e}"
                append_metrics_to_csv(
                    [error_metric], output_csv_file, current_header_needs_writing)
                if current_header_needs_writing:
                    current_header_needs_writing = False

    logging.info(
        f"Finished processing for repository: {repo_path}. Processed and saved metrics for {processed_count_in_repo} commits.")
    return current_header_needs_writing


# batch_process_repositories 関数のロジックを変更
def batch_process_repositories(
    repo_paths_list: List[str],
    output_csv_file: str,
    max_workers_per_repo: Optional[int] = None
):
    logging.info(
        f"Starting batch processing for {len(repo_paths_list)} repositories. Output CSV: {output_csv_file}")

    file_exists = os.path.exists(output_csv_file)
    header_needs_writing = not file_exists or (
        # ヘッダー書き込みが必要か判断
        file_exists and os.path.getsize(output_csv_file) == 0)

    # このセットは全リポジトリ処理を通じて共有・更新される
    global_processed_hashes = get_processed_commit_hashes(output_csv_file)
    logging.info(
        f"Found {len(global_processed_hashes)} already processed commit hashes in {output_csv_file}")

    for repo_path in repo_paths_list:
        if not os.path.isdir(repo_path):  # リポジトリパスの存在確認
            logging.warning(
                f"Repository path {repo_path} does not exist or is not a directory. Skipping.")
            continue

        # process_repository がヘッダーの状態を返し、それを次の呼び出しに使う
        # global_processed_hashes を渡し、内部で更新してもらう
        logging.info(
            f"Processing repository: {repo_path}. Current processed hashes count: {len(global_processed_hashes)}")
        header_needs_writing = process_repository(
            repo_path,
            global_processed_hashes,  # このセットが process_repository 内で更新される
            max_workers_per_repo,
            output_csv_file,
            header_needs_writing
        )
        logging.info(
            f"Finished repository: {repo_path}. Updated processed hashes count: {len(global_processed_hashes)}")

    logging.info("Batch processing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Git repositories to extract commit metrics and save them to a CSV file."
    )
    parser.add_argument(
        "repo_paths",
        metavar="REPO_PATH",
        type=str,
        nargs='+',  # シェルからは要素1つのリストとして渡される想定
        help="Path(s) to the LOCAL Git repository/repositories to analyze."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True, # 出力ディレクトリを必須に変更
        help="Directory to save the output CSV file."
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of worker threads per repository. Defaults to 4."
    )

    args = parser.parse_args()

    repositories_to_analyze = args.repo_paths # 例: ['cloned_repos/harfbuzz']
    output_directory = args.output_dir       # 例: './output/harfbuzz'
    workers_per_repo = args.workers

    # --- 出力ファイル名の修正 ---
    if not repositories_to_analyze:
        logging.error("No repositories specified to analyze.")
        sys.exit(1)
    
    # スクリプトはリポジトリごとに呼び出されるため、渡されるリストの最初の要素を使う
    # basename を使って、パスからリポジトリ名部分のみを取得
    first_repo_path = repositories_to_analyze[0]
    repo_basename_for_filename = os.path.basename(first_repo_path)
    output_filename = f"{repo_basename_for_filename}_commit_metrics_output_per_commit.csv"
 
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        try:
            # exist_ok=True で既に存在してもエラーにしない
            os.makedirs(output_directory, exist_ok=True)
            logging.info(
                f"Output directory '{output_directory}' created or already exists.")
        except OSError as e:
            logging.error(
                f"Failed to create output directory '{output_directory}': {e}")
            sys.exit(1)  # ディレクトリ作成に失敗したら終了

    # 完全な出力CSVファイルパスを構築
    csv_output_path = os.path.join(output_directory, output_filename)
    

    logging.info(f"Repositories to analyze: {repositories_to_analyze}")
    logging.info(f"Output CSV will be saved to: {csv_output_path}")
    logging.info(f"Number of workers per repository: {workers_per_repo}")

    # リポジトリパスの基本的な検証（ディレクトリであるか）
    valid_repo_paths = []
    for repo_path in repositories_to_analyze:
        if not os.path.isdir(repo_path):
            logging.warning(
                f"The provided path '{repo_path}' is not a directory or does not exist. Skipping.")
        else:
            # さらに git.Repo でアクセス可能かどうかの簡易チェックもここに入れられるが、
            # batch_process_repositories 内でもチェックされるため、ここでは省略も可。
            # ここでは、ディレクトリ存在チェックのみに留める。
            valid_repo_paths.append(repo_path)

    if not valid_repo_paths:
        logging.error("No valid repository paths provided or found. Exiting.")
        sys.exit(1)

    # バッチ処理の実行
    try:
        batch_process_repositories(
            repo_paths_list=valid_repo_paths,
            output_csv_file=csv_output_path,
            max_workers_per_repo=workers_per_repo
        )
        logging.info(
            f"Processing finished. Results potentially saved to {csv_output_path}")
    except Exception as e:
        logging.error(
            f"An error occurred during batch processing: {e}", exc_info=True)
        sys.exit(1)
