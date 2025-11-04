"""
This module contains functions to extract features from a commit.
コミットから特徴量を抽出するための関数を含むモジュール
"""
from typing import Dict, Any
import subprocess
import tempfile
import json
import git
from git import Repo, GitCommandError
from git.exc import InvalidGitRepositoryError
import os
import csv
import logging
from datetime import datetime, timedelta
import math
from typing import Dict, Set, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Lock, Manager
import pandas as pd
import re
import shutil
import statistics as stat
import scipy.stats as stats
from . import vccfinder_metrics_calculator
import json
import pickle


CODE_FILE_EXTENSIONS = [
    '.c', '.cc', '.cpp', '.cxx', '.c++',  # C/C++ ソースファイル
    '.h', '.hh', '.hpp', '.hxx', '.h++'   # C/C++ ヘッダーファイル
]

EXTENSIONS_FINDER = re.compile(r'\.(\w+)')


def get_changed_files(diff_commit):
    changed_files = set()
    for diff in diff_commit:
        if diff.a_path:
            changed_files.add(diff.a_path)
        if diff.b_path:
            changed_files.add(diff.b_path)

    filtered_files = []
    for actual_file_path in changed_files:
        _, ext = os.path.splitext(actual_file_path)
        if ext.lower() in CODE_FILE_EXTENSIONS:
            filtered_files.append(actual_file_path)
    return filtered_files


def count_subsystems(changed_files):
    subsystems = set()
    for file in changed_files:
        subsystem = file.split('/')[0] if '/' in file else file
        subsystems.add(subsystem)
    return len(subsystems)


def count_directories(changed_files):
    directories = set()
    for file in changed_files:
        directory = os.path.dirname(file)
        directories.add(directory)
    return len(directories)


def get_line_changes(diff_commit):
    total_lines_changed = 0
    lines_added = 0
    lines_deleted = 0
    for diff in diff_commit:
        diff_content = diff.diff
        if isinstance(diff_content, bytes):
            diff_content = diff_content.decode('utf-8', errors='ignore')
        elif not isinstance(diff_content, str):
            diff_content = str(diff_content)
        diff_lines = diff_content.split('\n')

        for line in diff_lines:
            if line.startswith('+') and not line.startswith('+++'):
                lines_added += 1
            elif line.startswith('-') and not line.startswith('---'):
                lines_deleted += 1
    total_lines_changed = lines_added + lines_deleted
    return total_lines_changed, lines_added, lines_deleted


def calculate_entropy(diff_commit):
    file_changes = {}
    total_changes = 0

    for diff in diff_commit:
        # file_path がフィルタリングに引っ掛かるかどうかを確認
        file_path = diff.a_path if diff.a_path else diff.b_path

        if not file_path:
            continue

        _, ext = os.path.splitext(file_path)
        if ext.lower() not in CODE_FILE_EXTENSIONS:
            continue
        # ここまで追加

        diff_content = diff.diff
        if isinstance(diff_content, bytes):
            diff_content = diff_content.decode('utf-8', errors='ignore')
        elif not isinstance(diff_content, str):
            diff_content = str(diff_content)

        diff_lines = diff_content.split('\n')
        changes_in_file = 0
        for line in diff_lines:
            if (line.startswith('+') and not line.startswith('+++')) or (line.startswith('-') and not line.startswith('---')):
                changes_in_file += 1

        file_path = diff.a_path if diff.a_path else diff.b_path
        file_changes[file_path] = changes_in_file
        total_changes += changes_in_file

    entropy = 0
    for changes in file_changes.values():
        if changes > 0 and total_changes > 0:
            p = changes / total_changes
            entropy -= p * math.log2(p)

    return entropy


def get_prev_file_line_count(repo, commit_sha, file_path):
    try:
        blob = repo.git.show(f"{commit_sha}^:{file_path}")
        line_count = len(blob.split('\n'))
        return line_count
    except git.exc.GitCommandError:
        return 0


def is_bug_fix(commit_message):
    bugfixing_keywords = {"bug", "defect", "fix",
                          "error", "repair", "patch", "issue", "exception"}
    msg = commit_message.lower()
    return any(k in msg for k in bugfixing_keywords)


def count_past_bug_fixes(repo, file_path, until_commit_sha):
    bug_fix_count = 0
    commits = list(repo.iter_commits(paths=file_path, rev=until_commit_sha))
    for commit in commits:
        if is_bug_fix(commit.message):
            bug_fix_count += 1
    return bug_fix_count


def count_ndev(repo, file_paths, until_commit_sha):
    developers = set()
    for file_path in file_paths:
        try:
            commits = list(repo.iter_commits(
                paths=file_path, rev=until_commit_sha))
            for commit in commits:
                developers.add(commit.author.email)
        except Exception as e:
            logging.error(f"NDEV計算中のエラー ({file_path}): {e}")
            continue
    return len(developers)


def calculate_age(repo, commit_date, file_paths, until_commit_sha):
    total_age = 0
    count = 0
    for file_path in file_paths:
        try:
            commits = list(repo.iter_commits(
                paths=file_path, rev=until_commit_sha))
            if len(commits) > 1:
                last_commit_date = commits[1].committed_datetime
                age = (commit_date - last_commit_date).days
                total_age += age
                count += 1
        except Exception as e:
            logging.error(f"AGE計算中のエラー ({file_path}): {e}")
            continue
    if count > 0:
        average_age = total_age / count
    else:
        average_age = 0
    return average_age


def count_nuc(repo, file_paths, commit_sha):
    commit = repo.commit(commit_sha)
    commit_files = set(commit.stats.files.keys())
    return len(commit_files)


def calculate_exp(repo, author_email, until_commit_sha):
    try:
        commits = list(repo.iter_commits(
            author=author_email, rev=until_commit_sha))
        return len(commits)-1
    except Exception as e:
        logging.error(f"EXP計算中のエラー ({author_email}): {e}")
        return 0


def calculate_rexp(repo, author_email, commit_date, until_commit_sha, days=90):
    start_date = commit_date - timedelta(days=days)
    rexp = 0
    try:
        commits = list(repo.iter_commits(author=author_email,
                       rev=until_commit_sha, since=start_date))
        for commit in commits:
            commit_time = datetime.fromtimestamp(commit.committed_date)
            year_diff = commit_date.year - commit_time.year
            int_year_diff = int(year_diff)
            commmit_whight = 1 / (1 + int_year_diff)
            rexp += commmit_whight
    except Exception as e:
        logging.error(f"REXP計算中のエラー ({author_email}): {e}")
        return 0

    return rexp - 1


def calculate_sexp(repo, author_email, subsystems, until_commit_sha):
    subsystem_commits = []
    for subsystem in subsystems:
        try:
            path = subsystem + '/*' if subsystem else ''
            commits = list(repo.iter_commits(
                paths=path, author=author_email, rev=until_commit_sha))
            temp_subsystem_commits = len(commits) - 1
            subsystem_commits.append(temp_subsystem_commits)
        except Exception as e:
            logging.error(f"SEXP計算中のエラー ({subsystem}): {e}")
            continue
    if len(subsystem_commits) == 0:
        return 0
    return max(subsystem_commits)


def file_exists_in_commit(repo, commit_sha, file_path):
    try:
        repo.git.show(f'{commit_sha}:{file_path}')
        return True
    except git.exc.GitCommandError:
        return False


def previous_changes(repo, until_commit_sha, file_path):
    # file_pathに対してuntil_commit_shaより前の変更履歴を取得(コミットのハッシュ一覧を古い順で取得）
    try:
        commits = list(repo.iter_commits(
            paths=file_path, rev=until_commit_sha))
        commits.reverse()  # 古い順にする
        return [c.hexsha for c in commits[:-1]]  # 現コミットを除く過去コミット
    except:
        return []


def get_file_creation_date(repo, file_path):
    commits = list(repo.iter_commits(paths=file_path, reverse=True))
    if commits:
        return commits[0].committed_datetime
    return None


def count_all_commits_until(repo, until_commit_sha):
    """対象コミット（until_commit_sha）までのコミット数を返す（将来のコミットは含まない）"""
    return sum(1 for _ in repo.iter_commits(until=until_commit_sha))


def count_commits_in_last_days(repo, commit_date, days=30):
    """
    指定されたコミット日時(commit_date)を基準に、過去days日間のコミット数を返す。
    これにより、コミット時点以降の情報が混入しないようにする。
    """
    since_date = commit_date - timedelta(days=days)
    commits = list(repo.iter_commits(since=since_date, until=commit_date))
    return len(commits)


def get_hunk_count(diff_content):
    diff_lines = diff_content.split('\n')
    hunk_count = 0
    for dl in diff_lines:
        if dl.startswith('@@'):
            hunk_count += 1
    return hunk_count


def mean_days_since_creation(repo, changed_files, commit_date):
    total_days = 0
    count = 0
    for f in changed_files:
        creation_date = get_file_creation_date(repo, f)
        if creation_date:
            days = (commit_date - creation_date).days
            total_days += days
            count += 1
    return total_days / count if count > 0 else 0


def mean_of_past_changes(repo, changed_files, until_commit_sha):
    total_changes = 0
    count = 0
    for f in changed_files:
        try:
            commits = list(repo.iter_commits(paths=f, rev=until_commit_sha))
            changes = max(len(commits)-1, 0)
            total_changes += changes
            count += 1
        except:
            continue
    return total_changes / count if count > 0 else 0


def past_different_authors(repo, changed_files, until_commit_sha):
    authors = set()
    for f in changed_files:
        try:
            commits = list(repo.iter_commits(paths=f, rev=until_commit_sha))
            for c in commits:
                authors.add(c.author.email)
        except:
            continue
    return len(authors)


def author_past_contributions(repo, author_email, until_commit_sha):
    try:
        commits = list(repo.iter_commits(
            author=author_email, rev=until_commit_sha))
        return len(commits)-1 if len(commits) > 0 else 0
    except:
        return 0


def author_past_contributions_ratio(repo, author_email, until_commit_sha):
    total_c = count_all_commits_until(repo, until_commit_sha)
    author_c = author_past_contributions(repo, author_email, until_commit_sha)
    return author_c / total_c if total_c > 0 else 0


def author_30days_past_contributions(repo, author_email, commit_date, until_commit_sha, days=30):
    try:
        commits = list(repo.iter_commits(author=author_email, rev=until_commit_sha,
                       since=commit_date - timedelta(days=days), until=commit_date))
        return len(commits)
    except Exception as e:
        logging.error(f"author_30days_past_contributions エラー: {e}")
        return 0


def author_30days_past_contributions_ratio(repo, author_email, commit_date, until_commit_sha, days=30):
    total_30d = count_commits_in_last_days(repo, commit_date, days=days)
    author_30d = author_30days_past_contributions(
        repo, author_email, commit_date, until_commit_sha, days=days)
    return author_30d / total_30d if total_30d > 0 else 0


# author_workload内で使う
def get_all_developer_contributions_in_period(repo, commit_date, until_commit_sha, days=30):
    """
    指定された期間内にコミットした全ての開発者とそのコミット数を取得する
    """
    # commit_date を until_commit_sha より前の日付に調整する必要がある場合がある
    # ここでは簡単のため、commit_date を基準とする
    since_date = commit_date - timedelta(days=days)

    # この期間のコミットを全て取得
    all_commits_in_period = list(repo.iter_commits(
        since=since_date, until=commit_date))

    developer_commits = {}
    for commit_obj in all_commits_in_period:
        author_email = commit_obj.author.email
        developer_commits[author_email] = developer_commits.get(
            author_email, 0) + 1

    return developer_commits


def author_workload(repo, author_email, commit_date, until_commit_sha, days=30):
    # 1. 対象期間の全開発者のコミット数を取得
    all_dev_contributions = get_all_developer_contributions_in_period(
        repo, commit_date, until_commit_sha, days
    )

    if not all_dev_contributions:
        return 0.0

    # 2. 特定の作者のコミット数を取得
    author_commits_in_period = all_dev_contributions.get(author_email, 0)

    # 3. 全開発者のコミット数のリストを作成
    commit_counts_list = list(all_dev_contributions.values())

    if not commit_counts_list:
        return 0.0

    # 4. パーセンタイル値を計算
    percentile = stats.percentileofscore(
        commit_counts_list, author_commits_in_period, kind='rank')

    return percentile / 100.0

# def author_workload(repo, author_email, commit_date, until_commit_sha, days=30):
#     author_30d = author_30days_past_contributions(
#         repo, author_email, commit_date, until_commit_sha, days)
#     total_30d = count_commits_in_last_days(repo, commit_date, days=days)
#     return author_30d / total_30d if total_30d > 0 else 0


def days_after_creation(repo, commit_date):
    commits = list(repo.iter_commits(reverse=True))
    if len(commits) == 0:
        return 0
    first_commit_date = commits[0].committed_datetime
    return (commit_date - first_commit_date).days


def touched_files(changed_files):
    exclude_patterns = ['.md', '.txt', 'test/']
    valid_files = [f for f in changed_files if not any(
        pat in f for pat in exclude_patterns)]
    return len(valid_files)


def read_file(filename):
    data = {}
    with open(filename, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['monorail_id']] = row
    return data


def write_data(data, filename):
    with open(filename, 'w+', newline='', encoding='utf-8') as write_file:
        if len(data) > 0:
            writer = csv.DictWriter(write_file, fieldnames=data[list(
                data.keys())[0]].keys(), dialect="excel")
            writer.writeheader()
            for entry in data:
                writer.writerow(data[entry])
        data.clear()


def project_detect(data: Dict[str, Dict[str, str]]) -> Set[str]:
    project_set = set()
    for entry in data.values():
        repo = entry.get('repo', '')
        project = ""

        try:
            if repo != "":
                project_2 = entry['repo'].split(
                    'github.com/')[1].rstrip('.git')
            project_set.add(project)
        except (IndexError, ValueError) as e:
            logging.error(f"URLの解析に失敗しました: {entry['url']} - {e}")
    return project_set


def calculate_commit_metrics(repo_path: str, commit_hash: str) -> Dict[str, Any]:
    metrics = {
        # A Large-Scale Empirical Study of Just-in-Time Quality Assurance
        "subsystems_changed": 0,
        "directories_changed": 0,
        "files_changed": 0,
        "total_lines_changed": 0,
        "lines_added": 0,
        "lines_deleted": 0,
        "total_prev_loc": 0,
        "is_bug_fix": False,
        "past_bug_fixes": 0,
        "entropy": 0.0,
        "ndev": 0,
        "age": 0.0,
        "nuc": 0,
        "exp": 0,
        "rexp": 0.0,
        "sexp": 0,
        # Just-in-time software vulnerability detection: Are we there yet?
        # "mean_days_since_creation": 0.0,
        # "mean_past_changes": 0.0,
        # "past_different_authors": 0,
        # "author_past_contributions": 0,
        # "author_past_contributions_ratio": 0.0,
        # "author_30days_past_contributions": 0,
        # "author_30days_past_contributions_ratio": 0.0,
        # "author_workload": 0.0,
        # "days_after_creation": 0,
        # "touched_files": 0,
        # "number_of_hunks": 0,
        
        # Revisiting the VCCFinder approach for the identification of vulnerability-contributing commits
        # Security-sensitive features (s1-s24)
        "VCC_s1_nb_added_sizeof": 0,
        "VCC_s2_nb_removed_sizeof": 0,
        "VCC_s3_diff_sizeof": 0,
        "VCC_s4_sum_sizeof": 0,
        "VCC_s5_nb_added_continue": 0,
        "VCC_s6_nb_removed_continue": 0,
        "VCC_s7_nb_added_break": 0,
        "VCC_s8_nb_removed_break": 0,
        "VCC_s9_nb_added_INTMAX": 0,
        "VCC_s10_nb_removed_INTMAX": 0,
        "VCC_s11_nb_added_goto": 0,
        "VCC_s12_nb_removed_goto": 0,
        "VCC_s13_nb_added_define": 0,
        "VCC_s14_nb_removed_define": 0,
        "VCC_s15_nb_added_struct": 0,
        "VCC_s16_nb_removed_struct": 0,
        "VCC_s17_diff_struct": 0,
        "VCC_s18_sum_struct": 0,
        "VCC_s19_nb_added_offset": 0,
        "VCC_s20_nb_removed_offset": 0,
        "VCC_s21_nb_added_void": 0,
        "VCC_s22_nb_removed_void": 0,
        "VCC_s23_diff_void": 0,
        "VCC_s24_sum_void": 0,

        # Code-fix features (f1-f29)
        "VCC_f1_sum_file_change": 0,
        "VCC_f2_nb_added_loop": 0,
        "VCC_f3_nb_removed_loop": 0,
        "VCC_f4_diff_loop": 0,
        "VCC_f5_sum_loop": 0,
        "VCC_f6_nb_added_if": 0,
        "VCC_f7_nb_removed_if": 0,
        "VCC_f8_diff_if": 0,
        "VCC_f9_sum_if": 0,
        "VCC_f10_nb_added_line": 0,
        "VCC_f11_nb_removed_line": 0,
        "VCC_f12_diff_line": 0,
        "VCC_f13_sum_line": 0,
        "VCC_f14_nb_added_paren": 0,
        "VCC_f15_nb_removed_paren": 0,
        "VCC_f16_diff_paren": 0,
        "VCC_f17_sum_paren": 0,
        "VCC_f18_nb_added_bool": 0,
        "VCC_f19_nb_removed_bool": 0,
        "VCC_f20_diff_bool": 0,
        "VCC_f21_sum_bool": 0,
        "VCC_f22_nb_added_assignement": 0,
        "VCC_f23_nb_removed_assignement": 0,
        "VCC_f24_diff_assignement": 0,
        "VCC_f25_sum_assignement": 0,
        "VCC_f26_nb_added_function": 0,
        "VCC_f27_nb_removed_function": 0,
        "VCC_f28_diff_function": 0,
        "VCC_f29_sum_function": 0,
    }
    # VCCFinderの生のメトリクスも初期化に追加
    for metric_name in vccfinder_metrics_calculator.VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER:
        if "raw" in metric_name and metric_name not in metrics:  # 生メトリクスも確実に初期化されるように
            metrics[metric_name] = 0

    diff_commit_obj = None  # リポジトリへのアクセスに失敗した場合に備えて初期化
    try:
        repo = git.Repo(repo_path)
        # 'commit' から 'commit_obj' に変更（変数名の衝突回避）
        commit_obj = repo.commit(commit_hash)
        if not commit_obj.parents:  # 親コミットが存在するか確認
            logging.warning(
                f"コミット {commit_hash} には親コミットが存在しません。メトリクス計算をスキップします。")
            return metrics  # 初期化されたメトリクスを返す

        # commit_obj を一貫して使用
        diff_commit_obj = commit_obj.parents[0].diff(
            commit_obj, create_patch=True)

    except Exception as e:
        logging.error(
            f"Gitリポジトリまたはコミットへのアクセス中にエラーが発生しました ({commit_hash}): {e}", exc_info=True)
        return metrics  # 初期化された（または部分的に計算された）メトリクスを返す

    metrics = commit_data(repo_path, commit_hash, diff_commit_obj,
                          commit_obj.committed_datetime, metrics)

    try:
        logging.info(f"メトリクス計算開始: {commit_hash}")

        changed_files = get_changed_files(diff_commit_obj)

        metrics['files_changed'] = len(changed_files)
        metrics['subsystems_changed'] = count_subsystems(changed_files)
        metrics['directories_changed'] = count_directories(changed_files)

        # TODO：total_lines_changed, lines_added, lines_deleted はいらない
        # VCCFinderのメトリクス計算でf10-f13で取得済み
        total_lines_changed, lines_added, lines_deleted = get_line_changes(diff_commit_obj)
        metrics['total_lines_changed'] = total_lines_changed
        metrics['lines_added'] = lines_added
        metrics['lines_deleted'] = lines_deleted

        metrics['entropy'] = calculate_entropy(diff_commit_obj)

        total_prev_loc = 0
        past_bug_fixes = 0
        for file_path in changed_files:
            if not file_exists_in_commit(repo, commit_hash, file_path):
                continue
            prev_loc = get_prev_file_line_count(repo, commit_hash, file_path)
            total_prev_loc += prev_loc
            past_bug_fixes += count_past_bug_fixes(repo, file_path, commit_hash)

        metrics['total_prev_loc'] = total_prev_loc
        metrics['past_bug_fixes'] = past_bug_fixes

        commit_date = commit_obj.committed_datetime
        metrics['is_bug_fix'] = is_bug_fix(commit_obj.message)
        metrics['ndev'] = count_ndev(repo, changed_files, commit_hash)
        metrics['age'] = calculate_age(repo, commit_date, changed_files, commit_hash)
        metrics['nuc'] = count_nuc(repo, changed_files, commit_hash)

        author_email = commit_obj.author.email
        metrics['exp'] = calculate_exp(repo, author_email, commit_hash)
        metrics['rexp'] = calculate_rexp(repo, author_email, commit_date, commit_hash)

        # fp は file_path の略
        subsystems = {fp.split('/')[0] for fp in changed_files if '/' in fp}
        metrics['sexp'] = calculate_sexp(repo, author_email, subsystems, commit_hash)

        # metrics['author_workload'] = author_workload(
        #     repo, author_email, commit_date, commit_hash, 30)
        # metrics['days_after_creation'] = days_after_creation(
        #     repo, commit_date)
        # metrics['touched_files'] = len(changed_files)

        # hunk_count = 0
        # for diff_item in diff_commit_obj:  # 'diff' を 'diff_item' に変更
        #     diff_content = diff_item.diff
        #     if isinstance(diff_content, bytes):
        #         diff_content = diff_content.decode('utf-8', errors='ignore')
        #     # get_hunk_count が同モジュール内にあると仮定
        #     hcount = get_hunk_count(diff_content)
        #     hunk_count += hcount
        # metrics['number_of_hunks'] = hunk_count

        # --- VCCFinder メトリクス計算 ---
        logging.info(f"VCCFinderメトリクス計算開始: {commit_hash}")
        if diff_commit_obj is not None:  # diff_commit_obj が正常に作成されたことを確認
            filtered_patch_lines = vccfinder_metrics_calculator.calculate_and_populate_vccfinder_metrics(
                metrics,  # 主要な metrics 辞書を渡す
                diff_commit_obj,
                commit_hash
            )
        else:
            logging.warning(
                f"diff_commit_obj が {commit_hash} に対して None です。VCCFinderメトリクスをスキップします。")
        # --- VCCFinder メトリクス計算終了 ---

    except Exception as e:
        logging.error(
            f"メトリクスの計算中にエラーが発生しました ({commit_hash}): {e}", exc_info=True)
        # metrics 辞書にはエラー発生前に計算された値が含まれる

    return metrics


def convert_diff_list_to_text(diff_list: List[git.diff.Diff]) -> str:
    """
    git.diff.Diff オブジェクトのリストを受け取り、
    それら全ての差分情報を連結した1つのテキストとして返します。

    Args:
        diff_list: git.diff.Diff オブジェクトのリスト。
                   通常、repo.commit().diff(..., create_patch=True) の結果。

    Returns:
        連結された差分情報のテキスト。
        差分がない場合やエラー時は空文字列やエラー情報を含むことがあります。
    """
    full_diff_text = []
    if not diff_list:
        return "No diffs found in the provided list."

    for i, diff_obj in enumerate(diff_list):
        try:
            diff_text = diff_obj.diff
            if isinstance(diff_text, bytes):
                diff_text = diff_text.decode('utf-8', errors='ignore')

            file_path_info = ""
            if diff_obj.new_file:
                file_path_info = f"new file: {diff_obj.b_path}"
            elif diff_obj.deleted_file:
                file_path_info = f"deleted file: {diff_obj.a_path}"
            elif diff_obj.renamed:  # renamed_file 属性は古いGitPythonにあったもの。renamed を使う
                file_path_info = f"renamed: {diff_obj.a_path} to {diff_obj.b_path}"
            elif diff_obj.a_path == diff_obj.b_path:  # 通常の変更 (パスが同じ)
                file_path_info = f"modified: {diff_obj.a_path}"
            else:  # パスが異なるがrenameではない場合 (稀だがモード変更のみなど)
                file_path_info = f"a_path: {diff_obj.a_path}, b_path: {diff_obj.b_path}"

            header = f"--- Diff {i+1}/{len(diff_list)} for {file_path_info} ---\n"
            full_diff_text.append(header)
            full_diff_text.append(diff_text)
            full_diff_text.append(
                "\n--------------------------------------------------\n")

        except Exception as e:
            error_message = f"Error processing diff object {diff_obj} (a_path: {diff_obj.a_path}, b_path: {diff_obj.b_path}): {e}\n"
            full_diff_text.append(error_message)

    return "".join(full_diff_text)


def all_changed_files(diff_commit_obj: List[git.diff.Diff]) -> List[str]:
    """
    コミット内で変更された全てのファイルのパスを取得
    """
    changed_files = set()
    for diff in diff_commit_obj:
        if diff.a_path:
            changed_files.add(diff.a_path)
        if diff.b_path and diff.b_path not in changed_files:
            changed_files.add(diff.b_path)
    return list(changed_files)


def commit_data(repo_path, commit_hash: str, diff_commit_obj: List[git.diff.Diff], commit_datetime: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    メトリクスにコミット自体のデータを追加する関数
    """
    repo = git.Repo(repo_path)
    filtered_patch_lines = vccfinder_metrics_calculator.filter_diff_for_code_files_from_gitpython(
        diff_commit_obj)
    # TODO：他のファイルでうまく処理できているか確認
    commit_code_patch_str = "\n".join(
        [line.rstrip('\n') for line in filtered_patch_lines])
    # print(f"commit_code_patch_str: {commit_code_patch_str}...")  # デバッグ用

    commit_diff_text = convert_diff_list_to_text(diff_commit_obj)

    commit_change_file_path = all_changed_files(diff_commit_obj)
    # ';' 区切りで保存する（既存データと互換）
    commit_change_file_path_str = ';'.join(commit_change_file_path)
    commit_change_file_path_filetered = get_changed_files(diff_commit_obj)
    # フィルタ済みは一覧文字列をそのまま保持
    commit_change_file_path_filetered_str = str(commit_change_file_path_filetered)

    commit = repo.commit(commit_hash)
    commit_message = commit.message

    commit_file_text = json.dumps(get_content_of_changed_files_in_commit(
        repo, commit_hash, diff_commit_obj), ensure_ascii=False)

    commit_data = {
        "commit_hash": commit_hash,
        "commit_datetime": commit_datetime,
        "commit_message": commit_message,
        "commit_change_file_path": commit_change_file_path_str,
        "commit_change_file_path_filetered": commit_change_file_path_filetered_str,
        "commit_patch": commit_diff_text,
        "commit_code_patch": commit_code_patch_str,  # codeファイルのみにフィルタリングされた差分
        "file_text": commit_file_text,
    }
    commit_data.update(metrics)
    return commit_data

# TODO: コミット内で単純に削除されたファイルの扱い


def get_content_of_changed_files_in_commit(repo: git.Repo, commit_hash: str, diff_commit_obj: List[git.diff.Diff]) -> Dict[str, str]:
    """
    特定のコミットで変更（追加，修正，または削除）されたファイルの状態を抽出
    """
    files_content: Dict[str, str] = {}

    for diff_entry in diff_commit_obj:
        file_path: str | None = None
        blob_to_read: git.Blob | None = None

        if diff_entry.b_path:  # ファイルが現在のコミットに存在する（つまり、追加または修正された可能性がある）
            file_path = diff_entry.b_path
            blob_to_read = diff_entry.b_blob
        # diff_entry.a_path が存在し、diff_entry.b_path が None の場合、それは削除されたファイルです。

        if file_path and blob_to_read:
            # このブロックは、ファイルが現在のコミットに存在し、かつ内容 (blob) がある場合
            try:
                content_bytes: bytes = blob_to_read.data_stream.read()
                files_content[file_path] = content_bytes.decode(
                    'utf-8', errors='ignore')
            except AttributeError:
                # blob_to_read が None ではないが、.data_stream.read() が AttributeError を出す場合
                logging.warning(
                    f"コミット {commit_hash} のファイル {file_path} のblobを読み取れませんでした（AttributeError）。"
                    f"これは特殊なファイルタイプや予期せぬblobオブジェクト構造の可能性があります。")
            except Exception as e:
                logging.error(
                    f"コミット {commit_hash} のファイル {file_path} の内容読み取り中にエラーが発生しました: {e}")
                files_content[file_path] = f"ファイルの読み取りエラー: {e}"
        elif diff_entry.deleted_file:
            # ファイルがこのコミットで削除された場合
            deleted_file_path: str | None = diff_entry.a_path  # 削除されたファイルのパス
            if deleted_file_path:
                files_content[deleted_file_path] = "[DELETED IN THIS COMMIT]"
                logging.info(
                    f"コミット {commit_hash} のファイル {deleted_file_path} は削除としてマークされました。")
            else:
                # 通常、deleted_file フラグが立っている diff エントリは a_path を持つはずです。
                logging.warning(
                    f"コミット {commit_hash} で削除フラグのあるdiffエントリにa_pathがありません: change_type={diff_entry.change_type}")

        elif file_path and not blob_to_read:
            logging.warning(
                f"コミット {commit_hash} のファイル {file_path} は現在のコミットに存在しますが、"
                f"読み取り可能なコンテンツblob (b_blob) がありません。サブモジュールや特殊なファイルタイプの可能性があります。")

    return files_content


# テスト用のサンプルコード (スクリプトが直接実行され、他のファイルにアクセス可能な場合)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 指定されたリポジトリパスとコミットハッシュを使用
    test_repo_path = "clone/apache___arrow"
    test_commit_hash = "071ea1bc245446f6ee257bb1ed7d056c46c2868e"
    project_id = "apache_arrow_test"  # プロジェクト識別子（適宜変更）

    logging.info(f"--- 指定されたリポジトリとコミットでテスト実行 ---")
    logging.info(f"リポジトリパス: {test_repo_path}")
    logging.info(f"コミットハッシュ: {test_commit_hash}")

    # リポジトリの存在チェック
    if not os.path.isdir(test_repo_path):
        logging.error(f"テスト用リポジトリのパスが見つかりません: {test_repo_path}")
        logging.error("テストを実行する前に、指定されたパスにリポジトリをクローンしてください。")
        logging.error(
            "例: git clone https://github.com/apache/arrow.git clone/apache___arrow")
    elif not os.path.isdir(os.path.join(test_repo_path, ".git")):
        logging.error(
            f"指定されたパスは有効なGitリポジトリではありません ('.git'フォルダが見つかりません): {test_repo_path}")
    else:
        calculated_metrics = calculate_commit_metrics(
            repo_path=test_repo_path,
            commit_hash=test_commit_hash
        )
        print("\n計算されたメトリクス:")
        if calculated_metrics:
            for key, value in calculated_metrics.items():
                print(f"  {key}: {value}")
        else:
            print("メトリクスの計算に失敗しました。")
