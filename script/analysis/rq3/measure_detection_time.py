import pandas as pd
import requests
import os
import time
import argparse
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

# GitHub API token
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
else:
    print("警告: 環境変数 'GITHUB_TOKEN' が見つかりません。")
    print("GitHub APIのレート制限が厳しくなります（認証なしの場合、1時間あたり60リクエスト）。")

# Cache to avoid re-fetching data for the same commit
COMMIT_DATE_CACHE = {}


def get_repo_path_from_url(repo_url):
    """Extracts 'owner/repo' from a git URL, ensuring it's a GitHub repo."""
    if not isinstance(repo_url, str) or not repo_url:
        return None
    try:
        # Handle git@github.com:owner/repo.git format
        if repo_url.startswith("git@github.com:"):
            path = repo_url.split(":")[1]
            if path.endswith('.git'):
                path = path[:-4]
            if path.count('/') == 1:
                return path

        parsed_url = urlparse(repo_url)
        path = parsed_url.path.strip('/')
        if path.endswith('.git'):
            path = path[:-4]

        # For non-github.com URLs, we cannot use the GitHub API.
        # We will only proceed if the domain is github.com.
        if parsed_url.netloc == 'github.com' and path.count('/') == 1:
            return path

        # For other URLs like git://git.ghostscript.com/mupdf.git,
        # we explicitly state they are not supported by this script's logic.
        if parsed_url.netloc != 'github.com':
            # This check is important because the rest of the script relies on the GitHub API.
            return None

    except Exception as e:
        print(f"URL解析エラー ({repo_url}): {e}")
    return None


def get_commit_date(repo_path, commit_hash):
    """Get the date of a commit using the GitHub API, with caching and rate limit awareness."""
    if not repo_path or not commit_hash:
        return None

    # Check cache first
    cache_key = f"{repo_path}/{commit_hash}"
    if cache_key in COMMIT_DATE_CACHE:
        return COMMIT_DATE_CACHE[cache_key]

    api_url = f"https://api.github.com/repos/{repo_path}/commits/{commit_hash}"
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=20)

        # Basic rate limit check
        if response.status_code == 403 and 'X-RateLimit-Remaining' in response.headers and response.headers['X-RateLimit-Remaining'] == '0':
            reset_time = datetime.fromtimestamp(
                int(response.headers['X-RateLimit-Reset']), tz=timezone.utc)
            wait_seconds = (
                reset_time - datetime.now(timezone.utc)).total_seconds() + 5
            print(
                f"レート制限に達しました。リセット時刻 ({reset_time.isoformat()}) まで約 {int(wait_seconds)} 秒待機します...")
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            # Retry the request once after waiting
            response = requests.get(api_url, headers=HEADERS, timeout=20)

        response.raise_for_status()
        commit_data = response.json()
        commit_date_str = commit_data["commit"]["author"]["date"]
        # The date is in ISO 8601 format with 'Z' for UTC.
        commit_dt = datetime.fromisoformat(
            commit_date_str.replace("Z", "+00:00"))

        # Store in cache
        COMMIT_DATE_CACHE[cache_key] = commit_dt
        return commit_dt

    except requests.exceptions.HTTPError as e:
        print(
            f"\nHTTPエラー ({e.response.status_code}) - コミット日時取得失敗: {repo_path}/{commit_hash}")
    except requests.exceptions.RequestException as e:
        print(f"\nリクエストエラー - コミット日時取得失敗 ({commit_hash}): {e}")

    # Cache the failure as None so we don't retry
    COMMIT_DATE_CACHE[cache_key] = None
    return None


def main(vulns_csv, issues_csv, output_csv):
    """
    Calculates the time between vulnerability introduction and detection.
    """
    print("脆弱性データとIssueデータを読み込んでいます...")
    try:
        vulns_df = pd.read_csv(vulns_csv)
        issues_df = pd.read_csv(issues_csv)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません - {e.filename}")
        return

    # Rename columns for merging
    issues_df = issues_df.rename(columns={"issue_id": "monorail_id"})

    # Merge the two dataframes
    merged_df = pd.merge(vulns_df, issues_df, on="monorail_id")

    # Calculate detection time
    detection_times = []
    commit_dates_list = []  # コミット日を保存するリストを追加
    for index, row in merged_df.iterrows():
        repo_url = row["repo"]
        commit_hash = row["introduced_commits"]
        reported_date_str = row["reported_date"]

        if pd.isna(commit_hash) or reported_date_str == "--":
            detection_times.append(None)
            commit_dates_list.append(pd.NaT)  # 日付がない場合はNaTを追加
            continue

        # Convert repo URL to repo path (owner/repo)
        repo_path = get_repo_path_from_url(repo_url)
        if not repo_path:
            print(f"警告: GitHubリポジトリのパスを抽出できませんでした。スキップします: {repo_url}")
            detection_times.append(None)
            commit_dates_list.append(pd.NaT)
            continue

        commit_date = get_commit_date(repo_path, commit_hash)
        commit_dates_list.append(
            commit_date if commit_date else pd.NaT)  # 取得したコミット日を追加

        if not commit_date:
            detection_times.append(None)
            continue

        try:
            # The date is in ISO 8601 format with 'Z' for UTC.
            reported_date = datetime.fromisoformat(
                reported_date_str.replace("Z", "+00:00"))
            detection_time = (reported_date - commit_date).days
            detection_times.append(detection_time)
        except (ValueError, TypeError):
            detection_times.append(None)

    merged_df["detection_time_days"] = detection_times
    merged_df["commit_date"] = commit_dates_list  # 新しい列としてコミット日を追加

    # Save the results
    merged_df.to_csv(
        output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="脆弱性の導入から検出までの時間を計算します。"
    )
    parser.add_argument(
        "vulns_csv", help="脆弱性情報を含むCSVファイルのパス (例: oss_fuzz_vulns.csv)"
    )
    parser.add_argument(
        "issues_csv", help="Issue情報と報告日を含むCSVファイルのパス (例: issue_redirect_mapping_selenium.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="detection_time_results.csv",
        help="結果を保存する出力CSVファイルのパス"
    )
    args = parser.parse_args()

    main(args.vulns_csv, args.issues_csv, args.output)
