import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
# from dotenv import load_dotenv

# load_dotenv()  # Load environment variables from .env file if present

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VULNS_REL = Path("datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv")
DEFAULT_REPOS_REL = Path("datasets/raw/cloned_c_cpp_projects")
DEFAULT_PROJECT_METADATA_REL = Path("datasets/derived_artifacts/oss_fuzz_metadata/oss_fuzz_project_metadata.csv")
DEFAULT_VULNS_ARG = str(DEFAULT_VULNS_REL)
DEFAULT_REPOS_ARG = str(DEFAULT_REPOS_REL)
DEFAULT_PROJECT_METADATA_ARG = str(DEFAULT_PROJECT_METADATA_REL)
LOG_PREFIX = "[measure_detection_time]"

# GitHub API token (disabled by request; retain reference for future use)
HEADERS = {"Accept": "application/vnd.github.v3+json"}
# GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
# if GITHUB_TOKEN:
#     HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
# else:
#     print(f"{LOG_PREFIX} 警告: 環境変数 'GITHUB_TOKEN' が見つかりません。")
#     print(f"{LOG_PREFIX} GitHub APIのレート制限が厳しくなります（認証なしの場合、1時間あたり60リクエスト）。")
print(f"{LOG_PREFIX} GitHub API は未認証モードで実行されます（GITHUB_TOKEN 利用を無効化）。")

# Cache to avoid re-fetching data for the same commit (local or remote)
COMMIT_DATE_CACHE: dict[str, Optional[datetime]] = {}
REPO_PROJECT_MAP: Dict[str, List[str]] = {}
PROJECT_MAP_INITIALISED = False


def _strip_git_suffix(path: str) -> str:
    if path.endswith(".git"):
        return path[:-4]
    return path


def get_repo_path_from_url(repo_url: str) -> Optional[str]:
    """Extract 'owner/repo' for GitHub URLs only (API 用)."""
    if not isinstance(repo_url, str) or not repo_url:
        return None
    try:
        if repo_url.startswith("git@github.com:"):
            path = _strip_git_suffix(repo_url.split(":", 1)[1])
            return path if path.count("/") == 1 else None

        parsed_url = urlparse(repo_url)
        if parsed_url.netloc != "github.com":
            return None
        path = _strip_git_suffix(parsed_url.path.strip("/"))
        return path if path.count("/") == 1 else None
    except Exception as exc:  # pragma: no cover - defensive
        print(f"URL解析エラー ({repo_url}): {exc}")
        return None


def get_repo_identifiers(repo_url: str) -> List[str]:
    """Return possible repo identifiers for local clone resolution."""
    if not isinstance(repo_url, str) or not repo_url:
        return []
    parsed = urlparse(repo_url)
    path = _strip_git_suffix(parsed.path.strip("/"))
    if not path:
        return []
    segments = [seg for seg in path.split("/") if seg]
    identifiers: List[str] = []
    if len(segments) >= 2:
        identifiers.append("/".join(segments[-2:]))  # owner/repo
    if segments:
        identifiers.append(segments[-1])  # repo name only
    return identifiers


def select_commit_hash(raw_value: object) -> Optional[str]:
    """Return the first commit hash found in the raw introduced_commits value."""
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return None
    text = str(raw_value).strip()
    if not text or text == "--":
        return None
    separators = [";", ",", " "]
    for sep in separators:
        if sep in text:
            text = text.split(sep)[0]
            break
    return text.strip() or None


def parse_datetime_to_utc(value: str) -> Optional[datetime]:
    if not value or value == "--":
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def _normalise_repo_url(value: Optional[str]) -> str:
    """Normalise repository URLs for consistent matching."""

    if not value:
        return ""
    text = value.strip()
    if not text:
        return ""
    text = text.rstrip("/")
    if text.lower().endswith(".git"):
        text = text[:-4]
    return text.lower()


def load_repo_project_map(metadata_path: Optional[str]) -> Dict[str, List[str]]:
    """Load project metadata CSV and build repo->project mapping."""

    if not metadata_path:
        return {}
    path = Path(metadata_path)
    if not path.is_file():
        print(f"{LOG_PREFIX} 警告: プロジェクトメタデータCSVが見つかりません: {path}")
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"{LOG_PREFIX} 警告: メタデータCSVの読み込みに失敗しました ({path}): {exc}")
        return {}
    required_columns = {"project", "main_repo"}
    if not required_columns.issubset(df.columns):
        print(f"{LOG_PREFIX} 警告: メタデータCSVに必要な列がありません: {required_columns}")
        return {}

    mapping: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        repo_norm = _normalise_repo_url(str(row.get("main_repo", "")).strip())
        project = str(row.get("project", "")).strip()
        if not repo_norm or not project:
            continue
        mapping.setdefault(repo_norm, [])
        if project not in mapping[repo_norm]:
            mapping[repo_norm].append(project)
    if not mapping:
        print(f"{LOG_PREFIX} 警告: メタデータCSVから有効なマッピングが生成できませんでした: {path}")
    return mapping


def iter_local_repo_candidates(repo_url: str, repos_root: Path, repo_project_map: Optional[Dict[str, List[str]]] = None) -> Iterable[Path]:
    identifiers = get_repo_identifiers(repo_url)
    seen: set[Path] = set()
    for identifier in identifiers:
        owner_repo = identifier.split("/")
        if len(owner_repo) == 2:
            candidate = (repos_root / owner_repo[0] / owner_repo[1]).resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
        candidate = (repos_root / owner_repo[-1]).resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate
        candidate = (repos_root / identifier.replace("/", "__")).resolve()
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

    repo_norm = _normalise_repo_url(repo_url)
    if repo_norm and repo_project_map:
        for project in repo_project_map.get(repo_norm, []):
            candidate = (repos_root / project).resolve()
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def get_commit_date_from_local(repo_url: str, commit_hash: str, repos_root: Optional[Path]) -> Optional[datetime]:
    if repos_root is None or not repos_root.exists():
        return None
    cache_key = f"local::{repo_url}::{commit_hash}"
    if cache_key in COMMIT_DATE_CACHE:
        return COMMIT_DATE_CACHE[cache_key]

    for candidate in iter_local_repo_candidates(repo_url, repos_root, REPO_PROJECT_MAP if PROJECT_MAP_INITIALISED else None):
        if not candidate.is_dir():
            continue
        try:
            result = subprocess.run(
                ["git", "-C", str(candidate), "show", "-s", "--format=%cI", commit_hash],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            continue
        commit_iso = result.stdout.strip()
        commit_dt = parse_datetime_to_utc(commit_iso)
        if commit_dt:
            COMMIT_DATE_CACHE[cache_key] = commit_dt
            return commit_dt

    COMMIT_DATE_CACHE[cache_key] = None
    return None


def get_commit_date_from_github(repo_path: str, commit_hash: str) -> Optional[datetime]:
    if not repo_path or not commit_hash:
        return None

    cache_key = f"github::{repo_path}/{commit_hash}"
    if cache_key in COMMIT_DATE_CACHE:
        return COMMIT_DATE_CACHE[cache_key]

    api_url = f"https://api.github.com/repos/{repo_path}/commits/{commit_hash}"
    try:
        response = requests.get(api_url, headers=HEADERS, timeout=20)

        # Basic rate limit check
        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_time = datetime.fromtimestamp(
                int(response.headers["X-RateLimit-Reset"]),
                tz=timezone.utc,
            )
            wait_seconds = (reset_time - datetime.now(timezone.utc)).total_seconds() + 5
            print(f"レート制限に達しました。リセット時刻 ({reset_time.isoformat()}) まで約 {int(wait_seconds)} 秒待機します...")
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            response = requests.get(api_url, headers=HEADERS, timeout=20)

        response.raise_for_status()
        commit_data = response.json()
        commit_dt = parse_datetime_to_utc(commit_data["commit"]["author"]["date"])
        COMMIT_DATE_CACHE[cache_key] = commit_dt
        return commit_dt

    except requests.exceptions.HTTPError as exc:
        print(f"\nHTTPエラー ({exc.response.status_code}) - コミット日時取得失敗: {repo_path}/{commit_hash}")
    except requests.exceptions.RequestException as exc:
        print(f"\nリクエストエラー - コミット日時取得失敗 ({commit_hash}): {exc}")

    COMMIT_DATE_CACHE[cache_key] = None
    return None


def main(vulns_csv: str, issues_csv: str, output_csv: str, repos_root: Optional[str], project_metadata_csv: Optional[str]) -> None:
    """
    Calculates the time between vulnerability introduction and detection.
    """
    global REPO_PROJECT_MAP, PROJECT_MAP_INITIALISED

    if not PROJECT_MAP_INITIALISED:
        REPO_PROJECT_MAP = load_repo_project_map(project_metadata_csv)
        PROJECT_MAP_INITIALISED = True

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

    detection_times: List[Optional[int]] = []
    commit_dates_list: List[Optional[str]] = []
    reported_dates_list: List[Optional[str]] = []

    repos_root_path = Path(repos_root).resolve() if repos_root else None

    for _, row in merged_df.iterrows():
        repo_url = row.get("repo", "")
        commit_hash = select_commit_hash(row.get("introduced_commits"))
        reported_date = parse_datetime_to_utc(str(row.get("reported_date", "")).strip())

        if reported_date:
            reported_dates_list.append(datetime_to_iso(reported_date))
        else:
            reported_dates_list.append(None)

        if not commit_hash:
            detection_times.append(None)
            commit_dates_list.append(None)
            continue

        commit_date = get_commit_date_from_local(repo_url, commit_hash, repos_root_path)
        if commit_date is None:
            repo_path = get_repo_path_from_url(repo_url)
            if not repo_path:
                print(f"警告: GitHubリポジトリのパスを抽出できませんでした。スキップします: {repo_url}")
            else:
                # NOTE: GitHub API へのフォールバック取得は一時的に無効化
                pass

        commit_dates_list.append(datetime_to_iso(commit_date))

        if not commit_date or not reported_date:
            detection_times.append(None)
            continue

        detection_times.append((reported_date - commit_date).days)

    merged_df["detection_time_days"] = detection_times
    merged_df["commit_date_utc"] = commit_dates_list
    merged_df["reported_date_utc"] = reported_dates_list

    # Save the results
    merged_df.to_csv(
        output_csv, index=False)
    print(f"Results saved to {output_csv}")


def _resolve_path(value: str, *, treat_as_default: bool) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    base = REPO_ROOT if treat_as_default else Path.cwd()
    return str((base / path).resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="脆弱性の導入から検出までの時間を計算します。"
    )
    parser.add_argument(
        "--vulns-csv",
        default=DEFAULT_VULNS_ARG,
        help=f"脆弱性情報を含むCSVファイルのパス (既定: {DEFAULT_VULNS_REL})",
    )
    parser.add_argument(
        "--issues-csv",
        default=None,
        help="Issue情報と報告日を含むCSVファイルのパス (例: issue_redirect_mapping_selenium.csv)",
    )
    parser.add_argument(
        "-o", "--output",
        default="detection_time_results.csv",
        help="結果を保存する出力CSVファイルのパス"
    )
    parser.add_argument(
        "--repos-root",
        default=DEFAULT_REPOS_ARG,
        help=f"ローカルにクローンしたリポジトリのルートディレクトリ (既定: {DEFAULT_REPOS_REL})",
    )
    parser.add_argument(
        "--project-metadata",
        default=DEFAULT_PROJECT_METADATA_ARG,
        help=f"OSS-FuzzプロジェクトメタデータCSVのパス (既定: {DEFAULT_PROJECT_METADATA_REL})",
    )
    parser.add_argument(
        "legacy_args",
        nargs="*",
        help="（互換用）位置引数: [vulns_csv] [issues_csv]",
    )
    args = parser.parse_args()

    vulns_csv_arg = args.vulns_csv
    issues_csv_arg = args.issues_csv
    repos_root_arg = args.repos_root
    project_metadata_arg = args.project_metadata

    if args.legacy_args:
        if len(args.legacy_args) == 1:
            issues_csv_arg = args.legacy_args[0]
        elif len(args.legacy_args) >= 2:
            vulns_csv_arg = args.legacy_args[0]
            issues_csv_arg = args.legacy_args[1]

    if not issues_csv_arg:
        parser.error("--issues-csv を指定するか、互換用位置引数で issues_csv を渡してください。")

    vulns_csv_path = _resolve_path(vulns_csv_arg, treat_as_default=vulns_csv_arg == DEFAULT_VULNS_ARG)
    issues_csv_path = _resolve_path(issues_csv_arg, treat_as_default=False)
    repos_root_path = _resolve_path(repos_root_arg, treat_as_default=repos_root_arg == DEFAULT_REPOS_ARG)
    project_metadata_path = _resolve_path(
        project_metadata_arg,
        treat_as_default=project_metadata_arg == DEFAULT_PROJECT_METADATA_ARG,
    )
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    if output_path.parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    main(vulns_csv_path, issues_csv_path, str(output_path), repos_root_path, project_metadata_path)
