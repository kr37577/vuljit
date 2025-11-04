#!/usr/bin/env python3
"""
コミットメトリクス抽出 -> 脆弱性ラベル付与 -> TF-IDF 付与を一括で実行するパイプライン。
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence

import pandas as pd
from git import Repo
from git.exc import GitError
from concurrent.futures import ThreadPoolExecutor, as_completed

from text_code_metrics.get_feature_commit_func import calculate_commit_metrics
from text_code_metrics.label import add_vcc_labels, load_vulnerabilities, NEW_LABEL_COLUMN
from text_code_metrics.vccfinder_metrics_calculator import VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER
from text_code_metrics.vccfinder_commit_message_metrics import add_commit_tfidf

BASE_METRIC_COLUMNS: List[str] = [
    "subsystems_changed",
    "directories_changed",
    "files_changed",
    "total_lines_changed",
    "lines_added",
    "lines_deleted",
    "total_prev_loc",
    "is_bug_fix",
    "past_bug_fixes",
    "entropy",
    "ndev",
    "age",
    "nuc",
    "exp",
    "rexp",
    "sexp",
]

COMMIT_DATA_COLUMNS: List[str] = [
    "commit_hash",
    "commit_datetime",
    "commit_message",
    "commit_change_file_path",
    "commit_change_file_path_filetered",
    "commit_patch",
    "commit_code_patch",
    "file_text",
]

WRAPPER_COLUMNS: List[str] = ["repo_path", "processing_error"]

CSV_FIELD_ORDER: List[str] = list(
    dict.fromkeys(WRAPPER_COLUMNS + COMMIT_DATA_COLUMNS + BASE_METRIC_COLUMNS + list(VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER))
)

TFIDF_COLUMN_PREFIX = "VCC_w"


def _format_float_legacy(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return value
        text = f"{value:.17g}"
        if "e" in text or "E" in text:
            text = f"{value:.17f}".rstrip("0").rstrip(".")
        if "." not in text and "e" not in text and "E" not in text:
            text += ".0"
        return text
    return value


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"日付の形式が不正です: {date_str} (期待: YYYYMMDD または YYYY-MM-DD)")


def collect_commits(
    repo: Repo,
    rev: str,
    since: Optional[datetime],
    until: Optional[datetime],
) -> List[str]:
    commits: List[str] = []
    for commit in repo.iter_commits(rev, reverse=True):
        committed_at = commit.committed_datetime.replace(tzinfo=None)
        if since and committed_at < since:
            continue
        if until and committed_at > until:
            continue
        commits.append(commit.hexsha)
    return commits


def compute_metrics(
    repo_path: Path,
    repo_path_display: str,
    commit_hashes: Sequence[str],
    max_workers: int,
) -> List[dict]:
    if not commit_hashes:
        return []

    metrics_results: List[dict] = []
    repo_path_str = str(repo_path.resolve())
    repo_path_display_str = str(repo_path_display)

    def _normalize_record(record: dict) -> dict:
        commit_dt = record.get("commit_datetime")
        if hasattr(commit_dt, "isoformat"):
            record["commit_datetime"] = commit_dt.isoformat().replace("T", " ")
        elif isinstance(commit_dt, str):
            record["commit_datetime"] = commit_dt.replace("T", " ")

        if record.get("commit_change_file_path") is None:
            record["commit_change_file_path"] = ""
        elif isinstance(record.get("commit_change_file_path"), list):
            record["commit_change_file_path"] = ";".join(record["commit_change_file_path"])

        filtered_value = record.get("commit_change_file_path_filetered")
        if isinstance(filtered_value, list):
            record["commit_change_file_path_filetered"] = str(filtered_value)

        if record.get("processing_error") is not None and not isinstance(record["processing_error"], str):
            record["processing_error"] = str(record["processing_error"])

        record["repo_path"] = repo_path_display_str
        return record

    def _empty_record(commit_hash: str, error_message: Optional[str] = None) -> dict:
        record = {column: None for column in CSV_FIELD_ORDER}
        record["repo_path"] = repo_path_display_str
        record["commit_hash"] = str(commit_hash)
        record["processing_error"] = error_message
        return _normalize_record(record)

    def _compute_single(commit_hash: str) -> dict:
        record = {column: None for column in CSV_FIELD_ORDER}
        record["repo_path"] = repo_path_display_str
        record["commit_hash"] = str(commit_hash)
        record["processing_error"] = None

        try:
            calculated = calculate_commit_metrics(repo_path_str, commit_hash)
            if calculated:
                for column in CSV_FIELD_ORDER:
                    if column in calculated:
                        record[column] = calculated[column]

            missing_datetime = record.get("commit_datetime") is None
            missing_message = record.get("commit_message") is None

            if missing_datetime or missing_message:
                try:
                    commit_obj = Repo(repo_path_str).commit(commit_hash)
                    if missing_datetime:
                        record["commit_datetime"] = commit_obj.committed_datetime.isoformat().replace("T", " ")
                    if missing_message:
                        record["commit_message"] = commit_obj.message
                except Exception as git_exc:  # pragma: no cover - 補助情報取得のみ
                    existing_error = record.get("processing_error")
                    git_error_message = f"GitInfoError: {git_exc}"
                    record["processing_error"] = (
                        f"{existing_error}; {git_error_message}" if existing_error else git_error_message
                    )
        except Exception as exc:
            record["processing_error"] = str(exc)

        return _normalize_record(record)

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_compute_single, ch): ch for ch in commit_hashes}
            for future in as_completed(future_map):
                commit_hash = future_map[future]
                try:
                    metrics_results.append(future.result())
                except Exception as exc:  # pragma: no cover - フォールバック記録のみ
                    print(f"警告: コミット {commit_hash} のメトリクス計算に失敗しました: {exc}")
                    metrics_results.append(_empty_record(commit_hash, f"FutureError: {exc}"))
    else:
        for commit_hash in commit_hashes:
            try:
                metrics_results.append(_compute_single(commit_hash))
            except Exception as exc:  # pragma: no cover
                print(f"警告: コミット {commit_hash} のメトリクス計算に失敗しました: {exc}")
                metrics_results.append(_empty_record(commit_hash, f"FutureError: {exc}"))

    return metrics_results


def build_pipeline(args: argparse.Namespace) -> None:
    repo_path_input = args.repo
    repo_path = Path(repo_path_input).resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"リポジトリパスが存在しません: {repo_path}")

    repo = Repo(repo_path)

    since_dt = parse_date(args.since)
    until_dt = parse_date(args.until)

    commit_hashes = collect_commits(repo, rev=args.rev, since=since_dt, until=until_dt)
    if not commit_hashes:
        print("対象期間内のコミットが見つかりませんでした。処理を終了します。")
        return

    output_dir = Path(args.metrics_dir) / args.project
    final_metrics_path = output_dir / f"{args.project}_commit_metrics_with_tfidf.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_metrics_df: Optional[pd.DataFrame] = None
    if final_metrics_path.exists() and not args.force:
        try:
            existing_metrics_df = pd.read_csv(final_metrics_path)
            existing_metrics_df = existing_metrics_df.drop(
                columns=[col for col in existing_metrics_df.columns if col.startswith(TFIDF_COLUMN_PREFIX)],
                errors="ignore",
            )
            if NEW_LABEL_COLUMN in existing_metrics_df.columns:
                existing_metrics_df = existing_metrics_df.drop(columns=[NEW_LABEL_COLUMN])
            if "commit_hash" in existing_metrics_df.columns:
                existing_metrics_df["commit_hash"] = existing_metrics_df["commit_hash"].astype(str)
        except Exception as exc:
            print(f"警告: 既存のメトリクスファイルを読み込めませんでした ({final_metrics_path}): {exc}")
            existing_metrics_df = None

    if args.force:
        commits_to_process = commit_hashes
        base_metrics_df = pd.DataFrame(columns=CSV_FIELD_ORDER)
    else:
        processed_hashes = (
            set(existing_metrics_df["commit_hash"])
            if existing_metrics_df is not None and "commit_hash" in existing_metrics_df.columns
            else set()
        )
        commits_to_process = [ch for ch in commit_hashes if ch not in processed_hashes]
        base_metrics_df = (
            existing_metrics_df.reindex(columns=CSV_FIELD_ORDER, fill_value=None).copy()
            if existing_metrics_df is not None
            else pd.DataFrame(columns=CSV_FIELD_ORDER)
        )

    if commits_to_process:
        print(f"対象コミット数: {len(commits_to_process)} 件。メトリクス計算を開始します...")
        metrics_records = compute_metrics(repo_path, repo_path_input, commits_to_process, max_workers=args.workers)
    else:
        print("新規に処理するコミットはありません。既存データを再利用します。")
        metrics_records = []

    new_metrics_df = pd.DataFrame(metrics_records)
    if not new_metrics_df.empty:
        new_metrics_df = new_metrics_df.reindex(columns=CSV_FIELD_ORDER)
        new_metrics_df["commit_hash"] = new_metrics_df["commit_hash"].astype(str)
        new_metrics_df["processing_error"] = new_metrics_df["processing_error"].fillna("")

    if base_metrics_df.empty and not args.force and existing_metrics_df is not None:
        base_metrics_df = existing_metrics_df.reindex(columns=CSV_FIELD_ORDER, fill_value=None)

    if not new_metrics_df.empty:
        metrics_df = pd.concat([base_metrics_df, new_metrics_df], ignore_index=True)
    else:
        metrics_df = base_metrics_df.copy()

    if metrics_df.empty:
        print("メトリクス計算結果が空です。処理を終了します。")
        return

    metrics_df = metrics_df.reindex(columns=CSV_FIELD_ORDER)
    metrics_df["commit_hash"] = metrics_df["commit_hash"].astype(str)
    metrics_df["processing_error"] = metrics_df["processing_error"].fillna("")
    metrics_df["repo_path"] = metrics_df["repo_path"].apply(
        lambda v: repo_path_input if not isinstance(v, str) or not v else v
    )
    def _normalize_datetime_value(value):
        if pd.isna(value):
            return ""
        if hasattr(value, "isoformat"):
            return value.isoformat().replace("T", " ")
        if isinstance(value, str):
            return value.replace("T", " ")
        return str(value)

    metrics_df["commit_datetime"] = metrics_df["commit_datetime"].apply(_normalize_datetime_value)

    order_map = {commit_hash: idx for idx, commit_hash in enumerate(commit_hashes)}
    metrics_df["__order"] = metrics_df["commit_hash"].map(order_map).fillna(float("inf"))
    metrics_df = metrics_df.sort_values("__order", kind="stable").drop(columns="__order")
    metrics_df = metrics_df.drop_duplicates(subset=["commit_hash"], keep="first", ignore_index=True)

    print("脆弱性ラベルを付与しています...")
    vulnerabilities_df = load_vulnerabilities(args.vuln_csv)
    labeled_df = add_vcc_labels(metrics_df, package_name=args.project, vulnerabilities=vulnerabilities_df)
    labeled_df["processing_error"] = labeled_df["processing_error"].fillna("")

    ordered_with_label = CSV_FIELD_ORDER.copy()
    if NEW_LABEL_COLUMN not in ordered_with_label:
        ordered_with_label.append(NEW_LABEL_COLUMN)
    ordered_with_label.extend(
        [col for col in labeled_df.columns if col not in ordered_with_label and not col.startswith(TFIDF_COLUMN_PREFIX)]
    )
    labeled_df = labeled_df.reindex(columns=ordered_with_label)

    print("コミットメッセージの TF-IDF 特徴量を計算しています...")
    tfidf_base_df = labeled_df.drop(
        columns=[col for col in labeled_df.columns if col.startswith(TFIDF_COLUMN_PREFIX)],
        errors="ignore",
    )
    final_df, feature_names = add_commit_tfidf(tfidf_base_df)
    final_df["processing_error"] = final_df["processing_error"].fillna("")

    tfidf_columns = [col for col in final_df.columns if col.startswith(TFIDF_COLUMN_PREFIX)]
    final_order = [col for col in ordered_with_label if col in final_df.columns]
    final_order.extend([col for col in final_df.columns if col not in final_order and col not in tfidf_columns])
    final_order.extend([col for col in tfidf_columns if col not in final_order])
    final_df = final_df.reindex(columns=final_order)

    for column in final_df.select_dtypes(include=["float64", "float32"]).columns:
        final_df[column] = final_df[column].apply(_format_float_legacy)

    final_df.to_csv(final_metrics_path, index=False)
    print(f"最終結果を '{final_metrics_path}' に保存しました。レコード数: {len(final_df)} 件。")
    if NEW_LABEL_COLUMN in final_df.columns:
        labeled_count = int(final_df[NEW_LABEL_COLUMN].fillna(False).astype(bool).sum())
        print(f"脆弱性導入コミット数: {labeled_count}")


def prepare_argument_parser() -> argparse.ArgumentParser:
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    default_metrics_dir = os.environ.get(
        "VULJIT_METRICS_DIR",
        os.path.join(repo_root, "datasets", "derived_artifacts", "commit_metrics"),
    )
    default_vuln_csv = os.environ.get(
        "VULJIT_VUL_CSV",
        os.path.join(repo_root, "datasets", "derived_artifacts", "vulnerability_reports", "oss_fuzz_vulnerabilities.csv"),
    )

    parser = argparse.ArgumentParser(
        description="コミットメトリクス抽出から TF-IDF 付与までを一括実行します。"
    )
    parser.add_argument("--project", required=True, help="OSS-Fuzz パッケージ名 / プロジェクト名")
    parser.add_argument("--repo", required=True, help="ローカル Git リポジトリのパス")
    parser.add_argument("--rev", default="HEAD", help="解析対象とするリビジョン (デフォルト: HEAD)")
    parser.add_argument("--since", help="解析開始日 (YYYYMMDD または YYYY-MM-DD)")
    parser.add_argument("--until", help="解析終了日 (YYYYMMDD または YYYY-MM-DD)")
    parser.add_argument("--metrics-dir", default=default_metrics_dir,
                        help=f"最終CSVを保存するベースディレクトリ (デフォルト: {default_metrics_dir})")
    parser.add_argument("--vuln-csv", default=default_vuln_csv,
                        help=f"脆弱性情報CSVのパス (デフォルト: {default_vuln_csv})")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                        help="メトリクス計算に使用する最大ワーカー数 (デフォルト: CPU コア数)")
    parser.add_argument("--force", action="store_true",
                        help="既存CSVが完全でも再計算する場合に指定します。")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = prepare_argument_parser()
    args = parser.parse_args(argv)
    try:
        build_pipeline(args)
    except (GitError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
