#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Select top-N projects by the number of days with is_vcc == True
and copy those project directories to a destination, preserving structure.

- Counts unique dates with positives per project (by default using a date-like column).
- Falls back to counting positive rows when no date column is available.
-
Usage example:
  python vuljit/prediction/select_top_projects_by_isvcc.py \
    --base-dir /work/riku-ka/vuljit/data \
    --out-dir /work/riku-ka/vuljit/data_top10_isvcc \
    --top 10

Notes:
  - Expects per-project subdirectories inside --base-dir.
  - Prefer a daily CSV named "<project>_daily_aggregated_metrics_with_predictions.csv".
    If not found, searches for any CSV containing an "is_vcc" column within the project dir.
  - Date columns tried (in order): merge_date, date, commit_date_date, commit_date, day
"""

import argparse
import csv
import glob
import os
import shutil
from typing import Optional, Tuple, List

import pandas as pd


PREFERRED_DAILY_FILENAME = "{project}_daily_aggregated_metrics_with_predictions.csv"
DATE_CANDIDATE_COLS = [
    "merge_date",
]


def list_projects(base_dir: str) -> List[str]:
    return [
        name for name in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, name)) and not name.startswith(".")
    ]


def _find_columns_case_insensitive(columns, targets):
    lower_map = {str(c).lower(): c for c in columns}
    found = []
    for t in targets:
        c = lower_map.get(t.lower())
        if c is not None:
            found.append(c)
    return found


def _detect_is_vcc_col(header_cols) -> Optional[str]:
    for c in header_cols:
        if str(c).lower() == "is_vcc":
            return c
    return None


def _detect_date_col(header_cols) -> Optional[str]:
    for t in DATE_CANDIDATE_COLS:
        for c in header_cols:
            if str(c).lower() == t.lower():
                return c
    return None


def _count_positive_days_from_csv(csv_path: str) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Return (positive_days_or_rows, used_date_col, used_isvcc_col) or (None, None, None) if not applicable."""
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception:
        return None, None, None

    isvcc_col = _detect_is_vcc_col(header.columns)
    if isvcc_col is None:
        return None, None, None

    date_col = _detect_date_col(header.columns)
    usecols = [isvcc_col] + ([date_col] if date_col else [])

    try:
        df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)
    except Exception:
        return None, None, None

    col = df[isvcc_col]
    # Normalize to boolean-like mask
    if pd.api.types.is_bool_dtype(col):
        mask = col.fillna(False)
    else:
        # Try numeric first (0/1)
        col_num = pd.to_numeric(col, errors="coerce")
        if col_num.notna().any():
            mask = col_num.fillna(0).astype(float) > 0.0
        else:
            s = col.astype(str).str.strip().str.lower()
            mask = s.isin(["1", "true", "t", "yes", "y"])

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce").dt.date
        pos_days = int(dates[mask].dropna().nunique())
        return pos_days, date_col, isvcc_col
    else:
        # No date column → count rows with is_vcc True
        pos_rows = int(mask.sum())
        return pos_rows, None, isvcc_col


def _preferred_csv_for_project(project_dir: str, project: str) -> Optional[str]:
    pref = os.path.join(project_dir, PREFERRED_DAILY_FILENAME.format(project=project))
    if os.path.exists(pref):
        return pref
    # fallback: try any csv at top-level
    top_csvs = glob.glob(os.path.join(project_dir, "*.csv"))
    # prioritise files that look like daily/aggregated
    top_csvs_sorted = sorted(
        top_csvs,
        key=lambda p: (
            0 if "daily" in os.path.basename(p).lower() or "aggregated" in os.path.basename(p).lower() else 1,
            os.path.basename(p).lower(),
        ),
    )
    for p in top_csvs_sorted:
        return p
    # as a last resort, look recursively, but avoid massive scans by limiting depth a bit
    rec_csvs = glob.glob(os.path.join(project_dir, "**/*.csv"), recursive=True)
    rec_csvs_sorted = sorted(rec_csvs)[:50]  # safety cap
    for p in rec_csvs_sorted:
        return p
    return None


def copy_project_tree(src_dir: str, dst_dir: str, overwrite: bool = False):
    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            raise FileExistsError(f"Destination already exists: {dst_dir}")
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    shutil.copytree(src_dir, dst_dir)


def main():
    ap = argparse.ArgumentParser(description="Select top-N projects by is_vcc-positive days and copy them.")
    ap.add_argument("--base-dir", default="./data", help="プロジェクトごとのデータが入ったベースディレクトリ")
    ap.add_argument("--out-dir", required=True, help="上位N件のプロジェクトを保存する出力ディレクトリ")
    ap.add_argument("--top", type=int, default=10, help="選抜するプロジェクト数（上位N件）")
    ap.add_argument("--overwrite", action="store_true", help="出力先に既存ディレクトリがある場合は削除して上書き")
    args = ap.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir
    top_n = max(1, int(args.top))

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir not found or not a directory: {base_dir}")
    os.makedirs(out_dir, exist_ok=True)

    projects = list_projects(base_dir)
    if not projects:
        print(f"No projects found under: {base_dir}")
        return

    print(f"Scanning {len(projects)} projects under: {base_dir}")

    results = []  # (project, pos_count, used_csv, used_date_col)
    for pj in projects:
        pj_dir = os.path.join(base_dir, pj)
        csv_path = _preferred_csv_for_project(pj_dir, pj)
        if not csv_path:
            results.append((pj, 0, None, None))
            continue
        pos, date_col, isvcc_col = _count_positive_days_from_csv(csv_path)
        if pos is None:
            pos = 0
        results.append((pj, int(pos), csv_path, date_col))

    results.sort(key=lambda x: x[1], reverse=True)
    selected = results[:min(top_n, len(results))]

    print(f"Top {len(selected)} projects by positive days (is_vcc=True):")
    for pj, cnt, csv_path, date_col in selected:
        hint = f" (date={date_col})" if date_col else ""
        print(f" - {pj}: {cnt}{hint}")

    # Copy selected project directories
    for pj, _, _, _ in selected:
        src = os.path.join(base_dir, pj)
        dst = os.path.join(out_dir, pj)
        copy_project_tree(src, dst, overwrite=args.overwrite)

    # Emit summary CSV
    summary_csv = os.path.join(out_dir, "top_projects_by_isvcc.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["project", "positive_days", "source_dir", "used_csv", "used_date_col"])
        for pj, cnt, csv_path, date_col in selected:
            w.writerow([pj, cnt, os.path.join(base_dir, pj), csv_path or "", date_col or ""])

    print(f"\nDone. Copied projects to: {out_dir}")
    print(f"Summary: {summary_csv}")


if __name__ == "__main__":
    main()

