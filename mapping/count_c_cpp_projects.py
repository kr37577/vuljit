#!/usr/bin/env python3
import os
import csv
import argparse
from collections import Counter

def main():
    p = argparse.ArgumentParser(description="Count matches between cloned_repos dirs and CSV.directory_name")
    # Defaults prefer environment variables; fall back to repo-relative paths
    default_csv = os.environ.get(
        "VULJIT_PROJECT_MAPPING",
        os.path.join(os.path.dirname(__file__), "filtered_project_mapping.csv"),
    )
    # typical clone location used by the CLI patch-coverage pipeline
    default_repos = os.environ.get(
        "VULJIT_CLONED_REPOS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "intermediate", "cloned_repos"),
    )
    p.add_argument("--csv", default=default_csv)
    p.add_argument("--repos", default=default_repos)
    p.add_argument("--out-csv", default=None, help="optional: write per-row match results to this CSV")
    args = p.parse_args()

    # load directory names from CSV (keep duplicates to count rows)
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get("directory_name") or "").strip()
            rows.append({"project_id": r.get("project_id",""), "directory_name": name})

    # list top-level directories in repos path
    try:
        entries = os.listdir(args.repos)
    except FileNotFoundError:
        print(f"ERROR: repos directory not found: {args.repos}")
        return 2
    dirs = {d for d in entries if os.path.isdir(os.path.join(args.repos, d))}

    total_rows = sum(1 for r in rows if r["directory_name"]) 
    matched_rows = 0
    counter = Counter()  # counts per directory_name in CSV
    match_map = {}       # directory_name -> exists(bool)

    for r in rows:
        dn = r["directory_name"]
        if not dn:
            continue
        counter[dn] += 1
        exists = dn in dirs
        # mark matched row if exists
        if exists:
            matched_rows += 1
        # record existence (True if any occurrence exists)
        match_map[dn] = match_map.get(dn, False) or exists

    unique_total = len([d for d in counter])
    unique_matched = sum(1 for k,v in match_map.items() if v)
    unique_unmatched = unique_total - unique_matched

    # also compute unique project_id count that has at least one existing repo
    project_ids_all = set()
    project_ids_matched = set()
    for r in rows:
        pid = (r.get("project_id") or "").strip()
        dn = (r.get("directory_name") or "").strip()
        if pid:
            project_ids_all.add(pid)
        if dn and dn in dirs and pid:
            project_ids_matched.add(pid)

    print("CSV rows with non-empty directory_name:", total_rows)
    print("Rows matched (directory_name exists in cloned_repos):", matched_rows)
    print("Unique directory_name entries in CSV:", unique_total)
    print("Unique directory_name present in cloned_repos:", unique_matched)
    print("Unique directory_name NOT present in cloned_repos:", unique_unmatched)
    print("Unique project_id in CSV:", len(project_ids_all))
    print("Unique project_id with at least one matched repo:", len(project_ids_matched))
    print()

    if unique_unmatched:
        print("Unmatched directory_name (sample up to 50):")
        cnt = 0
        for k,v in sorted(match_map.items()):
            if not v:
                print(" -", k)
                cnt += 1
                if cnt >= 50:
                    print(" ... (truncated)")
                    break

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["directory_name","count_in_csv","exists_in_repos"])
            for dn, c in sorted(counter.items()):
                w.writerow([dn, c, "yes" if match_map.get(dn, False) else "no"])
        print()
        print("Per-directory results written to:", args.out_csv)

if __name__ == "__main__":
    main()
