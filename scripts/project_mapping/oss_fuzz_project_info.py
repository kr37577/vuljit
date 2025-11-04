#!/usr/bin/env python3
"""
Clone the oss-fuzz repository (if necessary) and extract metadata from
projects/<name>/project.yaml.

Usage example:
    python oss_fuzz_project_info.py --project abseil-cpp --commit <hash>

When --commit is provided the script checks out that revision after ensuring
the repository is cloned. Otherwise it leaves the current checkout untouched.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_REPO_URL = "https://github.com/google/oss-fuzz.git"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = REPO_ROOT / "datasets" / "raw" / "oss-fuzz"
DEFAULT_OUT = REPO_ROOT / "datasets" / "derived_artifacts" / "oss_fuzz_metadata" / "oss_fuzz_project_metadata.csv"
DEFAULT_VULN_CSV = REPO_ROOT / "datasets" / "derived_artifacts" / "vulnerability_reports" / "oss_fuzz_vulnerabilities.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "datasets" / "derived_artifacts" / "oss_fuzz_metadata" / "c_cpp_vulnerability_summary.csv"
CSV_FIELD_ORDER = ["project", "language", "main_repo", "homepage", "primary_contact"]

def run_git(args: Iterable[str], cwd: Path) -> str:
    """Run a git command and return its stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def ensure_repo(repo_dir: Path, repo_url: str, commit: str | None) -> Path:
    """Clone oss-fuzz if missing and optionally check out the requested commit."""
    if repo_dir.exists():
        if not (repo_dir / ".git").is_dir():
            raise RuntimeError(f"{repo_dir} exists but is not a git repository.")
        if commit:
            # Ensure the commit is available before checking out.
            run_git(["fetch", "origin", commit], repo_dir)
            run_git(["checkout", commit], repo_dir)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        clone_args = ["git", "clone", repo_url, str(repo_dir)]
        if commit:
            # Shallow clone that still allows checkout of the commit if it is a ref.
            clone_args = ["git", "clone", "--origin", "origin", repo_url, str(repo_dir)]
        subprocess.run(clone_args, check=True, cwd=str(repo_dir.parent))
        if commit:
            run_git(["checkout", commit], repo_dir)
    return repo_dir


def parse_project_yaml(
    path: Path,
    required_keys: Iterable[str],
    optional_keys: Iterable[str] | None = None,
) -> Dict[str, str]:
    """Extract simple key/value pairs from project.yaml without extra dependencies."""
    optional_keys = set(optional_keys or [])
    wanted = {key: None for key in required_keys}
    for key in optional_keys:
        wanted.setdefault(key, None)

    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key not in wanted or wanted[key] is not None:
                continue
            value = value.strip()
            if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
                value = value[1:-1]
            wanted[key] = value
    missing = [k for k, v in wanted.items() if v is None and k not in optional_keys]
    if missing:
        raise KeyError(f"{path} missing keys: {', '.join(missing)}")
    return {k: v for k, v in wanted.items() if v is not None}


def normalize_repo(url: str | None) -> str:
    """Normalize repository URLs for consistent matching."""
    if not url:
        return ""
    norm = url.strip()
    if not norm:
        return ""
    norm = norm.rstrip("/")
    if norm.lower().endswith(".git"):
        norm = norm[:-4]
    return norm.lower()


def extract_day(value: str | None) -> str:
    """Convert an ISO-like timestamp string to YYYY-MM-DD."""
    if not value:
        return ""
    raw = value.strip()
    if not raw:
        return ""
    cleaned = raw.replace("Z", "+00:00")
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    # fallback: take leading 10 characters if looks like date
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return ""


def summarize_vulnerabilities(
    metadata_entries: Dict[str, Dict[str, str]],
    vuln_csv: Path,
    summary_out: Path | None,
    collect_commit_days: bool = False,
    day_field: str = "published",
    include_fixed_commits: bool = False,
) -> tuple[Counter[str], Dict[str, str], int]:
    """Join vulnerability CSV with metadata to count C/C++ project vulnerabilities."""
    repo_map: Dict[str, List[str]] = defaultdict(list)
    for project, entry in metadata_entries.items():
        repo = normalize_repo(entry.get("main_repo"))
        if repo:
            repo_map[repo].append(project)

    if not vuln_csv.is_file():
        print(
            f"WARNING: Vulnerability CSV not found: {vuln_csv}. Skipping summary.",
            file=sys.stderr,
        )
        return Counter(), {}, 0

    counts: Counter[str] = Counter()
    matched_rows = 0
    commit_days: Dict[str, str] = {}
    with vuln_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo = normalize_repo(row.get("repo"))
            if not repo:
                continue
            projects = repo_map.get(repo)
            if not projects:
                continue
            matched_rows += 1
            for project in projects:
                counts[project] += 1

            if collect_commit_days:
                day_value = extract_day(row.get(day_field))
                commits: List[str] = []
                introduced = row.get("introduced_commits") or ""
                if introduced:
                    commits.extend([c.strip() for c in introduced.split(";") if c.strip()])
                if include_fixed_commits:
                    fixed = row.get("fixed_commits") or ""
                    if fixed:
                        commits.extend([c.strip() for c in fixed.split(";") if c.strip()])
                for commit in commits:
                    commit_days.setdefault(commit, day_value)

    print("Vulnerability summary for C/C++ projects:")
    print(f"  metadata projects: {len(metadata_entries)}")
    print(f"  matched vulnerability rows: {matched_rows}")
    print(f"  unique projects with vulnerabilities: {len(counts)}")
    top = counts.most_common(10)
    if top:
        print("  top projects:")
        for project, count in top:
            print(f"    {project}: {count}")

    if summary_out:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        with summary_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["project", "language", "main_repo", "vulnerability_count"])
            for project, count in counts.most_common():
                entry = metadata_entries.get(project, {})
                writer.writerow(
                    [
                        project,
                        entry.get("language", ""),
                        entry.get("main_repo", ""),
                        count,
                    ]
                )
        print(f"  summary CSV written to {summary_out}")

    return counts, commit_days, matched_rows


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch oss-fuzz project metadata.")
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help="Git URL to clone oss-fuzz from (default: %(default)s)",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Directory to clone oss-fuzz into (default: %(default)s)",
    )
    parser.add_argument(
        "--project",
        help="Project directory name under projects/, e.g. abseil-cpp. "
             "Omit to process every project.",
    )
    parser.add_argument(
        "--commit",
        help="Optional commit hash or ref to check out after cloning.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Include homepage and primary_contact fields in output if present.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="CSV file path to append results to (default: %(default)s)",
    )
    parser.add_argument(
        "--summarize-vulns",
        action="store_true",
        help="After metadata extraction, summarize vulnerability counts for C/C++ projects.",
    )
    parser.add_argument(
        "--vuln-csv",
        default=str(DEFAULT_VULN_CSV),
        help="Path to oss_fuzz_vulnerabilities.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--summary-out",
        default=str(DEFAULT_SUMMARY_OUT),
        help="Output CSV for vulnerability summary (default: %(default)s)",
    )
    parser.add_argument(
        "--count-unique-days",
        action="store_true",
        help="Count unique commits and distinct days (based on --day-field) among matched vulnerabilities.",
    )
    parser.add_argument(
        "--day-field",
        default="published",
        help="Vulnerability CSV column to interpret as date for unique-day counting (default: %(default)s).",
    )
    parser.add_argument(
        "--include-fixed-commits",
        action="store_true",
        help="Include fixed_commits in unique-commit/day counting (default: only introduced_commits).",
    )
    args = parser.parse_args(argv)

    repo_dir = Path(args.dest).expanduser().resolve()

    try:
        ensure_repo(repo_dir, args.repo_url, args.commit)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"git command failed: {exc}\n{exc.stderr}")
        return exc.returncode
    except Exception as exc:  # pylint: disable=broad-except
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1

    head = "unknown"
    try:
        head = run_git(["rev-parse", "HEAD"], repo_dir)
    except subprocess.CalledProcessError:
        pass  # Repo might be in a detached state but still usable.

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    metadata_entries: Dict[str, Dict[str, str]] = {}
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELD_ORDER)
        if write_header:
            writer.writeheader()
        projects = []
        if args.project:
            projects = [args.project]
        else:
            projects_dir = repo_dir / "projects"
            if not projects_dir.is_dir():
                sys.stderr.write(f"ERROR: {projects_dir} not found.\n")
                return 1
            projects = sorted(p.name for p in projects_dir.iterdir() if p.is_dir())

        print(f"Repository: {repo_dir}")
        print(f"HEAD: {head}")
        success_count = 0
        skipped_missing = 0
        skipped_language = 0
        total_projects = len(projects)

        required_fields = ["language", "main_repo"]
        optional_fields = ["homepage", "primary_contact"]

        for project in projects:
            yaml_path = repo_dir / "projects" / project / "project.yaml"
            if not yaml_path.is_file():
                sys.stderr.write(f"WARNING: {yaml_path} not found. Skipping.\n")
                skipped_missing += 1
                continue

            try:
                data = parse_project_yaml(yaml_path, required_fields, optional_fields)
            except KeyError as exc:
                sys.stderr.write(f"WARNING: {exc}. Skipping.\n")
                skipped_missing += 1
                continue

            language = (data.get("language") or "").strip().lower()
            if language not in {"c", "c++"}:
                skipped_language += 1
                continue

            print(f"Project: {project}")
            print_keys = ["language", "main_repo"]
            if args.print_all:
                print_keys.extend(["homepage", "primary_contact"])
            for key in print_keys:
                value = data.get(key, "")
                print(f"{key}: {value}")
            print()

            csv_row = {k: "" for k in CSV_FIELD_ORDER}
            csv_row["project"] = project
            for key in CSV_FIELD_ORDER[1:]:
                csv_row[key] = data.get(key, "")
            writer.writerow(csv_row)
            success_count += 1
            metadata_entries[project] = {
                "project": project,
                "language": csv_row["language"],
                "main_repo": csv_row["main_repo"],
            }

        print("Summary:")
        print(f"  total discovered: {total_projects}")
        print(f"  processed (C/C++): {success_count}")
        print(f"  skipped missing YAML/data: {skipped_missing}")
        print(f"  skipped by language filter: {skipped_language}")
        print(f"Metadata appended to {out_path}")
    if args.summarize_vulns:
        vuln_csv_path = Path(args.vuln_csv).expanduser().resolve()
        summary_out_path = Path(args.summary_out).expanduser().resolve() if args.summary_out else None
        counts, commit_days, _ = summarize_vulnerabilities(
            metadata_entries,
            vuln_csv_path,
            summary_out_path,
            collect_commit_days=args.count_unique_days,
            day_field=args.day_field,
            include_fixed_commits=args.include_fixed_commits,
        )
        if args.count_unique_days:
            unique_commits = len(commit_days)
            unique_days = len({day for day in commit_days.values() if day})
            print("Unique commit/day summary:")
            print(f"  commits counted: {unique_commits}")
            print(f"  distinct {args.day_field} days: {unique_days}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
