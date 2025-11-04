import os
import glob
import json
import argparse
from typing import Dict, Any, Set, List, Tuple

from pathlib import Path
from datetime import datetime, date
import re
import pandas as pd

def _coerce_is_vcc(s: 'pd.Series') -> 'pd.Series':
    if s.dtype == bool:
        return s.astype(int)
    # map common string forms to 0/1, then force numeric
    mapping = {
        'True': 1, 'true': 1, 'TRUE': 1, True: 1,
        'False': 0, 'false': 0, 'FALSE': 0, False: 0,
    }
    s2 = s.replace(mapping)
    return pd.to_numeric(s2, errors='coerce')


# (pandas 前提のため、CSVフォールバック処理は不要)


def compute_final_stats(base_data_dir: str,
                       start_date: date | None = None,
                       end_date: date | None = None) -> Dict[str, Any]:
    pattern = os.path.join(base_data_dir, '*', '*_daily_aggregated_metrics.csv')
    csv_files = sorted(glob.glob(pattern))

    frames = []
    for path in csv_files:
        project = os.path.basename(os.path.dirname(path))
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in ('merge_date', 'is_vcc'))
        except Exception:
            continue

        if 'merge_date' not in df.columns or 'is_vcc' not in df.columns:
            continue

        # normalize types
        df['merge_date'] = pd.to_datetime(df['merge_date'], errors='coerce', utc=True).dt.date
        df['is_vcc'] = _coerce_is_vcc(df['is_vcc'])
        df = df.dropna(subset=['merge_date', 'is_vcc'])
        # optional period filter
        if start_date is not None:
            df = df[df['merge_date'] >= start_date]
        if end_date is not None:
            df = df[df['merge_date'] <= end_date]

        if df.empty:
            continue

        df['project'] = project
        # remove accidental duplicates (same project+date)
        df = df.drop_duplicates(subset=['project', 'merge_date'])
        frames.append(df[['project', 'merge_date', 'is_vcc']])

    if not frames:
        return {
            'number_of_projects': 0,
            'total_days': 0,
            'positive_days': 0,
            'negative_days': 0,
            'positive_ratio': 0.0,
            'per_project': [],
        }

    all_df = pd.concat(frames, axis=0, ignore_index=True)

    # per-project stats
    grp = all_df.groupby('project')
    per_project_stats = (
        grp['is_vcc']
        .agg(total_days='count', positive_days=lambda s: int((s == 1).sum()))
        .reset_index()
    )
    per_project_stats['negative_days'] = per_project_stats['total_days'] - per_project_stats['positive_days']
    per_project_stats['positive_ratio'] = per_project_stats.apply(
        lambda r: (r['positive_days'] / r['total_days']) if r['total_days'] else 0.0, axis=1
    )

    number_of_projects = int((per_project_stats['total_days'] > 0).sum())
    total_days = int(per_project_stats['total_days'].sum())
    positive_days = int(per_project_stats['positive_days'].sum())
    negative_days = int(per_project_stats['negative_days'].sum())
    positive_ratio = float((positive_days / total_days) if total_days else 0.0)

    return {
        'number_of_projects': number_of_projects,
        'total_days': total_days,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'positive_ratio': positive_ratio,
        'per_project': per_project_stats.to_dict(orient='records'),
    }


def compute_final_stats_for_projects(base_data_dir: str,
                                     allowed_projects: Set[str],
                                     start_date: date | None = None,
                                     end_date: date | None = None) -> Dict[str, Any]:
    """allowed_projects に含まれるプロジェクトだけを対象に最終統計を計算"""
    pattern = os.path.join(base_data_dir, '*', '*_daily_aggregated_metrics.csv')
    csv_files = sorted(glob.glob(pattern))

    frames = []
    for path in csv_files:
        project = os.path.basename(os.path.dirname(path))
        if project not in allowed_projects:
            continue
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in ('merge_date', 'is_vcc'))
        except Exception:
            continue
        if 'merge_date' not in df.columns or 'is_vcc' not in df.columns:
            continue
        df['merge_date'] = pd.to_datetime(df['merge_date'], errors='coerce', utc=True).dt.date
        df['is_vcc'] = _coerce_is_vcc(df['is_vcc'])
        df = df.dropna(subset=['merge_date', 'is_vcc'])
        if start_date is not None:
            df = df[df['merge_date'] >= start_date]
        if end_date is not None:
            df = df[df['merge_date'] <= end_date]
        if df.empty:
            continue
        df['project'] = project
        df = df.drop_duplicates(subset=['project', 'merge_date'])
        frames.append(df[['project', 'merge_date', 'is_vcc']])

    if not frames:
        return {
            'number_of_projects': 0,
            'total_days': 0,
            'positive_days': 0,
            'negative_days': 0,
            'positive_ratio': 0.0,
            'per_project': [],
        }

    all_df = pd.concat(frames, axis=0, ignore_index=True)
    grp = all_df.groupby('project')
    per_project_stats = (
        grp['is_vcc']
        .agg(total_days='count', positive_days=lambda s: int((s == 1).sum()))
        .reset_index()
    )
    per_project_stats['negative_days'] = per_project_stats['total_days'] - per_project_stats['positive_days']
    per_project_stats['positive_ratio'] = per_project_stats.apply(
        lambda r: (r['positive_days'] / r['total_days']) if r['total_days'] else 0.0, axis=1
    )

    number_of_projects = int((per_project_stats['total_days'] > 0).sum())
    total_days = int(per_project_stats['total_days'].sum())
    positive_days = int(per_project_stats['positive_days'].sum())
    negative_days = int(per_project_stats['negative_days'].sum())
    positive_ratio = float((positive_days / total_days) if total_days else 0.0)

    return {
        'number_of_projects': number_of_projects,
        'total_days': total_days,
        'positive_days': positive_days,
        'negative_days': negative_days,
        'positive_ratio': positive_ratio,
        'per_project': per_project_stats.to_dict(orient='records'),
    }


def _find_projects(base_dir: Path) -> Set[str]:
    if not base_dir.exists() or not base_dir.is_dir():
        return set()
    return {p.name for p in base_dir.iterdir() if p.is_dir()}


def _detect_experiments(base_dir: Path, project: str) -> Set[int]:
    exp_ids: Set[int] = set()
    proj_dir = base_dir / project
    if not proj_dir.exists():
        return exp_ids
    for f in proj_dir.glob('exp*_metrics.json'):
        m = re.search(r'exp(\d+)_metrics\.json$', f.name)
        if m:
            exp_ids.add(int(m.group(1)))
    return exp_ids


def _load_metrics_json(base_dir: Path, project: str, exp_id: int) -> Dict[str, Any] | None:
    p = base_dir / project / f"exp{exp_id}_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None


# ------------------------------
# OSV / Coverage dataset helpers
# ------------------------------

def _load_c_cpp_projects(list_path: str | Path) -> Set[str]:
    p = Path(list_path)
    if not p.exists():
        return set()
    lines: List[str] = []
    with p.open('r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            lines.append(ln)
    return set(lines)


def _resolve_default_coverage_root() -> str | None:
    """Pick a reasonable default coverage root if present in this workspace."""
    env = os.environ.get('VULJIT_COVERAGE_DIR')
    if env and os.path.isdir(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    candidates = [
        os.path.join(repo_root, 'ossfuzz_downloaded_coverage_2025802_gz'),
        os.path.join(repo_root, 'ossfuzz_downloaded_coverage_gz'),
        os.path.join(repo_root, 'vuljit', 'data', 'coverage_gz'),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None


def _scan_coverage_root(coverage_root: str,
                        start: date | None = None,
                        end: date | None = None) -> Tuple[Dict[str, date], int, int, date | None, date | None, Set[str]]:
    """Walk the coverage root to determine:
    - earliest coverage date per project (only if a summary.json.gz exists for that date)
    - total number of daily coverage reports (count of existing summary.json.gz files)
    - number of projects with at least one report
    - min/max date across all projects
    - set of projects discovered
    Directory layout: coverage_root/<project>/<YYYYMMDD>/linux/summary.json.gz
    """
    earliest: Dict[str, date] = {}
    total_reports = 0
    projects: Set[str] = set()
    overall_min: date | None = None
    overall_max: date | None = None

    for proj in os.listdir(coverage_root):
        proj_dir = os.path.join(coverage_root, proj)
        if not os.path.isdir(proj_dir):
            continue
        projects.add(proj)

        first_date: date | None = None
        for d in os.listdir(proj_dir):
            if not re.match(r'^\d{8}$', d):
                continue
            day_dir = os.path.join(proj_dir, d)
            if not os.path.isdir(day_dir):
                continue
            summ = os.path.join(day_dir, 'linux', 'summary.json.gz')
            if not os.path.isfile(summ):
                continue
            dt = datetime.strptime(d, '%Y%m%d').date()
            if start is not None and dt < start:
                continue
            if end is not None and dt > end:
                continue
            total_reports += 1
            if first_date is None or dt < first_date:
                first_date = dt
            if overall_min is None or dt < overall_min:
                overall_min = dt
            if overall_max is None or dt > overall_max:
                overall_max = dt

        if first_date is not None:
            earliest[proj] = first_date

    projects_with_reports: Set[str] = set(earliest.keys())
    return earliest, total_reports, len(projects_with_reports), overall_min, overall_max, projects_with_reports


def _load_osv_csv(vulns_csv: str | Path) -> pd.DataFrame | None:
    p = Path(vulns_csv)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    if 'published' in df.columns:
        df['published'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
    if 'modified' in df.columns:
        df['modified'] = pd.to_datetime(df['modified'], errors='coerce', utc=True)
    for col in ('package_name', 'ecosystem', 'repo'):
        if col not in df.columns:
            df[col] = ''
    return df


def compute_osv_and_coverage_stats(vulns_csv: str | Path,
                                   c_cpp_list: str | Path,
                                   coverage_root: str | None,
                                   min_issues_per_repo: int = 10,
                                   coverage_start: date | None = None,
                                   coverage_end: date | None = None) -> Dict[str, Any]:
    """Compute the OSV/coverage statistics required by the narrative (pandas-based)."""
    projects = _load_c_cpp_projects(c_cpp_list)
    df = _load_osv_csv(vulns_csv)
    if df is None:
        df = pd.DataFrame()

    total_osv_issues = int(len(df))

    if not df.empty:
        mask_ecosys = (df['ecosystem'].astype(str).str.lower() == 'oss-fuzz')
        mask_proj = df['package_name'].astype(str).isin(projects) if projects else True
        df_cc = df[mask_ecosys & mask_proj].copy()
    else:
        df_cc = pd.DataFrame()

    c_cpp_issues = int(len(df_cc))
    c_cpp_percent = float((c_cpp_issues / total_osv_issues * 100.0) if total_osv_issues else 0.0)
    c_cpp_repos = int(df_cc['repo'].nunique()) if not df_cc.empty else 0

    if not df_cc.empty:
        gh_mask = df_cc['repo'].astype(str).str.contains('github.com/', na=False)
        df_gh = df_cc[gh_mask].copy()
    else:
        df_gh = pd.DataFrame()
    gh_issues = int(len(df_gh))
    gh_repos = int(df_gh['repo'].nunique()) if not df_gh.empty else 0

    if not df_gh.empty:
        # Repositories with at least N unique vulnerability-injecting commits (introduced_commits)
        # Normalize introduced_commits to string and split if multiple are present
        temp = df_gh[['repo', 'introduced_commits']].copy()
        temp['introduced_commits'] = temp['introduced_commits'].astype(str).str.strip()
        # build unique (repo, commit) pairs
        pairs = []
        for _, row in temp.iterrows():
            repo = str(row['repo'])
            ic = row['introduced_commits']
            if not ic or ic.lower() == 'nan':
                continue
            # handle potential separators (robust in case of future multi-commit rows)
            parts = [p for p in re.split(r'[\s,;]+', ic) if p]
            for c in parts:
                pairs.append((repo, c))
        if pairs:
            vcc_df = pd.DataFrame(pairs, columns=['repo', 'commit'])
            vcc_counts = vcc_df.drop_duplicates().groupby('repo').size()
            keep_repos = set(vcc_counts[vcc_counts >= min_issues_per_repo].index)
        else:
            keep_repos = set()
        df_gh_min = df_gh[df_gh['repo'].isin(keep_repos)].copy()
    else:
        df_gh_min = pd.DataFrame()
    gh_min10_issues = int(len(df_gh_min))
    gh_min10_repos = int(df_gh_min['repo'].nunique()) if not df_gh_min.empty else 0

    # Coverage scanning (common to both paths)
    cov_total_reports = 0
    cov_projects = 0
    cov_min_date: str | None = None
    cov_max_date: str | None = None
    earliest_by_proj: Dict[str, date] = {}
    covered_projects: Set[str] = set()
    if coverage_root and os.path.isdir(coverage_root):
        earliest_by_proj, cov_total_reports, cov_projects, cov_min, cov_max, covered_projects = _scan_coverage_root(
            coverage_root, start=coverage_start, end=coverage_end
        )
        cov_min_date = cov_min.isoformat() if cov_min else None
        cov_max_date = cov_max.isoformat() if cov_max else None

    # Align OSV by first coverage date
    if coverage_root and earliest_by_proj:
        if 'published' in df_gh.columns:
            df_gh['published_date'] = pd.to_datetime(df_gh['published'], errors='coerce', utc=True).dt.date
        else:
            df_gh['published_date'] = pd.NaT
        has_cov_mask = df_gh['package_name'].astype(str).isin(covered_projects)
        def ok_row(r) -> bool:
            proj = str(r['package_name'])
            d = r['published_date']
            # Guard NaT/NaN
            if pd.isna(d):
                return False
            first = earliest_by_proj.get(proj)
            if first is None:
                return False
            if d < first:
                return False
            if coverage_end is not None and d > coverage_end:
                return False
            if coverage_start is not None and d < coverage_start:
                return False
            return True
        kept = df_gh[has_cov_mask].copy()
        if not kept.empty:
            kept = kept[kept.apply(ok_row, axis=1)]
        osv_after_cov_issues = int(len(kept))
        osv_after_cov_repos = int(kept['repo'].nunique()) if not kept.empty else 0
    else:
        osv_after_cov_issues = 0
        osv_after_cov_repos = 0

    return {
        'total_osv_issues': int(total_osv_issues),
        'c_cpp_issues': int(c_cpp_issues),
        'c_cpp_percent': float(c_cpp_percent),
        'c_cpp_repos': int(c_cpp_repos),
        'gh_issues': int(gh_issues),
        'gh_repos': int(gh_repos),
        'gh_min10_issues': int(gh_min10_issues),
        'gh_min10_repos': int(gh_min10_repos),
        'cov_total_reports': int(cov_total_reports),
        'cov_projects': int(cov_projects),
        'cov_min_date': cov_min_date,
        'cov_max_date': cov_max_date,
        'osv_after_cov_issues': int(osv_after_cov_issues),
        'osv_after_cov_repos': int(osv_after_cov_repos),
    }


def main():
    try:
        # settings は任意。存在すれば既定のBASE_DATA_DIRECTORYとRESULTSを取得
        import settings  # type: ignore
        default_base = getattr(settings, 'BASE_DATA_DIRECTORY', None)
        default_results = getattr(settings, 'RESULTS_BASE_DIRECTORY', None)
    except Exception:
        settings = None  
        default_base = None
        default_results = None

    parser = argparse.ArgumentParser(description='Collect final daily-level dataset statistics and fill OSV/coverage narrative counts.')
    parser.add_argument('--base', default=default_base, help='Base data directory containing per-project daily CSVs (auto-detects common datasets if omitted).')
    parser.add_argument('--out', default=default_results, help='Results base directory to write stats under.')
    parser.add_argument('--results', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets', 'model_outputs'), help='Base results directory containing per-model subfolders.')
    parser.add_argument('--models', nargs='*', default=['xgboost', 'random_forest', 'random'], help='Model folder names under --results to consider for common projects.')
    parser.add_argument('--use-common-projects', action='store_true', help='If set, restrict to projects common to all specified model folders (directory presence).')
    parser.add_argument('--strict-common', action='store_true', help='If set, further restrict to projects that have metrics for ALL specified experiments and ALL models (like analyze_comparison).')
    parser.add_argument('--exp', nargs='*', type=int, help='Experiment ids to consider for strict common (e.g., 1 2 3). If omitted, auto-detect across models/projects.')
    parser.add_argument('--random-exp-id', type=int, default=0, help='Experiment id used by Random baseline (maps all exps to this id).')
    # OSV / Coverage params
    here2 = os.path.dirname(os.path.abspath(__file__))
    repo_root2 = os.path.abspath(os.path.join(here2, '..', '..'))
    default_vulns_csv = os.environ.get('VULJIT_VUL_CSV') or os.path.join(repo_root2, 'datasets', 'raw', 'rq3_dataset', 'oss_fuzz_vulns_2025802.csv')
    parser.add_argument('--vulns-csv', default=default_vulns_csv, help='Path to OSV vulnerabilities CSV (from oss-fuzz-vulns).')
    default_c_cpp_list = os.path.join(repo_root2, 'datasets', 'reference_mappings', 'c_cpp_projects.txt')
    parser.add_argument('--c-cpp-list', default=default_c_cpp_list, help='Path to list of C/C++ OSS-Fuzz project names.')
    parser.add_argument('--coverage-root', default=_resolve_default_coverage_root(), help='Coverage reports root (downloaded OSS-Fuzz daily coverage).')
    parser.add_argument('--min-issues-per-repo', type=int, default=10, help='Threshold for min vulnerability reports per repo.')
    # Period filters
    parser.add_argument('--coverage-start', type=str, default='2016-01-01', help='Coverage period start YYYY-MM-DD (inclusive).')
    parser.add_argument('--coverage-end', type=str, default='2025-06-01', help='Coverage period end YYYY-MM-DD (inclusive).')
    parser.add_argument('--dataset-start', type=str, default='2016-01-01', help='Dataset period start YYYY-MM-DD (inclusive).')
    parser.add_argument('--dataset-end', type=str, default='2025-06-01', help='Dataset period end YYYY-MM-DD (inclusive).')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, '..', '..'))
    workspace_root = os.path.abspath(os.path.join(repo_root, '..'))

    env_base = os.environ.get('VULJIT_BASE_DATA_DIR') or os.environ.get('VULJIT_DATASET_DIR')
    base_candidates: List[str] = []
    if args.base:
        base_candidates.append(args.base)
    if env_base:
        base_candidates.append(env_base)
    if default_base and default_base not in base_candidates:
        base_candidates.append(default_base)
    base_candidates.extend([
        os.path.join(workspace_root, 'daily_commit_summary_past_vul_0802'),
        os.path.join(workspace_root, 'daily_commit_summary_past_vul'),
        os.path.join(repo_root, 'datasets'),
    ])
    resolved_base = next((c for c in base_candidates if c and os.path.isdir(c)), os.path.join(repo_root, 'datasets'))
    args.base = resolved_base

    # Parse period args to date
    def _parse_date_arg(s: str | None) -> date | None:
        if not s:
            return None
        try:
            return datetime.fromisoformat(s).date()
        except Exception:
            return None
    cov_start = _parse_date_arg(args.coverage_start)
    cov_end = _parse_date_arg(args.coverage_end)
    ds_start = _parse_date_arg(args.dataset_start)
    ds_end = _parse_date_arg(args.dataset_end)

    # 共通プロジェクトに制限する場合
    if args.use_common_projects:
        # 1) モデルフォルダの存在プロジェクトの積集合（初期）
        pretty = {'xgboost': 'xgboost', 'random_forest': 'random_forest', 'random': 'random'}
        model_dirs: Dict[str, Path] = {m: Path(os.path.join(args.results, pretty.get(m.lower(), m))) for m in args.models}
        projects_sets = [_find_projects(p) for p in model_dirs.values()]
        projects_sets = [s for s in projects_sets if s]
        if not projects_sets:
            print('Error: No valid model result directories or no projects found.')
            return
        common_projects = set.intersection(*projects_sets)
        if not common_projects:
            print('Error: No common projects across specified models.')
            return

        # 2) strict-common: 全実験×全モデルで metrics が存在するプロジェクトに厳格化
        if args.strict_common:
            # 実験集合を検出（オプションで絞り込み）
            if args.exp:
                all_exps: Set[int] = set(args.exp)
            else:
                all_exps: Set[int] = set()
                for m, base in model_dirs.items():
                    for prj in common_projects:
                        all_exps |= _detect_experiments(base, prj)
                if not all_exps:
                    print('Error: No experiments detected under common projects.')
                    return

            # 各実験で、全モデルにmetricsがあるプロジェクト集合を求める
            common_per_exp: List[Set[str]] = []
            for exp in sorted(all_exps):
                ok_sets: List[Set[str]] = []
                for m, base in model_dirs.items():
                    eff_exp = args.random_exp_id if m.lower() == 'random' else exp
                    ok_prj = {prj for prj in common_projects if _load_metrics_json(base, prj, eff_exp) is not None}
                    ok_sets.append(ok_prj)
                if ok_sets:
                    common_exp = set.intersection(*ok_sets)
                    if common_exp:
                        common_per_exp.append(common_exp)
            if not common_per_exp:
                print('Error: No projects have metrics across all models for the specified experiments.')
                return
            strict_common_projects = set.intersection(*common_per_exp)
            if not strict_common_projects:
                print('Error: Intersection across all experiments is empty.')
                return
            target_projects = strict_common_projects
        else:
            target_projects = common_projects

        stats = compute_final_stats_for_projects(args.base, target_projects, start_date=ds_start, end_date=ds_end)
    else:
        stats = compute_final_stats(args.base, start_date=ds_start, end_date=ds_end)

    # OSV and coverage stats for narrative
    osv_cov_stats = compute_osv_and_coverage_stats(
        vulns_csv=args.vulns_csv,
        c_cpp_list=args.c_cpp_list,
        coverage_root=args.coverage_root,
        min_issues_per_repo=args.min_issues_per_repo,
        coverage_start=cov_start,
        coverage_end=cov_end,
    )

    # pretty print (narrative + Table 2)
    print('(1) Labels via OSV. To train models, we label each day as having at least one vulnerability when any commit made on that day is associated with a vulnerability in OSV (via introduced/fixed links where available).')
    print('(2) OSV fields. The OSV entries include project name (package_name), the date the vulnerability was reported (published), severity, summary/details, introduced and fixed commits, and affected versions. For OSS-Fuzz discoveries, reports contain the issue tracker link (report_url).')
    print(f"(3) C/C++ focus. From {osv_cov_stats['total_osv_issues']} OSS-Fuzz OSV issues, we obtained {osv_cov_stats['c_cpp_issues']} issues ({osv_cov_stats['c_cpp_percent']:.2f}%) from {osv_cov_stats['c_cpp_repos']} repositories targeting C/C++ projects.")
    print(f"(4) Public GitHub repos. Excluding repositories not publicly available on GitHub yields {osv_cov_stats['gh_issues']} issues from {osv_cov_stats['gh_repos']} repositories. Further restricting to repositories with at least {args.min_issues_per_repo} unique vulnerability-injecting commits (introduced_commits) results in {osv_cov_stats['gh_min10_issues']} issues from {osv_cov_stats['gh_min10_repos']} repositories.")
    if args.coverage_root:
        rng = ''
        if osv_cov_stats['cov_min_date'] and osv_cov_stats['cov_max_date']:
            rng = f" from {osv_cov_stats['cov_min_date']} to {osv_cov_stats['cov_max_date']}"
        print(f"(5) Daily coverage. We collected a total of {osv_cov_stats['cov_total_reports']} daily coverage reports from {osv_cov_stats['cov_projects']} projects{rng}. These files aggregate line, function, region, branch, and instruction coverage per day.")
        print(f"(6) Coverage alignment. We excluded vulnerability reports created before each project's first coverage report, yielding {osv_cov_stats['osv_after_cov_issues']} vulnerability reports across {osv_cov_stats['osv_after_cov_repos']} repositories.")
    print(f"(5) Final Dataset. After applying this entire pipeline, the final dataset used for the Within-Project evaluation consists of {stats['total_days']} daily instances from {stats['number_of_projects']} projects. The final statistics are shown in Table 2.")
    print('Table 2. Statistics of the Final Daily-level Dataset')
    print(f'Metric,Value')
    print(f'Number of Projects,{stats["number_of_projects"]}')
    print(f'Total Days (Instances),{stats["total_days"]}')
    print(f'# Positive Days (with VICs),{stats["positive_days"]}')
    print(f'# Negative Days (without VICs),{stats["negative_days"]}')
    print(f'Ratio of Positive Days,{stats["positive_ratio"]:.6f}')

    # save outputs
    out_base = args.out
    if not out_base:
        # default to repo datasets/model_outputs
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, '..', '..'))
        out_base = os.path.join(repo_root, 'datasets', 'model_outputs')

    out_dir = os.path.join(out_base, 'final_dataset_stats' + ('_common_strict' if args.use_common_projects and args.strict_common else ('_common' if args.use_common_projects else '')))
    os.makedirs(out_dir, exist_ok=True)

    # JSON summary
    summary = {k: v for k, v in stats.items() if k != 'per_project'}
    summary.update({
        'osv_total_issues': osv_cov_stats.get('total_osv_issues', 0),
        'osv_c_cpp_issues': osv_cov_stats.get('c_cpp_issues', 0),
        'osv_c_cpp_percent': osv_cov_stats.get('c_cpp_percent', 0.0),
        'osv_c_cpp_repos': osv_cov_stats.get('c_cpp_repos', 0),
        'osv_github_issues': osv_cov_stats.get('gh_issues', 0),
        'osv_github_repos': osv_cov_stats.get('gh_repos', 0),
        'osv_min10_issues': osv_cov_stats.get('gh_min10_issues', 0),
        'osv_min10_repos': osv_cov_stats.get('gh_min10_repos', 0),
        'coverage_total_reports': osv_cov_stats.get('cov_total_reports', 0),
        'coverage_projects': osv_cov_stats.get('cov_projects', 0),
        'coverage_min_date': osv_cov_stats.get('cov_min_date'),
        'coverage_max_date': osv_cov_stats.get('cov_max_date'),
        'osv_after_coverage_issues': osv_cov_stats.get('osv_after_cov_issues', 0),
        'osv_after_coverage_repos': osv_cov_stats.get('osv_after_cov_repos', 0),
    })
    with open(os.path.join(out_dir, 'final_dataset_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # CSV summary (single row)
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'final_dataset_stats.csv'), index=False)
    # Per-project CSV
    pd.DataFrame(stats['per_project']).to_csv(os.path.join(out_dir, 'per_project_stats.csv'), index=False)


if __name__ == '__main__':
    main()
