import argparse
import os
import sys
import subprocess
from typing import Dict, List


def _repo_root() -> str:
    # This file lives in <repo>/vuljit/cli.py; repo root is dirname of this file
    return os.path.dirname(os.path.abspath(__file__))


def _load_env(env_path: str) -> Dict[str, str]:
    env = {}
    if not os.path.isfile(env_path):
        return env
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip()
    except Exception:
        pass
    return env


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


def cmd_download_srcmap(args: argparse.Namespace) -> int:
    root = _repo_root()
    script = os.path.join(root, 'download_gcs', 'download_srcmap.py')

    csv = args.csv or os.environ.get('VULJIT_VUL_CSV') or os.path.join(root, 'data', 'oss_fuzz_vulns_0802.csv')
    start = args.start or os.environ.get('VULJIT_START_DATE', '20181012')
    end = args.end or os.environ.get('VULJIT_END_DATE', '20250802')
    out_dir = args.out or os.environ.get('VULJIT_SRCDOWN_DIR') or os.path.join(root, 'data', 'srcmap_json')
    workers = str(args.workers or int(os.environ.get('VULJIT_WORKERS', '8')))

    _ensure_dirs(out_dir)

    cmd = [sys.executable, script, csv, start, end, '-d', out_dir, '-w', workers]
    if args.csv_col is not None:
        cmd.extend(['--csv-column', str(args.csv_col)])
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd)


def cmd_download_coverage(args: argparse.Namespace) -> int:
    root = _repo_root()
    script = os.path.join(root, 'download_gcs', 'coverage_download_reports.py')

    csv = args.csv or os.environ.get('VULJIT_VUL_CSV') or os.path.join(root, 'data', 'oss_fuzz_vulns_0802.csv')
    start = args.start or os.environ.get('VULJIT_START_DATE', '20181012')
    end = args.end or os.environ.get('VULJIT_END_DATE', '20250802')
    out_dir = args.out or os.environ.get('VULJIT_COVERAGE_DIR') or os.path.join(root, 'data', 'coverage_gz')
    files = args.files or os.environ.get('VULJIT_COVERAGE_FILES', 'summary.json')
    files_list = [f.strip() for f in files.split(',') if f.strip()]
    workers = str(args.workers or int(os.environ.get('VULJIT_WORKERS', '8')))

    _ensure_dirs(out_dir)

    cmd = [sys.executable, script,
           '--csv', csv,
           '--start', start,
           '--end', end,
           '--out', out_dir,
           '--workers', workers]
    for f in files_list:
        cmd.extend(['--file', f])

    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd)


def cmd_extract_metrics(args: argparse.Namespace) -> int:
    root = _repo_root()
    # Use existing shell wrapper for compatibility
    sh = os.path.join(root, 'metrics_extraction', 'collect_metrics_from_github_project.sh')
    env = os.environ.copy()
    if args.metrics_dir:
        env['VULJIT_METRICS_DIR'] = args.metrics_dir
    else:
        env.setdefault('VULJIT_METRICS_DIR', os.path.join(root, 'data', 'metrics_output'))

    _ensure_dirs(env['VULJIT_METRICS_DIR'])

    cmd = ['bash', sh]
    print('Running:', ' '.join(cmd))
    return subprocess.call(cmd, env=env)


# ---------------- Metrics subcommands -----------------

def _clone_repos_from_csvs(csv_dir: str, clones_dir: str) -> None:
    # Parse all revisions_*.csv files and clone unique URLs
    import csv as _csv
    os.makedirs(clones_dir, exist_ok=True)
    urls: List[str] = []
    for name in os.listdir(csv_dir):
        if not name.startswith('revisions_') or not name.endswith('.csv'):
            continue
        path = os.path.join(csv_dir, name)
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    u = (row.get('url') or '').strip()
                    if u:
                        urls.append(u)
        except Exception:
            continue
    unique_urls = sorted(set(urls))
    for u in unique_urls:
        repo_name = u.split('/')[-1].replace('.git', '') if u.endswith('.git') else u.split('/')[-1]
        dest = os.path.join(clones_dir, repo_name)
        if os.path.isdir(dest):
            continue
        print('Cloning', u, '->', dest)
        subprocess.call(['git', 'clone', '--depth', '1', u, dest])


def cmd_metrics_code_text(args: argparse.Namespace) -> int:
    # Alias for existing shell wrapper
    ns = argparse.Namespace(metrics_dir=args.metrics_dir)
    return cmd_extract_metrics(ns)


def cmd_metrics_patch_coverage(args: argparse.Namespace) -> int:
    root = _repo_root()
    step1 = os.path.join(root, 'metrics_extraction', 'patch_coverage_extract', 'create_project_csvs_from_srcmap.py')
    step2 = os.path.join(root, 'metrics_extraction', 'patch_coverage_extract', 'revision_with_date.py')
    step3 = os.path.join(root, 'metrics_extraction', 'patch_coverage_extract', 'create_daily_diff.py')
    step4 = os.path.join(root, 'metrics_extraction', 'patch_coverage_extract', 'calculate_patch_coverage_per_project_test.py')

    src_root = args.src or os.environ.get('VULJIT_SRCDOWN_DIR') or os.path.join(root, 'data', 'srcmap_json')
    inter_dir = args.intermediate or os.environ.get('VULJIT_INTERMEDIATE_DIR') or os.path.join(root, 'data', 'intermediate')
    clones_dir = args.clones or os.environ.get('VULJIT_CLONED_REPOS_DIR') or os.path.join(inter_dir, 'cloned_repos')
    daily_diffs_dir = os.path.join(inter_dir, 'patch_coverage', 'daily_diffs')
    csv_results_dir = os.path.join(inter_dir, 'patch_coverage', 'csv_results')
    rev_with_date_dir = os.path.join(inter_dir, 'patch_coverage', 'revision_with_commit_date')
    out_root = args.out or os.environ.get('VULJIT_PATCH_COVERAGE_OUT') or os.path.join(os.environ.get('VULJIT_OUTPUTS_DIR', os.path.join(root, 'outputs')), 'metrics', 'patch_coverage')
    parsing_root = os.environ.get('VULJIT_PARSING_RESULTS_DIR') or os.path.join(inter_dir, 'patch_coverage', 'parsing_results')

    _ensure_dirs(csv_results_dir, rev_with_date_dir, daily_diffs_dir, out_root, parsing_root, clones_dir)

    # Step 1: build revisions_*.csv from srcmap JSONs
    cmd = [sys.executable, step1, '--root', src_root, '--out', csv_results_dir]
    print('Running:', ' '.join(cmd))
    if subprocess.call(cmd) != 0:
        return 1

    # Optionally clone repositories
    if args.clone_repos:
        _clone_repos_from_csvs(csv_results_dir, clones_dir)

    # Step 2: append commit dates
    cmd = [sys.executable, step2, '--src', csv_results_dir, '--repos', clones_dir, '--out', rev_with_date_dir]
    print('Running:', ' '.join(cmd))
    if subprocess.call(cmd) != 0:
        return 1

    # Step 3: create daily diffs and patches
    cmd = [sys.executable, step3, '--src', rev_with_date_dir, '--repos', clones_dir, '--out', daily_diffs_dir]
    print('Running:', ' '.join(cmd))
    if subprocess.call(cmd) != 0:
        return 1

    # Step 4: compute patch coverage per project
    projects = [d for d in os.listdir(daily_diffs_dir) if os.path.isdir(os.path.join(daily_diffs_dir, d))]
    for proj in projects:
        cmd = [sys.executable, step4, '-p', proj, '--diffs', daily_diffs_dir, '--out', out_root, '--parsing-out', parsing_root]
        print('Running:', ' '.join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print('Warning: project failed', proj)
    return 0


def cmd_metrics_coverage_aggregate(args: argparse.Namespace) -> int:
    root = _repo_root()
    script = os.path.join(root, 'metrics_extraction', 'coverage_analyze', 'process_coverage_project.py')
    coverage_root = args.src or os.environ.get('VULJIT_COVERAGE_DIR') or os.path.join(root, 'data', 'coverage_gz')
    out_root = args.out or os.path.join(os.environ.get('VULJIT_OUTPUTS_DIR', os.path.join(root, 'outputs')), 'metrics', 'coverage_aggregate')
    _ensure_dirs(out_root)
    # each project dir under coverage_root
    projects = [d for d in os.listdir(coverage_root) if os.path.isdir(os.path.join(coverage_root, d))]
    for proj in projects:
        proj_dir = os.path.join(coverage_root, proj)
        cmd = [sys.executable, script, proj_dir, '--out', out_root]
        print('Running:', ' '.join(cmd))
        subprocess.call(cmd)
    return 0


def cmd_metrics_aggregate_daily(args: argparse.Namespace) -> int:
    root = _repo_root()
    script = os.path.join(root, 'prediction', 'merge_coverage_metrics_test12.py')

    metrics = args.metrics or os.environ.get('VULJIT_METRICS_DIR') or os.path.join(root, 'data', 'metrics_output')
    coverage = args.coverage or os.environ.get('VULJIT_COVERAGE_AGG_DIR') or os.path.join(root, 'outputs', 'metrics', 'coverage_aggregate')
    patch = args.patch or os.environ.get('VULJIT_PATCH_COV_DIR') or os.path.join(root, 'outputs', 'metrics', 'patch_coverage')
    out = args.out or os.environ.get('VULJIT_BASE_DATA_DIR') or os.path.join(root, 'data')

    def _run_pair(project_id: str, directory_name: str) -> int:
        cmd = [sys.executable, script, project_id, directory_name,
               '--metrics', metrics, '--coverage', coverage, '--patch-coverage', patch, '--out', out]
        print('Running:', ' '.join(cmd))
        return subprocess.call(cmd)

    if args.project and args.dir:
        return _run_pair(args.project, args.dir)

    # Mapping CSV: two columns (project_id,directory_name). Skip header if present.
    mapping = args.mapping or os.environ.get('VULJIT_PROJECT_MAPPING') or os.path.join(root, 'mapping', 'project_mapping.csv')
    if not os.path.isfile(mapping):
        print('Mapping CSV not found:', mapping)
        return 1

    rc_all = 0
    with open(mapping, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # skip obvious header
            if idx == 0 and ('project' in line.lower() and 'dir' in line.lower()):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            pid, dname = parts[0], parts[1]
            rc = _run_pair(pid, dname)
            rc_all = rc_all or rc
    return rc_all


def main(argv=None) -> int:
    # Load .env if present
    root = _repo_root()
    dotenv_path = os.path.join(root, '.env')
    env_overrides = _load_env(dotenv_path)
    os.environ.update({k: v for k, v in env_overrides.items() if k not in os.environ})

    parser = argparse.ArgumentParser(prog='vuljit', description='VULJIT replication CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('download-srcmap', help='Download srcmap JSONs from GCS')
    p1.add_argument('--csv', help='CSV path containing package names')
    p1.add_argument('--start', help='Start date YYYYMMDD')
    p1.add_argument('--end', help='End date YYYYMMDD')
    p1.add_argument('--out', help='Output directory')
    p1.add_argument('--workers', type=int, help='Parallel workers')
    p1.add_argument('--csv-column', type=int, dest='csv_col', help='0-based column index of package names')
    p1.set_defaults(func=cmd_download_srcmap)

    p2 = sub.add_parser('download-coverage', help='Download coverage reports (gz) from GCS')
    p2.add_argument('--csv', help='CSV path containing package names')
    p2.add_argument('--start', help='Start date YYYYMMDD')
    p2.add_argument('--end', help='End date YYYYMMDD')
    p2.add_argument('--out', help='Output directory')
    p2.add_argument('--workers', type=int, help='Parallel workers')
    p2.add_argument('--files', help='Comma separated file list (default from env)')
    p2.add_argument('--file', action='append', help='Repeatable file list item (overrides --files)')
    p2.set_defaults(func=cmd_download_coverage)

    p3 = sub.add_parser('extract-metrics', help='Run code+text metrics extraction (compat alias)')
    p3.add_argument('--metrics-dir', help='Metrics root directory')
    p3.set_defaults(func=cmd_extract_metrics)

    # metrics group
    m = sub.add_parser('metrics', help='Metrics related commands')
    msub = m.add_subparsers(dest='metrics_cmd', required=True)

    m1 = msub.add_parser('code-text', help='Merge code and text metrics')
    m1.add_argument('--metrics-dir', help='Metrics root directory')
    m1.set_defaults(func=cmd_metrics_code_text)

    m2 = msub.add_parser('patch-coverage', help='Compute patch coverage per project')
    m2.add_argument('--src', help='Srcmap root directory')
    m2.add_argument('--intermediate', help='Intermediate root directory')
    m2.add_argument('--clones', help='Cloned repos directory')
    m2.add_argument('--out', help='Output root directory')
    m2.add_argument('--clone-repos', action='store_true', help='Clone repositories referenced in revisions CSVs')
    m2.set_defaults(func=cmd_metrics_patch_coverage)

    m3 = msub.add_parser('coverage-aggregate', help='Aggregate coverage JSON summaries to CSVs')
    m3.add_argument('--src', help='Coverage root directory')
    m3.add_argument('--out', help='Output root directory')
    m3.set_defaults(func=cmd_metrics_coverage_aggregate)

    m4 = msub.add_parser('aggregate-daily', help='Aggregate commit-level to daily-level dataset')
    m4.add_argument('--project', help='Project ID (e.g., apache-httpd)')
    m4.add_argument('--dir', help='Directory name (e.g., httpd)')
    m4.add_argument('--mapping', help='CSV mapping file with project_id,dir')
    m4.add_argument('--metrics', help='Metrics base path')
    m4.add_argument('--coverage', help='Coverage totals base path')
    m4.add_argument('--patch', help='Patch coverage base path')
    m4.add_argument('--out', help='Output datasets root (BASE_DATA_DIRECTORY)')
    m4.set_defaults(func=cmd_metrics_aggregate_daily)

    # prediction group
    pr = sub.add_parser('prediction', help='Prediction related commands')
    prsub = pr.add_subparsers(dest='pred_cmd', required=True)

    def _pred_cmd(script_rel: str, extra_args: List[str] | None = None) -> int:
        root = _repo_root()
        script = os.path.join(root, 'prediction', script_rel)
        cmd = [sys.executable, script] + (extra_args or [])
        print('Running:', ' '.join(cmd))
        return subprocess.call(cmd)

    def _cmd_prediction_train(args: argparse.Namespace) -> int:
        extra = []
        if args.project:
            extra = ['-p', args.project]
        return _pred_cmd('main_per_project.py', extra)

    def _cmd_prediction_rq3(args: argparse.Namespace) -> int:
        extra = []
        if args.project:
            extra = ['-p', args.project]
        return _pred_cmd('rq3_prepare.py', extra)

    pt = prsub.add_parser('train', help='Train and evaluate within-project models')
    pt.add_argument('-p', '--project', help='Single project name')
    pt.set_defaults(func=_cmd_prediction_train)

    prq = prsub.add_parser('rq3', help='Re-train on first half and score second half')
    prq.add_argument('-p', '--project', help='Single project name')
    prq.set_defaults(func=_cmd_prediction_rq3)

    # mapping group (lightweight helpers)
    def _cmd_mapping_count_candidates(args: argparse.Namespace) -> int:
        root = _repo_root()
        script = os.path.join(root, 'mapping', 'count_c_cpp_projects.py')
        cmd = [sys.executable, script]
        if args.csv:
            cmd.extend(['--csv', args.csv])
        if args.repos:
            cmd.extend(['--repos', args.repos])
        if args.out_csv:
            cmd.extend(['--out-csv', args.out_csv])
        print('Running:', ' '.join(cmd))
        return subprocess.call(cmd)

    mp = sub.add_parser('mapping', help='Mapping utilities')
    mpsub = mp.add_subparsers(dest='map_cmd', required=True)
    mpc = mpsub.add_parser('count-candidates', help='Count candidate projects from mapping + cloned repos')
    mpc.add_argument('--csv', help='Mapping CSV (project_id,directory_name)')
    mpc.add_argument('--repos', help='Top-level cloned repos directory')
    mpc.add_argument('--out-csv', help='Optional: write per-directory match CSV')
    mpc.set_defaults(func=_cmd_mapping_count_candidates)

    args = parser.parse_args(argv)

    # Normalize files option
    if getattr(args, 'file', None):
        args.files = ','.join(args.file)

    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
