# VULJIT — Replication Package (Work-in-Progress)

This folder provides a self-contained replication package for the VULJIT study.
It offers a CLI, Docker image, and Makefile to standardize data download,
metrics extraction, and prediction steps.

Status: CLI/Docker/Makefile with downloads, metrics (code+text, patch coverage, coverage aggregate) and prediction wrappers. Further polishing is incremental.

## Requirements
- Python 3.12
- Or Docker (recommended for reproducibility)

## Quick Start (Local)
1) Copy `.env.example` to `.env` and adjust values as needed.
2) Install dependencies:
   - `make setup`
3) Run basic commands:
   - `python -m vuljit download-srcmap`  # download srcmap JSONs
   - `python -m vuljit download-coverage`  # download coverage gz files
   - `python -m vuljit metrics code-text`  # merge code/text metrics
   - `python -m vuljit metrics patch-coverage --clone-repos`  # compute patch coverage
   - `python -m vuljit metrics coverage-aggregate`  # aggregate coverage summaries
   - `python -m vuljit metrics aggregate-daily --project <pkg> --dir <repo>`  # commit→daily dataset
   - `python -m vuljit prediction train`  # within-project training/evaluation
   - `python -m vuljit prediction rq3`  # RQ3: score second-half timeline

By default, data is written under `vuljit/data` and results under `vuljit/outputs`.
Override via `.env` or environment variables.

## Quick Start (Docker)
- Build: `make docker-build`
- Shell: `make docker-bash` (mounts `vuljit/data` and `vuljit/outputs`)
- Inside container, run: `python -m vuljit --help`

## Environment Variables
Create `vuljit/.env` based on `vuljit/.env.example`.

Key entries:
- `VULJIT_DATA_DIR`, `VULJIT_OUTPUTS_DIR`, `VULJIT_METRICS_DIR`
- `VULJIT_VUL_CSV` (CSV with `package_name` column)
- `VULJIT_PROJECT_MAPPING` (defaults to `vuljit/mapping/project_mapping.csv`)
- `VULJIT_SRCDOWN_DIR`, `VULJIT_COVERAGE_DIR`, `VULJIT_COVERAGE_FILES` (comma-separated)
- `VULJIT_START_DATE`, `VULJIT_END_DATE`, `VULJIT_WORKERS`
- For Google/GitHub access if needed: `GOOGLE_APPLICATION_CREDENTIALS`, `GITHUB_TOKEN`

## CLI Commands
- `download-srcmap`: Downloads `[date].json` under `package/srcmap/` from GCS (oss-fuzz-coverage)
- `download-coverage`: Downloads and gzips coverage files (e.g., `summary.json`) under `package/reports/[date]/(target?)/linux/`
- `metrics code-text`: Runs existing shell pipeline that merges code and text metrics
- `metrics patch-coverage`: End-to-end pipeline (srcmap -> revisions CSV -> commit dates -> daily diffs -> patch coverage)
- `metrics coverage-aggregate`: Summarize coverage JSONs to CSVs per project
- `metrics aggregate-daily`: Join metrics + coverage (file/total + patch) and aggregate to per-day dataset
  - Uses mapping CSV (`project_id,dir`), default: `vuljit/mapping/project_mapping.csv` (override via `--mapping` or `VULJIT_PROJECT_MAPPING`)
- `prediction train`: Train and evaluate within-project models (default: random_forest)
- `prediction rq3`: Re-train on first half and score second half, write predicted risks

Options can be passed as flags or via environment variables.

### RQ3 Additional-Build Simulation Settings
- Walkforward scheduling honours the prediction settings `N_SPLITS_TIMESERIES` and `USE_ONLY_RECENT_FOR_TRAINING`. When invoking the additional-build CLI or notebooks, ensure the same values are passed (or exported via `RQ3_WALKFORWARD_SPLITS` / `RQ3_WALKFORWARD_USE_RECENT`) so the detection statistics and simulation windows stay aligned.
- Changing these values alters the fold boundaries, so regenerate both predictions and additional-build schedules after any update.

## Replicate Project Selection (Section 1)
To reproduce the candidate project count used in the paper (unique projects with at least one cloned repository):

- Ensure your mapping CSV and cloned repos are available.
  - Mapping CSV example: `vuljit/mapping/filtered_project_mapping.csv` (columns: `project_id,directory_name`)
  - Cloned repos root: a directory containing cloned repositories as top-level folders.
- Set the environment variables (or pass flags):
  - `VULJIT_PROJECT_MAPPING` -> mapping CSV path
  - `VULJIT_CLONED_REPOS_DIR` -> cloned repos directory
- Run:
  - `make selection-count`
  - Or directly: `python -m vuljit mapping count-candidates --csv <mapping.csv> --repos <cloned_repos_dir>`

The output shows:
- CSV rows with non-empty directory_name
- Rows matched (directory_name exists in cloned_repos)
- Unique directory_name entries in CSV
- Unique directory_name present/not present in cloned_repos
- Unique project_id in CSV
- Unique project_id with at least one matched repo

In our reference environment, the candidate projects count is 224.

## Notes
- Existing scripts remain intact; the CLI wraps them. Absolute paths are replaced by env/relative defaults.
- Intermediates go under `vuljit/data/intermediate`, final artifacts under `vuljit/outputs`.
- Patch-coverage requires cloned repos; add `--clone-repos` to let the CLI clone into `VULJIT_CLONED_REPOS_DIR`.

## Minimal End-to-End (single project)
- Prepare one line mapping or pass flags:
  - `python -m vuljit metrics aggregate-daily --project <ossfuzz_pkg> --dir <repo_dir_name>`
    - Inputs (defaults, override via `.env`):
      - Metrics: `VULJIT_METRICS_DIR/<dir>/<dir>_commit_metrics_with_tfidf.csv`
      - Coverage totals: `VULJIT_OUTPUTS_DIR/metrics/coverage_aggregate/<project>/<project>_total_and_date.csv`
      - Patch coverage: `VULJIT_OUTPUTS_DIR/metrics/patch_coverage/<project>/<project>_patch_coverage.csv`
    - Output: `VULJIT_BASE_DATA_DIR/<project>/<project>_daily_aggregated_metrics.csv`
- Train on the produced dataset:
  - `python -m vuljit prediction train -p <project>`

## Troubleshooting
- Ensure your `.env` is set and volumes are mounted in Docker.
- For GCS access, anonymous reads are used for public buckets; authenticated
  endpoints require `GOOGLE_APPLICATION_CREDENTIALS`.
