#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RQ3="${SCRIPT_DIR}"

python "$RQ3/extract_build_counts.py" \
  --projects-dir "$REPO/datasets/raw/oss-fuzz/projects" \
  --out "$REPO/datasets/derived_artifacts/oss_fuzz_build_counts/project_build_counts.csv"

python "$RQ3/measure_detection_time.py" \
  --vulns-csv "$REPO/datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv" \
  --issues-csv "$REPO/datasets/derived_artifacts/issue_redirect_mapping_selenium/issue_redirect_mapping_selenium_.csv" \
  --repos-root "$REPO/datasets/raw/cloned_c_cpp_projects" \
  -o "$REPO/datasets/derived_artifacts/detection_time/detection_time_results.csv"
