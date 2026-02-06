#!/usr/bin/env bash
set -euo pipefail

vuljit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"  # 必要に応じて上書き

# 実行フラグ（デフォルトですべて実行）
run_vulcsv=1
run_coverage=1
run_srcmap=1
run_selenium=1

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# Environment-based skip controls for container/batch execution.
if is_truthy "${SKIP_VULCSV:-0}"; then run_vulcsv=0; fi
if is_truthy "${SKIP_COVERAGE:-0}"; then run_coverage=0; fi
if is_truthy "${SKIP_SRCDOWN:-0}"; then run_srcmap=0; fi
if is_truthy "${SKIP_SELENIUM:-0}"; then run_selenium=0; fi

usage() {
  cat <<'USAGE'
Usage: bash replication/data_acquisition.sh [options]

Options (all ON by default, disable with --no-xxx):
  --no-vulcsv      Skip ossfuzz_vulnerability_issue_report_extraction.py
  --no-coverage    Skip coverage_download_reports.py
  --no-srcmap      Skip download_srcmap.py
  --no-selenium    Skip osv_monorail_selenium.py
  -h, --help       Show this help
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --no-vulcsv) run_vulcsv=0 ;;
    --no-coverage) run_coverage=0 ;;
    --no-srcmap) run_srcmap=0 ;;
    --no-selenium) run_selenium=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] Unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

# .env を読み込み（任意）
if [ -f "${vuljit_dir}/.env" ]; then
  set -a
  # shellcheck source=/dev/null
  source "${vuljit_dir}/.env"
  set +a
fi

###############################################
# 1) ossfuzz_vulnerability_issue_report_extraction.py
if [[ $run_vulcsv -eq 1 ]]; then
  echo "[info] Running ossfuzz_vulnerability_issue_report_extraction.py"
  PYTHON_SCRIPT_PATH_3="${vuljit_dir}/scripts/data_acquisition/ossfuzz_vulnerability_issue_report_extraction.py"
  # 日付形式は YYYY-MM-DD または ISO8601
  "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_3}" --published-since "2018-08-22" --published-until "2025-06-01"
else
  echo "[info] Skipped ossfuzz_vulnerability_issue_report_extraction.py (--no-vulcsv)"
fi

##############################################
# 2) coverage_download_reports.py
if [[ $run_coverage -eq 1 ]]; then
  echo "[info] Running coverage_download_reports.py"
  PYTHON_SCRIPT_PATH_1="${vuljit_dir}/scripts/data_acquisition/coverage_download_reports.py"
  DEFAULT_COVERAGE_DIR="${VULJIT_COVERAGE_DIR:-${vuljit_dir}/datasets/raw/coverage_report}"
  export VULJIT_COVERAGE_DIR="$DEFAULT_COVERAGE_DIR"

  "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}"
else
  echo "[info] Skipped coverage_download_reports.py (--no-coverage)"
fi

##############################################
# 3) download_srcmap.py
if [[ $run_srcmap -eq 1 ]]; then
  echo "[info] Running download_srcmap.py"
  # Defaults (can be overridden by args or ENV)
  DEFAULT_VUL_CSV="${VULJIT_VUL_CSV:-${vuljit_dir}/datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv}"
  DEFAULT_OUTPUT_DIR="${VULJIT_SRCDOWN_DIR:-${vuljit_dir}/datasets/raw/srcmap_json}"
  DEFAULT_START_DATE="${VULJIT_START_DATE:-20160101}"
  DEFAULT_END_DATE="${VULJIT_END_DATE:-20250802}"
  DEFAULT_WORKERS="${VULJIT_WORKERS:-8}"

  CSV_PATH="${DEFAULT_VUL_CSV}"
  START_DATE="${DEFAULT_START_DATE}"
  END_DATE="${DEFAULT_END_DATE}"
  OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
  WORKERS="${DEFAULT_WORKERS}"

  PYTHON_SCRIPT_PATH_2="${vuljit_dir}/scripts/data_acquisition/download_srcmap.py"
  "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_2}" "${CSV_PATH}" "${START_DATE}" "${END_DATE}" -d "${OUTPUT_DIR}" -w "${WORKERS}"
else
  echo "[info] Skipped download_srcmap.py (--no-srcmap)"
fi

##############################################
# 4) osv_monorail_selenium.py
if [[ $run_selenium -eq 1 ]]; then
  echo "[info] Running osv_monorail_selenium.py"
  PYTHON_SCRIPT_PATH_4="${vuljit_dir}/scripts/data_acquisition/osv_monorail_selenium.py"
  "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_4}"
else
  echo "[info] Skipped osv_monorail_selenium.py (--no-selenium)"
fi
