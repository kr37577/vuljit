#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
vuljit_dir="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"  # 必要に応じて上書き


PYTHON_SCRIPT_PATH_1="${SCRIPT_DIR}/build_commit_metrics_pipeline.py"
usage() {
  cat <<EOF
Usage: $(basename "$0") [-s since] [-u until] <project_name> <repo_path> [additional args...]

Options:
  -s since   Default since date to pass as --since (e.g., 20180101). Ignored if --since already provided.
  -u until   Default until date to pass as --until (e.g., 20251001). Ignored if --until already provided.
  -h         Show this help message.

All remaining arguments after <repo_path> are forwarded to build_commit_metrics_pipeline.py.
EOF
}

since_arg=""
until_arg=""

while getopts ":s:u:h" opt; do
  case "${opt}" in
    s) since_arg="${OPTARG}" ;;
    u) until_arg="${OPTARG}" ;;
    h)
      usage
      exit 0
      ;;
    \?)
      echo "ERROR: Invalid option -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
    :)
      echo "ERROR: Option -${OPTARG} requires an argument." >&2
      usage >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND - 1))

if [[ $# -lt 2 ]]; then
  echo "ERROR: Missing required arguments." >&2
  usage >&2
  exit 1
fi

PROJECT_NAME="$1"
REPO_PATH="$2"
shift 2
EXTRA_ARGS=("$@")

contains_arg() {
  local needle="$1"
  shift
  for arg in "$@"; do
    if [[ "$arg" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -n "${since_arg}" ]] && ! contains_arg "--since" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS=(--since "${since_arg}" "${EXTRA_ARGS[@]}")
fi

if [[ -n "${until_arg}" ]] && ! contains_arg "--until" "${EXTRA_ARGS[@]}"; then
  EXTRA_ARGS=(--until "${until_arg}" "${EXTRA_ARGS[@]}")
fi

if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "ERROR: Python executable not found at ${PYTHON_EXEC}" >&2
  exit 1
fi

if [[ ! -d "${REPO_PATH}" ]]; then
  echo "ERROR: Repository directory not found: ${REPO_PATH}" >&2
  exit 1
fi

task_id="${SLURM_ARRAY_TASK_ID:-0}"
echo "====== Starting Slurm Task ${task_id} ======"

# --- メイン処理 ---
echo "Running commit metrics pipeline for project '${PROJECT_NAME}' (repo: ${REPO_PATH})"
"${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}" \
  --project "${PROJECT_NAME}" \
  --repo "${REPO_PATH}" \
  --vuln-csv "${vuljit_dir}/datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv" \
  --workers 4 \
  --since 20180101 \
  --until 20251001 \
  --metrics-dir "${vuljit_dir}/datasets/derived_artifacts/commit_metrics/" \
  "${EXTRA_ARGS[@]}"

echo "====== Task Completed ======"
