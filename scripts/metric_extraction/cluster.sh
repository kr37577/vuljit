#!/bin/bash
#SBATCH --job-name=extract_commit_metrics
#SBATCH --output=logs/extract_commit_metrics_%A_%a.out
#SBATCH --error=errors/extract_commit_metrics_%A_%a.err
#SBATCH --array=1
#SBATCH --time=100:00:00
#SBATCH --partition=cluster_low
#SBATCH --ntasks=1
#SBATCH --mem=300G
#SBATCH --cpus-per-task=52

echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
PYTHON_SCRIPT_PATH_1="/work/riku-ka/vuljit/scripts/metric_extraction/build_commit_metrics_pipeline.py"
# ログ用、エラー用ディレクトリ作成
mkdir -p errors
mkdir -p logs

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
  --vuln-csv "/work/riku-ka/vuljit/datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv" \
  --workers 52 \
  --since 20180101 \
  --until 20251001 \
  --metrics-dir "/work/riku-ka/vuljit/datasets/derived_artifacts/commit_metrics/" \
  "${EXTRA_ARGS[@]}"

echo "====== Task Completed ======"
