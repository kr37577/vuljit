#!/bin/bash

#SBATCH --job-name=prediction_one            # sbatch 側で --job-name で上書き可
#SBATCH --output=logs/pred_%x_%j.out         # %x=job名, %j=jobID
#SBATCH --error=errors/pred_%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=13

# Run cross-project predictions for one or more targets.
# Each target project is evaluated using models trained on other projects.

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"


SCRIPT_DIR="/work/riku-ka/vuljit/scripts/modeling"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_DIR}/modeling"

DEFAULT_DATA_DIR="/work/riku-ka/vuljit/datasets/derived_artifacts/aggregate"
DATA_DIR="${DEFAULT_DATA_DIR}"
TRAIN_SCOPE="exclude_target"
EVAL_MODE="full"
TRAIN_PROJECTS=""
TRAIN_PROJECTS_FILE=""
PROJECT_LIST=()

usage() {
  cat <<'USAGE'
Usage: prediction_cross.sh [options]

Options:
  -d DIR       Aggregated metrics root (default: /work/riku-ka/vuljit/datasets/derived_artifacts/aggregate)
  -p NAME      Target project (exactly one). Use prediction_cross_batch.sh to submit many jobs.
  -s SCOPE     Training scope: list | all | exclude_target (default: exclude_target)
  -P LIST      Comma-separated projects to train on (used when -s list)
  -F FILE      File with one project name per line for training (used when -s list)
  -m MODE      Cross-project evaluation mode: fold | full (default: full)
  -h           Show this help.

Environment:
  VULJIT_BASE_DATA_DIR is set to DIR for each run if not already defined.

Examples:
  # Train on the libxml2 project
  ./prediction_cross.sh -p libxml2 -s exclude_target

  # Submit many projects via Slurm
  ./prediction_cross_batch.sh -s list -P "ffmpeg,openssl"
USAGE
}

while getopts ":d:p:s:P:F:m:h" opt; do
  case "$opt" in
    d) DATA_DIR="$OPTARG" ;;
    p) PROJECT_LIST+=("$OPTARG") ;;
    s) TRAIN_SCOPE="$OPTARG" ;;
    P) TRAIN_PROJECTS="$OPTARG" ;;
    F) TRAIN_PROJECTS_FILE="$OPTARG" ;;
    m) EVAL_MODE="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "[error] Unknown option: -$OPTARG" >&2; usage; exit 2 ;;
    :) echo "[error] Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

case "$TRAIN_SCOPE" in
  list|all|exclude_target) ;;
  *) echo "[error] Invalid --train-scope: $TRAIN_SCOPE" >&2; exit 2 ;;
esac

case "$EVAL_MODE" in
  fold|full) ;;
  *) echo "[error] Invalid cross-project mode: $EVAL_MODE" >&2; exit 2 ;;
esac

if [[ ${#PROJECT_LIST[@]} -eq 0 ]]; then
  cat >&2 <<'ERR'
[error] prediction_cross.sh now processes exactly one project per invocation.
        Pass a single -p <project>. To submit many projects, use prediction_cross_batch.sh.
ERR
  exit 2
fi

if [[ ${#PROJECT_LIST[@]} -ne 1 ]]; then
  echo "[error] Provide exactly one -p <project> per job." >&2
  exit 2
fi

if [[ "$TRAIN_SCOPE" == "list" ]] && [[ -z "$TRAIN_PROJECTS" && -z "$TRAIN_PROJECTS_FILE" ]]; then
  echo "[error] train scope 'list' requires -P or -F to be set." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[error] python3 not found. Set PYTHON_BIN manually." >&2
  exit 1
fi

echo "[info] Using data dir: $DATA_DIR"
echo "[info] Target project: ${PROJECT_LIST[0]}"
echo "[info] Training scope: $TRAIN_SCOPE"
echo "[info] Cross eval mode: $EVAL_MODE"

run_for_project() {
  local project="$1"
  local -a cmd=("$PYTHON_EXEC" "${MODEL_DIR}/main_per_project.py" "--project" "$project" "--cross-project" "--train-scope" "$TRAIN_SCOPE" "--cross-project-mode" "$EVAL_MODE")
  [[ -n "$TRAIN_PROJECTS" ]] && cmd+=("--train-projects" "$TRAIN_PROJECTS")
  [[ -n "$TRAIN_PROJECTS_FILE" ]] && cmd+=("--train-projects-file" "$TRAIN_PROJECTS_FILE")

  echo "======================================================================"
  echo "[info] Target: $project"
  echo "[info] Command: ${cmd[*]}"
  echo "----------------------------------------------------------------------"
  VULJIT_BASE_DATA_DIR="${VULJIT_BASE_DATA_DIR:-$DATA_DIR}" "${cmd[@]}"
  echo "======================================================================"
}

if run_for_project "${PROJECT_LIST[0]}"; then
  echo "[info] Completed target: ${PROJECT_LIST[0]}"
  exit 0
fi

echo "[error] Failed target: ${PROJECT_LIST[0]}" >&2
exit 1
