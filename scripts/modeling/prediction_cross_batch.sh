#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_DATA_DIR="/work/riku-ka/vuljit/datasets/derived_artifacts/aggregate"
DATA_DIR="$DEFAULT_DATA_DIR"
TRAIN_SCOPE="exclude_target"
EVAL_MODE="full"
TRAIN_PROJECTS=""
TRAIN_PROJECTS_FILE=""
PROJECT_LIST=()

usage() {
  cat <<'USAGE'
Usage: prediction_cross_batch.sh [options]

Submits one Slurm job per target project by invoking prediction_cross.sh
with exactly one -p argument. Use the same dataset/options you would for
the single-project script.

Options:
  -d DIR       Aggregated metrics root (default: /work/riku-ka/vuljit/datasets/derived_artifacts/aggregate)
  -p NAME      Target project (may repeat). If omitted, scan DIR for all projects.
  -s SCOPE     Training scope: list | all | exclude_target (default: exclude_target)
  -P LIST      Comma-separated projects to train on (used when -s list)
  -F FILE      File with one project name per line for training (used when -s list)
  -m MODE      Cross-project evaluation mode: fold | full (default: full)
  -h           Show this help.

Examples:
  # Submit every project in DIR
  ./prediction_cross_batch.sh

  # Submit only specific targets
  ./prediction_cross_batch.sh -p libxml2 -p curl
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
  mapfile -t PROJECT_LIST < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
fi

if [[ ${#PROJECT_LIST[@]} -eq 0 ]]; then
  echo "[error] No target projects found under $DATA_DIR" >&2
  exit 1
fi

if [[ "$TRAIN_SCOPE" == "list" ]] && [[ -z "$TRAIN_PROJECTS" && -z "$TRAIN_PROJECTS_FILE" ]]; then
  echo "[error] train scope 'list' requires -P or -F to be set." >&2
  exit 2
fi

echo "[info] Submitting ${#PROJECT_LIST[@]} project(s) from $DATA_DIR"

overall_rc=0
for project in "${PROJECT_LIST[@]}"; do
  sbatch_cmd=(sbatch "$SCRIPT_DIR/prediction_cross.sh" -d "$DATA_DIR" -p "$project" -s "$TRAIN_SCOPE" -m "$EVAL_MODE")
  [[ -n "$TRAIN_PROJECTS" ]] && sbatch_cmd+=(-P "$TRAIN_PROJECTS")
  [[ -n "$TRAIN_PROJECTS_FILE" ]] && sbatch_cmd+=(-F "$TRAIN_PROJECTS_FILE")

  echo "[info] sbatch submit: ${sbatch_cmd[*]}"
  if "${sbatch_cmd[@]}"; then
    continue
  fi
  echo "[error] sbatch submission failed for $project" >&2
  overall_rc=1
done

exit $overall_rc
