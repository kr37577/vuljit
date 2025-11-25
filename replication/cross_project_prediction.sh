#!/usr/bin/env bash
# Run cross-project predictions for all projects without Slurm by directly invoking main_per_project.py.
set -euo pipefail

vuljit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PRED_PY="${vuljit_dir}/scripts/modeling/main_per_project.py"
MAPPING_FILE="${VULJIT_PROJECT_MAPPING:-${vuljit_dir}/datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv}"
DATA_DIR="${VULJIT_BASE_DATA_DIR:-${vuljit_dir}/datasets/derived_artifacts/aggregate}"
TRAIN_SCOPE="${TRAIN_SCOPE:-exclude_target}"  # list | all | exclude_target
EVAL_MODE="${EVAL_MODE:-full}"                # fold | full

if [[ ! -f "${PRED_PY}" ]]; then
  echo "[error] prediction script not found: ${PRED_PY}" >&2
  exit 1
fi

projects=()
if [[ -f "${MAPPING_FILE}" ]]; then
  echo "[info] Using mapping file: ${MAPPING_FILE}"
  while IFS=, read -r project language main_repo homepage primary_contact vuln_count extra; do
    project="${project%$'\r'}"
    project="${project%\"}"
    project="${project#\"}"
    if [[ -n "${project}" && "${project}" != "project" ]]; then
      projects+=("${project}")
    fi
  done < <(tail -n +2 "${MAPPING_FILE}")
else
  echo "[warn] mapping file not found: ${MAPPING_FILE}. Falling back to directory names under ${DATA_DIR}"
  while IFS= read -r dir; do
    projects+=("$(basename "${dir}")")
  done < <(find "${DATA_DIR}" -maxdepth 1 -type d -print)
fi

if [[ ${#projects[@]} -eq 0 ]]; then
  echo "[error] no projects found to process." >&2
  exit 1
fi

echo "[info] Aggregate data dir: ${DATA_DIR}"
echo "[info] Train scope: ${TRAIN_SCOPE}, Eval mode: ${EVAL_MODE}"
echo "[info] Projects: ${#projects[@]}"

for project in "${projects[@]}"; do
  [[ -z "${project}" ]] && continue
  echo "------------------------------------------------------------------"
  echo "[info] Running cross-project prediction for project=${project}"

  VULJIT_BASE_DATA_DIR="${DATA_DIR}" \
  "${PYTHON_BIN}" "${PRED_PY}" \
    --project "${project}" \
    --cross-project \
    --train-scope "${TRAIN_SCOPE}" \
    --cross-project-mode "${EVAL_MODE}"
done

echo "[info] Completed all projects."
