#!/usr/bin/env bash
# Run the full RQ3 additional-build simulation pipeline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASETS_ROOT="${REPO_ROOT}/datasets"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_ROOT}"
fi

run_step() {
  local description="$1"
  shift
  echo "==> ${description}"
  "$@"
  echo ""
}

TIMELINE_ARGS=()
if [[ -n "${RQ3_TIMELINE_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  TIMELINE_ARGS=(${RQ3_TIMELINE_ARGS})
fi

SIM_ARGS=()
if [[ -n "${RQ3_SIMULATION_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  SIM_ARGS=(${RQ3_SIMULATION_ARGS})
fi

CROSS_PROJECT_ARGS=(
  --strategy1-mode cross_project
  --strategy2-mode cross_project
  --strategy3-mode cross_project
)

SIMPLE_OUTPUT_DIR="${RQ3_SIM_SIMPLE_OUTPUT_DIR:-${DATASETS_ROOT}/derived_artifacts/rq3/simulation_outputs_strategy4_simple}"
MULTI_OUTPUT_DIR="${RQ3_SIM_MULTI_OUTPUT_DIR:-${DATASETS_ROOT}/derived_artifacts/rq3/simulation_outputs_strategy4_multi}"

cd "${SCRIPT_DIR}"

export PYTHONLOGLEVEL="${PYTHONLOGLEVEL:-INFO}" 
export RQ3_LOG_LEVEL="${RQ3_LOG_LEVEL:-DEBUG}"

run_step "Generating build timelines" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/timeline_cli_wrapper.py" "${TIMELINE_ARGS[@]}"

run_step "Running additional-build simulation (Strategy4 Simple Regression)" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/simulate_additional_builds.py" \
  "${SIM_ARGS[@]}" \
  "${CROSS_PROJECT_ARGS[@]}" \
  --strategy4-mode simple \
  --output-dir "${SIMPLE_OUTPUT_DIR}"

run_step "Running additional-build simulation (Strategy4 Multi Regression)" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/simulate_additional_builds.py" \
  "${SIM_ARGS[@]}" \
  "${CROSS_PROJECT_ARGS[@]}" \
  --strategy4-mode multi \
  --output-dir "${MULTI_OUTPUT_DIR}"

echo "RQ3 simulation completed. Outputs are available under: ${DATASETS_ROOT}/derived_artifacts/rq3/simulation_outputs"
