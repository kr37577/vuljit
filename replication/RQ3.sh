#!/usr/bin/env bash
# Run RQ3 preparation and/or simulations locally (no Slurm). Choose steps via options.
set -euo pipefail

vuljit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RQ3_DIR="${vuljit_dir}/analysis/research_question3"
PYTHON_BIN="${PYTHON_BIN:-python3}"

prepare_script="${RQ3_DIR}/run_prepare_RQ3.sh"
sim_script="${RQ3_DIR}/rq3.sh"

run_prepare=1
run_sim=1

usage() {
  cat <<'USAGE'
Usage: bash replication/RQ3.sh [options]

Options (default: both steps):
  --prepare-only    Run only data preparation (run_prepare_RQ3.sh)
  --simulate-only   Run only simulations (rq3.sh)
  -h, --help        Show this help
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --prepare-only) run_sim=0 ;;
    --simulate-only) run_prepare=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] Unknown option: $arg" >&2; usage; exit 1 ;;
  esac
done

if [[ ${run_prepare} -eq 0 && ${run_sim} -eq 0 ]]; then
  echo "[error] Both steps disabled. Enable at least one step." >&2
  exit 1
fi

if [[ ! -f "${prepare_script}" || ! -f "${sim_script}" ]]; then
  echo "[error] RQ3 scripts not found under ${RQ3_DIR}" >&2
  exit 1
fi

# PYTHONPATH をリポジトリルートに通す（必要なら）
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${vuljit_dir}:${PYTHONPATH}"
else
  export PYTHONPATH="${vuljit_dir}"
fi

echo "[info] Changing directory to RQ3: ${RQ3_DIR}"
cd "${RQ3_DIR}" || exit 1

if [[ ${run_prepare} -eq 1 ]]; then
  echo "---------------------------------"
  echo "[info] RQ3 data preparation"
  bash "${prepare_script}"
fi

if [[ ${run_sim} -eq 1 ]]; then
  echo "---------------------------------"
  echo "[info] RQ3 simulations"
  bash "${sim_script}"
fi

echo "---------------------------------"
echo "[info] RQ3 completed."
