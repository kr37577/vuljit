#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/workspace/RAISE"
cd "${REPO_ROOT}"

load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${env_file}"
    set +a
  fi
}

# Support both the project's standard .env and docker-specific overrides.
load_env_file "${REPO_ROOT}/.env"
load_env_file "${REPO_ROOT}/.env.docker"

cmd="${1:-run_all}"
shift || true

case "${cmd}" in
  run_all)
    exec bash "${REPO_ROOT}/run_all_process.sh" "$@"
    ;;
  run_step)
    step_name="${1:-}"
    if [[ -z "${step_name}" ]]; then
      echo "[error] run_step requires a step name (e.g., RQ3 or data_acquisition)." >&2
      exit 2
    fi
    shift || true
    step_script="${step_name}"
    if [[ "${step_script}" != *.sh ]]; then
      step_script="${step_script}.sh"
    fi
    step_path="${REPO_ROOT}/replication/${step_script}"
    if [[ ! -f "${step_path}" ]]; then
      echo "[error] step script not found: ${step_path}" >&2
      exit 1
    fi
    exec bash "${step_path}" "$@"
    ;;
  shell)
    exec bash "$@"
    ;;
  *)
    exec "$cmd" "$@"
    ;;
esac
