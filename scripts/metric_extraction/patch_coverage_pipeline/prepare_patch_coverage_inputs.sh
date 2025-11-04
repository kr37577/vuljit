#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_PY="${SCRIPT_DIR}/prepare_patch_coverage_inputs.py"

if [[ ! -f "${PIPELINE_PY}" ]]; then
  echo "エラー: Pythonパイプラインスクリプトが見つかりません: ${PIPELINE_PY}" >&2
  exit 1
fi

DEFAULT_SRCMAP_ROOT="${DEFAULT_SRCMAP_ROOT:-${REPO_ROOT}/datasets/raw/srcmap_json}"
DEFAULT_REPOS="${DEFAULT_REPOS:-${REPO_ROOT}/data/intermediate/cloned_repos}"
DEFAULT_COMMIT_OUT="${DEFAULT_COMMIT_OUT:-${REPO_ROOT}/datasets/derived_artifacts/patch_coverage_inputs}"

PY_ARGS=(
  --srcmap-root "${DEFAULT_SRCMAP_ROOT}"
  --repos "${DEFAULT_REPOS}"
  --commit-out "${DEFAULT_COMMIT_OUT}"
)

"${PYTHON_BIN}" "${PIPELINE_PY}" "${PY_ARGS[@]}" "$@"
