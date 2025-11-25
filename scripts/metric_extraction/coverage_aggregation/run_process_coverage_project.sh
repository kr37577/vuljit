#!/bin/bash

# Wrapper to execute process_coverage_project.py for one project or
# まとめて複数プロジェクトを処理するユーティリティ。pyenv の `py3`
# 環境を利用する点は既存スクリプトと揃えている。

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_process_coverage_project.sh [--single] <project_dir> [output_root]
  run_process_coverage_project.sh --all <projects_root> [output_root]

  --single       : (既定) 個別のプロジェクトディレクトリを処理
  --all          : 直下に複数プロジェクトディレクトリがあるルートをまとめて処理
  <project_dir>  : 日付ディレクトリにJSONが格納された解凍済みプロジェクト
  <projects_root>: 複数プロジェクトの親ディレクトリ
  [output_root]  : CSVの出力ルート（省略時: VULJIT_COVERAGE_METRICS_DIR または
                   リポジトリルート/datasets/derived_artifacts/coverage_metrics）

Examples:
  # 単一プロジェクトを処理
  run_process_coverage_project.sh /path/to/coverage/project_a

  # 出力先を指定
  run_process_coverage_project.sh /path/to/coverage/project_a /tmp/coverage_metrics

  # 複数プロジェクトをまとめて処理
  run_process_coverage_project.sh --all /path/to/coverage_report
USAGE
}

MODE="single"
if [[ $# -ge 1 ]]; then
  case "$1" in
    --single)
      MODE="single"
      shift
      ;;
    --all)
      MODE="batch"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
  esac
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

TARGET_DIR="$1"
if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "Error: directory '${TARGET_DIR}' not found." >&2
  exit 1
fi

OUT_ROOT_ARG=""
if [[ $# -eq 2 ]]; then
  OUT_ROOT_ARG="$2"
fi

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DEFAULT_OUT_ROOT="${VULJIT_COVERAGE_METRICS_DIR:-${REPO_ROOT}/datasets/derived_artifacts/coverage_metrics}"
OUT_ROOT="${OUT_ROOT_ARG:-${DEFAULT_OUT_ROOT}}"
mkdir -p "${OUT_ROOT}"

echo "Preparing Python environment via pyenv..."
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="${PYENV_ENV_NAME:-py3}"
PYTHON_EXEC="${PYTHON_EXEC:-${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python}"

if [[ ! -x "${PYTHON_EXEC}" ]]; then
  echo "Error: Python executable '${PYTHON_EXEC}' not found. Check your pyenv setup." >&2
  exit 1
fi

PYTHON_SCRIPT="${SCRIPT_DIR}/process_coverage_project.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "Error: Python script '${PYTHON_SCRIPT}' not found." >&2
  exit 1
fi

run_project() {
  local project_path="$1"
  local cmd=("${PYTHON_EXEC}" "${PYTHON_SCRIPT}" "${project_path}" "--out" "${OUT_ROOT}")
  echo "Running: ${cmd[*]}"
  "${cmd[@]}"
}

if [[ "${MODE}" == "single" ]]; then
  run_project "${TARGET_DIR}"
else
  echo "Processing all project directories under '${TARGET_DIR}'"
  shopt -s nullglob
  project_dirs=("${TARGET_DIR}"/*)
  shopt -u nullglob
  if [[ ${#project_dirs[@]} -eq 0 ]]; then
    echo "Warning: no subdirectories found under '${TARGET_DIR}'."
  fi
  for proj_dir in "${project_dirs[@]}"; do
    [[ -d "${proj_dir}" ]] || continue
    run_project "${proj_dir}"
  done
fi

echo "Done."
