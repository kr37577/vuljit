#!/bin/bash
#SBATCH --job-name=patch_coverage
#SBATCH --output=logs/patch_coverage_%A_%a.out
#SBATCH --error=errors/patch_coverage_%A_%a.err
#SBATCH --array=1
#SBATCH --time=4:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4


# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
PYTHON_PATH="/work/riku-ka/vuljit/scripts/metric_extraction/patch_coverage_pipeline/run_culculate_patch_coverage_pipeline.py"

# 引数を受け取る


# --- 引数処理（第1引数がプロジェクト名、それ以降は Python へそのまま渡す） ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <project_name> [extra-args...]" >&2
  exit 1
fi
PROJECT="$1"
shift
EXTRA_ARGS=("$@")

mkdir -p logs errors

echo "Using Python interpreter: ${PYTHON_EXEC}"
echo "Project: ${PROJECT}"
echo "追加引数: ${EXTRA_ARGS[*]:-<none>}"

# --- パッチカバレッジ計算スクリプトの実行 ---
echo "パッチカバレッジ計算スクリプトの実行..."
"${PYTHON_EXEC}" "${PYTHON_PATH}" --project "${PROJECT}" "${EXTRA_ARGS[@]}"