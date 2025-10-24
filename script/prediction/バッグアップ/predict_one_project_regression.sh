#!/bin/bash
#SBATCH --job-name=prediction_one            # sbatch 側で --job-name で上書き可
#SBATCH --output=logs/pred_%x_%j.out         # %x=job名, %j=jobID
#SBATCH --error=errors/pred_%x_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4

set -euo pipefail

# --- プロジェクト名の取得（引数 > 環境変数 PROJECT）---
PROJECT="${1:-${PROJECT:-}}"
if [[ -z "${PROJECT}" ]]; then
  echo "[error] PROJECT is empty. pass as first arg or --export=PROJECT=..."
  exit 2
fi

# --- 作業ディレクトリ/ログ用 ---
cd /work/riku-ka/vuljit
mkdir -p logs errors

# --- Python環境（pyenv が無ければ python3 にフォールバック） ---
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
if [[ -x "${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python" ]]; then
  PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
else
  PYTHON_EXEC="$(command -v python3)"
fi


export VULJIT_RANDOM_STATE="${VULJIT_RANDOM_STATE:-42}"
export VULJIT_STRICT_REPRO="${VULJIT_STRICT_REPRO:-1}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "======================================================================"
echo "Project: ${PROJECT}"
echo "Python : ${PYTHON_EXEC}"
echo "Running: main_per_project_regression.py --p ${PROJECT}"
"${PYTHON_EXEC}"  /work/riku-ka/vuljit/prediction/main_per_project_regression.py  --p "${PROJECT}"
rc=$?
echo "Exit code: $rc"
echo "======================================================================"
exit $rc