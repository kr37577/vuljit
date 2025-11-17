#!/bin/bash
#SBATCH --job-name=extract_patch_coverage_metrics
#SBATCH --output=logs/extract_patch_coverage_metrics_%A_%a.out
#SBATCH --error=errors/extract_patch_coverage_metrics_%A_%a.err
#SBATCH --array=1
#SBATCH --time=4:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

# リポジトリルート推定と既定値設定
export DEFAULT_REPOS="/work/riku-ka/vuljit/datasets/raw/cloned_c_cpp_projects"
export DEFAULT_COMMIT_OUT="/work/riku-ka/vuljit/datasets/derived_artifacts/patch_coverage_inputs"
export DEFAULT_SRCMAP_ROOT="/work/riku-ka/vuljit/datasets/raw/srcmap_json"

PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
PYTHON_SCRIPT_PATH="/work/riku-ka/vuljit/scripts/metric_extraction/patch_coverage_pipeline/prepare_patch_coverage_inputs.py"

echo "Patch Coverage inputsを準備..."
"${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH}" \
  --srcmap-root "${DEFAULT_SRCMAP_ROOT}" \
  --repos "${DEFAULT_REPOS}" \
  --commit-out "${DEFAULT_COMMIT_OUT}" \
  --filter-to-main-repo
