#!/bin/bash
#SBATCH --job-name=extract_coverage_metrics
#SBATCH --output=logs/extract_coverage_metrics_%A_%a.out
#SBATCH --error=errors/extract_coverage_metrics_%A_%a.err
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

PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
SHELL_SCRIPT_PATH="/work/riku-ka/vuljit/scripts/metric_extraction/coverage_aggregation/run_process_coverage_project.sh"

echo "Coverageレポートの集計を開始..."
# --- スクリプトとディレクトリの設定 ---
bash "${SHELL_SCRIPT_PATH}" --all /work/riku-ka/vuljit/datasets/raw/coverage_report  
