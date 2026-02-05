#!/bin/bash

#SBATCH --job-name=merge_coverage_metrics
#SBATCH --output=logs/merge_coverage_metrics_%A_%a.out
#SBATCH --error=errors/merge_coverage_metrics_%A_%a.err
#SBATCH --array=1
#SBATCH --time=04:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL 

# PYTHON_EXEC="${PYTHON_EXEC:-python3}"
# script_dir="$(cd "$(dirname "$0")" && pwd)"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
script_dir=""
PYTHON_SCRIPT_PATH_1="${script_dir}/analyze_comparison.py"
PYTHON_SCRIPT_PATH_2="${script_dir}/analyze_trends_comparison.py"


echo "====== Starting Slurm Task ${SLURM_ARRAY_TASK_ID} ======"
python ${PYTHON_SCRIPT_PATH_1} 
echo "---------------------------------"
echo "====== Starting Slurm Task ${SLURM_ARRAY_TASK_ID} ======"
python ${PYTHON_SCRIPT_PATH_2}  
echo "---------------------------------"
echo "====== All tasks completed ======"
