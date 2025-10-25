#!/bin/bash

#SBATCH --job-name=merge_coverage_metrics
#SBATCH --output=logs/merge_coverage_metrics_%A_%a.out
#SBATCH --error=errors/merge_coverage_metrics_%A_%a.err
#SBATCH --array=1-279
#SBATCH --time=04:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=26
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

# PYTHON_EXEC="${PYTHON_EXEC:-python3}"
# script_dir="$(cd "$(dirname "$0")" && pwd)"

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
script_dir="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT_PATH_1="${script_dir}/aggregate_metrics_pipeline.py"

# 後で消す
VULJIT_BASE_DATA_DIR='/work/riku-ka/vuljit/datasets/derived_artifacts'
VULJIT_PROJECT_MAPPING='/work/riku-ka/vuljit/datasets/reference_mappings/filtered_project_mapping.csv'

# ログ用、エラー用ディレクトリ作成
mkdir -p errors
mkdir -p logs


echo "====== Starting Slurm Task ${SLURM_ARRAY_TASK_ID} ======"

# --- メイン処理 ---
MAPPING_FILE="${VULJIT_PROJECT_MAPPING:-${script_dir}/../project_mapping/filtered_project_mapping.csv}"
TARGET_LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID} + 1))p" ${MAPPING_FILE})
PROJECT_ID=$(echo ${TARGET_LINE} | cut -d',' -f1)
DIRECTORY_NAME=$(echo ${TARGET_LINE} | cut -d',' -f2)

if [ -z "${DIRECTORY_NAME}" ]; then
    DIRECTORY_NAME=${PROJECT_ID}
    echo "Info: directory_name is empty. Using project_id ('${PROJECT_ID}') instead."
fi

echo "Processing Project: ${PROJECT_ID}, Directory: ${DIRECTORY_NAME}"


# いったんベタ書き
# metrics_base_path = '/work/riku-ka/metrics_culculator/output_0802'
# coverage_base_project_path = '/work/riku-ka/metrics_culculator/coverage_zip/output_project_0802'
# patch_coverage_base_path = '/work/riku-ka/patch_coverage_culculater/patch_coverage_results_0802_now'
# output_base_path = '/work/riku-ka/daily_commit_summary_past_vul_0802_now' 

## いったん絶対パスでメトリクスやカバレッジを指定
METRICS_BASE_PATH="/work/riku-ka/metrics_culculator/output_0802"
COVERAGE_BASE_PROJECT_PATH="/work/riku-ka/metrics_culculator/coverage_zip/output_project_0802"
PATCH_COVERAGE_BASE_PATH="/work/riku-ka/patch_coverage_culculater/patch_coverage_results_0802_now"
# OUTPUT_BASE_PATH="/work/riku-ka/daily_commit_summary_past_vul_0802_now"

${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_1}" "${PROJECT_ID}" "${DIRECTORY_NAME}" \
  --metrics "${METRICS_BASE_PATH}" \
  --coverage "${COVERAGE_BASE_PROJECT_PATH}" \
  --patch-coverage "${PATCH_COVERAGE_BASE_PATH}" \
  --out "${VULJIT_BASE_DATA_DIR:-${script_dir}/../../datasets/derived_artifacts}"
# --out "${OUTPUT_BASE_PATH}"

# ${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_1}" "${PROJECT_ID}" "${DIRECTORY_NAME}" \
#   --metrics "${VULJIT_METRICS_DIR:-${script_dir}/../../datasets/metric_inputs}" \
#   --coverage "${VULJIT_COVERAGE_AGG_DIR:-${script_dir}/../../datasets/derived_artifacts/metrics/coverage_aggregate}" \
#   --patch-coverage "${VULJIT_PATCH_COV_DIR:-${script_dir}/../../datasets/derived_artifacts/metrics/patch_coverage}" \
#   --out "${VULJIT_BASE_DATA_DIR:-${script_dir}/../../datasets/derived_artifacts}"

echo "====== Finished Slurm Task ${SLURM_ARRAY_TASK_ID} ======"
