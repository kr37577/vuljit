#!/usr/bin/env bash

set -euo pipefail

vuljit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"  # 必要に応じて上書き

# コミットメトリクスの収集
bash "${vuljit_dir}/scripts/project_mapping/clone_and_run_projects.sh" \
  -r "${vuljit_dir}/scripts/metric_extraction/build_commit_metrics_pipeline.sh" \
  -- --since 20180101 --until 20251001


# Coverageレポートの集約
bash "${vuljit_dir}/scripts/metric_extraction/coverage_aggregation/run_process_coverage_project.sh" \
  --all "${vuljit_dir}/datasets/raw/coverage_report"

# パッチカバレッジ用の入力データセットの準備
bash "${vuljit_dir}/scripts/metric_extraction/patch_coverage_pipeline/prepare_patch_coverage_inputs.sh"
# パッチカバレッジの計算
PYTHON_EXEC=python3
INPUT_DIR="${vuljit_dir}/datasets/derived_artifacts/patch_coverage_inputs"
REPOS_DIR="${vuljit_dir}/datasets/raw/cloned_c_cpp_projects"  # 必要なら変更
for csv in "${INPUT_DIR}"/revisions_with_commit_date_*.csv; do
  proj="$(basename "${csv}" .csv | sed 's/^revisions_with_commit_date_//')"
  echo "project=${proj}"
  VULJIT_CLONED_REPOS_DIR="${REPOS_DIR}" \
  "${PYTHON_EXEC}" ${vuljit_dir}/scripts/metric_extraction/patch_coverage_pipeline/run_culculate_patch_coverage_pipeline.py \
    --project "${proj}" --workers 8
done

