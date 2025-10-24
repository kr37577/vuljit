#!/bin/bash
set -euo pipefail

# 対象データディレクトリ（プロジェクト名の一覧をここから取得）
DATA_DIR='/work/riku-ka/vuljit/data'
SBATCH_SCRIPT="/work/riku-ka/vuljit/prediction/predict_one_project_regression.sh"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi
cd /work/riku-ka/vuljit



mkdir -p logs errors

# プロジェクト一覧を取得（深さ1のディレクトリ名）
mapfile -t PROJECTS < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

echo "[info] submitting ${#PROJECTS[@]} jobs from: $DATA_DIR"
for p in "${PROJECTS[@]}"; do
  # ジョブ名をプロジェクトで上書きし、環境に PROJECT を渡す
  jid=$(sbatch --parsable --job-name="pred_${p}" --export=ALL,PROJECT="${p}" "$SBATCH_SCRIPT")
  echo "submitted: ${p} -> job ${jid}"
done
