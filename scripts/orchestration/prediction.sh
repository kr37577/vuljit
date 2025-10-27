#!/bin/bash
set -euo pipefail

# 対象データディレクトリ（プロジェクト名の一覧をここから取得）
DATA_DIR='/work/riku-ka/vuljit/datasets/raw'
SBATCH_SCRIPT="/work/riku-ka/vuljit/scripts/modeling/predict_one_project.sh"

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi
cd /work/riku-ka/vuljit



mkdir -p logs errors

# プロジェクト一覧を取得（深さ1のディレクトリ名）
mapfile -t PROJECTS < <(find "$DATA_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if command -v sbatch >/dev/null 2>&1; then
  echo "[info] submitting ${#PROJECTS[@]} jobs via sbatch from: $DATA_DIR"
  for p in "${PROJECTS[@]}"; do
    jid=$(sbatch --parsable --job-name="pred_${p}" --export=ALL,PROJECT="${p}" "$SBATCH_SCRIPT")
    echo "submitted: ${p} -> job ${jid}"
  done
  exit 0
fi

echo "[info] sbatch not found; running projects sequentially."
overall_rc=0
for p in "${PROJECTS[@]}"; do
  echo "[info] running project locally: ${p}"
  if bash "$SBATCH_SCRIPT" "${p}"; then
    echo "[info] completed: ${p}"
  else
    echo "[error] failed: ${p}" >&2
    overall_rc=1
  fi
done
exit $overall_rc
