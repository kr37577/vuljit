#!/bin/bash

#SBATCH --job-name=metrics_extract
#SBATCH --output=logs/metrics_extract_%A_%a.out
#SBATCH --error=errors/metrics_extract_%A_%a.err
#SBATCH --array=1
#SBATCH --time=1000:00:00
#SBATCH --partition=cluster_low
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

# --- Python環境の準備 ---
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

# Resolve repo root
script_dir="$(cd "$(dirname "$0")" && pwd)"
vuljit_dir="$(cd "$script_dir/.." && pwd)"

# Defaults (can be overridden by args or ENV)
DEFAULT_VUL_CSV="${VULJIT_VUL_CSV:-${vuljit_dir}/data/oss_fuzz_vulns.csv}"
DEFAULT_OUTPUT_DIR="${VULJIT_SRCDOWN_DIR:-${vuljit_dir}/data/srcmap_json}"
DEFAULT_START_DATE="${VULJIT_START_DATE:-20160101}"
DEFAULT_END_DATE="${VULJIT_END_DATE:-20251231}"
DEFAULT_WORKERS="${VULJIT_WORKERS:-8}"

# Args: [csv_path] [start_date YYYYMMDD] [end_date YYYYMMDD] [output_dir] [workers]
CSV_PATH="${1:-$DEFAULT_VUL_CSV}"
START_DATE="${2:-$DEFAULT_START_DATE}"
END_DATE="${3:-$DEFAULT_END_DATE}"
OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"
WORKERS="${5:-$DEFAULT_WORKERS}"

PYTHON_SCRIPT_PATH_1="${vuljit_dir}/download_gcs/download_srcmap.py"

echo "実行設定:"
echo "  CSV:        $CSV_PATH"
echo "  start_date: $START_DATE"
echo "  end_date:   $END_DATE"
echo "  out_dir:    $OUTPUT_DIR"
echo "  workers:    $WORKERS"
echo "  python:     $PYTHON_EXEC"
echo "  script:     $PYTHON_SCRIPT_PATH_1"

# 実行
echo "Pythonスクリプトを実行します..."
"$PYTHON_EXEC" "$PYTHON_SCRIPT_PATH_1" "$CSV_PATH" "$START_DATE" "$END_DATE" -d "$OUTPUT_DIR" -w "$WORKERS"
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "エラー: Pythonスクリプトの実行に失敗しました (終了コード: $PYTHON_EXIT_CODE)。"
    exit $PYTHON_EXIT_CODE
fi

echo ""
echo "Pythonスクリプトの実行が完了しました。"
