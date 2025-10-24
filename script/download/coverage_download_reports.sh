#!/bin/bash

#SBATCH --job-name=coverage_download_reports
#SBATCH --output=logs/coverage_download_reports_%A_%a.out
#SBATCH --error=errors/coverage_download_reports_%A_%a.err
#SBATCH --array=1
#SBATCH --time=100:00:00
#SBATCH --partition=cluster_long
#SBATCH --ntasks=1
#SBATCH --mem=210G
#SBATCH --cpus-per-task=30
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

PYTHON_EXEC="${PYTHON_EXEC:-python3}"

# Resolve repo root
script_dir="$(cd "$(dirname "$0")" && pwd)"
vuljit_dir="$(cd "$script_dir/.." && pwd)"

PYTHON_SCRIPT_PATH_1="${vuljit_dir}/download_gcs/coverage_download_reports.py"

# --- zip化スクリプトとパラメータの設定 ---
ZIP_SCRIPT_PATH="${vuljit_dir}/zip.sh"

SOURCE_PARENT_DIR_FOR_ZIP="${VULJIT_COVERAGE_DIR:-${vuljit_dir}/data/coverage_gz}"
NUM_TO_ZIP_PARAM="${VULJIT_NUM_TO_ZIP:-330}"
OUTPUT_ZIP_DIR_PARAM="${VULJIT_COVERAGE_ZIP_DIR:-${vuljit_dir}/outputs/coverage_zip}"

# ログ用、エラー用ディレクトリ作成
mkdir -p errors
mkdir -p logs

echo "Pythonスクリプトを実行します..."
${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_1}"
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "エラー: Pythonスクリプトの実行に失敗しました (終了コード: $PYTHON_EXIT_CODE)。zip化処理をスキップします。"
    exit $PYTHON_EXIT_CODE
fi

echo ""
echo "Pythonスクリプトの実行が完了しました。"
echo "次に、指定されたディレクトリのzip化を開始します..."
echo "--------------------------------------------------"
echo "zip化スクリプト: ${ZIP_SCRIPT_PATH}"
echo "対象の親ディレクトリ: ${SOURCE_PARENT_DIR_FOR_ZIP}"
echo "zip化する数: ${NUM_TO_ZIP_PARAM}"
echo "出力先ディレクトリ: ${OUTPUT_ZIP_DIR_PARAM}"
echo "--------------------------------------------------"

# 最初のスクリプトを実行してzip化を行う
# スクリプトに実行権限がない場合は `bash ${ZIP_SCRIPT_PATH} ...` のように呼び出します。
if [ -f "${ZIP_SCRIPT_PATH}" ]; then
    if [ ! -x "${ZIP_SCRIPT_PATH}" ]; then
        echo "情報: ${ZIP_SCRIPT_PATH} に実行権限がないため、bash経由で実行します。"
        bash "${ZIP_SCRIPT_PATH}" "${SOURCE_PARENT_DIR_FOR_ZIP}" "${NUM_TO_ZIP_PARAM}" "${OUTPUT_ZIP_DIR_PARAM}"
    else
        "${ZIP_SCRIPT_PATH}" "${SOURCE_PARENT_DIR_FOR_ZIP}" "${NUM_TO_ZIP_PARAM}" "${OUTPUT_ZIP_DIR_PARAM}"
    fi
    ZIP_EXIT_CODE=$?

    if [ $ZIP_EXIT_CODE -eq 0 ]; then
        echo "zip化処理が正常に完了しました。"
    else
        echo "エラー: zip化処理中に問題が発生しました (終了コード: $ZIP_EXIT_CODE)。"
        exit $ZIP_EXIT_CODE
    fi
else
    echo "エラー: zip化スクリプト '${ZIP_SCRIPT_PATH}' が見つかりません。"
    exit 1
fi

echo ""
echo "全ての処理が完了しました。"
