#!/bin/bash

#SBATCH --job-name=vccfinder_metrics_extract
#SBATCH --output=logs/metrics_extract_%A_%a.out
#SBATCH --error=errors/metrics_extract_%A_%a.err
#SBATCH --array=1
#SBATCH --time=120:00:00
#SBATCH --partition=cluster_low
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 


mkdir -p errors
mkdir -p logs



# エラーが発生した場合にスクリプトを終了する
set -e
set -o pipefail

# --- 設定 ---

## Python 実行環境（未設定なら python3 を使用）
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

## パス解決
script_dir="$(cd "$(dirname "$0")" && pwd)"
vuljit_dir="$(cd "$script_dir/../.." && pwd)"

# 実行するPythonスクリプトのパス（相対）
PYTHON_SCRIPT="${script_dir}/vccfinder_commit_message_metrics.py"

# 処理対象のファイルが含まれるベースディレクトリ（環境変数で上書き可）
SEARCH_BASE_DIR="${VULJIT_OUTPUT_DIR:-${vuljit_dir}/data/metrics_output}"

# プロジェクトリストファイル（このディレクトリ配下）
PROJECT_LIST_FILE="${script_dir}/project_dir.txt"

# --- メイン処理 ---

# Pythonスクリプトの存在確認
if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "エラー: Pythonスクリプトが見つかりません: ${PYTHON_SCRIPT}"
  exit 1
fi

echo "プロジェクトリストを作成しています..."
# ls "${SEARCH_BASE_DIR}" > "${PROJECT_LIST_FILE}"
# echo "プロジェクトリストを '${PROJECT_LIST_FILE}' に保存しました。"

# プロジェクトリストを1行ずつ読み込んで処理を実行
echo "各プロジェクトの処理を開始します..."
while IFS= read -r project_name || [[ -n "$project_name" ]]; do
  echo "======================================================================"
  echo "プロジェクトを処理中: ${project_name}"
  echo "  Pythonスクリプトを実行しています..."

  # Pythonスクリプトを実行
  if "${PYTHON_EXEC}" "${PYTHON_SCRIPT}" -p "${project_name}"; then
    echo "  '${project_name}' の処理が正常に完了しました。"
  else
    echo "  エラー: '${project_name}' の処理中にPythonスクリプトでエラーが発生しました。"
    # エラーが発生しても次のプロジェクトに進む
  fi

done < "${PROJECT_LIST_FILE}"

echo "======================================================================"
echo "すべてのプロジェクトの処理が完了しました。"
