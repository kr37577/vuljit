#!/bin/bash

#SBATCH --job-name=metrics_extract
#SBATCH --output=logs/metrics_extract_%A_%a.out
#SBATCH --error=errors/metrics_extract_%A_%a.err
#SBATCH --array=1
#SBATCH --time=24:00:00
#SBATCH --partition=cluster_long
#SBATCH --ntasks=1
#SBATCH --mem=190G
#SBATCH --cpus-per-task=26
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

## Python 実行環境（未設定なら python3 を使用）
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

## リポジトリルートとスクリプトの場所を解決
script_dir="$(cd "$(dirname "$0")" && pwd)"
vuljit_dir="$(cd "$script_dir/.." && pwd)"

## 対象Pythonスクリプト（相対パス）
PYTHON_SCRIPT_PATH_1="${vuljit_dir}/scripts/metric_extraction/collect_code_text_metrics.py"

## メトリクス出力ディレクトリ（環境変数で上書き可能）
metrics_dir="${VULJIT_METRICS_DIR:-${vuljit_dir}/datasets/metric_inputs}"

mkdir -p "${metrics_dir}" || true

## プロジェクト一覧ファイル（このディレクトリに作成）
ls "${metrics_dir}" > "${script_dir}/project_dir.txt"
project_dir_file="${script_dir}/project_dir.txt"


# プロジェクトリストを1行ずつ読み込んで処理を実行
echo "各プロジェクトの処理を開始します..."
while IFS= read -r project_name || [[ -n "$project_name" ]]; do
  echo "======================================================================"
  echo "プロジェクトを処理中: ${project_name}" 
    # 処理対象のCSVファイルパスを構築   
    text_metrics_file="${metrics_dir}/${project_name}/${project_name}_commit_text_metrics_results_expanded.csv"
    code_metrics_file="${metrics_dir}/${project_name}/${project_name}_commit_metrics_with_tfidf.csv"
    output_file="${metrics_dir}/${project_name}/${project_name}_code_text_metrics.csv"
    # Pythonスクリプトを実行
    echo "Pythonスクリプトを実行中: ${PYTHON_SCRIPT_PATH_1}"
    "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}" \
      -p "${project_name}" \
      -c "${code_metrics_file}" \
      -t "${text_metrics_file}" \
      -o "${output_file}" \
    
    done < "${project_dir_file}"
echo "======================================================================"
echo "全プロジェクトの処理が完了しました。"
