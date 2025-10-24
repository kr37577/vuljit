#!/bin/bash
#SBATCH --job-name=patch_coverage
#SBATCH --output=logs/patch_coverage_%A_%a.out
#SBATCH --error=errors/patch_coverage_%A_%a.err
#SBATCH --array=1-277
#SBATCH --time=100:00:00
#SBATCH --partition=cluster_long
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=9
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"


PYTHON_SCRIPT_PATH_1="/work/riku-ka/patch_coverage_culculater/calculate_patch_coverage_per_project_test.py"

# ログ用、エラー用ディレクトリ作成
mkdir -p errors
mkdir -p logs

# --- 並列処理のための設定 ---
diff_dir="/work/riku-ka/patch_coverage_culculater/daily_diffs"
project_list_file="projects_dir.txt"

# プロジェクトのリストをファイルに作成（ジョブごとに実行する必要はないので、sbatch実行前に一度だけ手動で実行するか、マスターノードで実行するのが望ましい）
# ここでは、ジョブID 1が代表してファイルを作成するようにします。
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    ls -1 "${diff_dir}" > "${project_list_file}"
    # ファイルが作成されるまで少し待つ
    sleep 5
fi

# 全てのジョブがファイル作成を待つためのバリア
while [ ! -f "${project_list_file}" ]; do
  echo "Waiting for project_list.txt to be created..."
  sleep 2
done


# ジョブアレイのIDに基づいて、処理するプロジェクト名を取得
project_name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${project_list_file}")

# プロジェクト名が空でない場合のみ処理を実行
if [ -n "${project_name}" ]; then
    echo "======================================================================"
    echo "Job ID: ${SLURM_ARRAY_TASK_ID}"
    echo "プロジェクトを処理中: ${project_name}"
    echo "Pythonスクリプトを実行中: ${PYTHON_SCRIPT_PATH_1}"
    "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}" \
        -p "${project_name}"
    echo "======================================================================"
    echo "プロジェクトの処理が完了しました: ${project_name}"
else
    echo "Job ID ${SLURM_ARRAY_TASK_ID}: 処理するプロジェクトが見つかりませんでした。"
fi