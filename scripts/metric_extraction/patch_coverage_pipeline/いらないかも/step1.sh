#!/bin/bash
#SBATCH --job-name=create_revision
#SBATCH --output=logs/create_revision_%A_%a.out
#SBATCH --error=errors/create_revision_%A_%a.err
#SBATCH --array=1
#SBATCH --time=100:00:00
#SBATCH --partition=cluster_long
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 

# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

# Pythonのバージョンを指定 (pyenvでインストール済みのバージョン)
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"

PYTHON_SCRIPT_PATH_1='/work/riku-ka/patch_coverage_culculater/create_project_csvs_from_srcmap.py'
PYTHON_SCRIPT_PATH_2='/work/riku-ka/patch_coverage_culculater/revision_with_date.py'
PYTHON_SCRIPT_PATH_3='/work/riku-ka/patch_coverage_culculater/create_daily_diff.py'


# ログ用、エラー用ディレクトリ作成
mkdir -p logs errors


# 実行
echo "Pythonスクリプト1を実行します..."
${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_1}" 
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "エラー: Pythonスクリプト1の実行に失敗しました (終了コード: $PYTHON_EXIT_CODE)。"
    exit $PYTHON_EXIT_CODE
fi

echo "Pythonスクリプト1の実行が完了しました。"

echo "Pythonスクリプト2を実行します..."
${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_2}" 
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "エラー: Pythonスクリプト2の実行に失敗しました (終了コード: $PYTHON_EXIT_CODE)。"
    exit $PYTHON_EXIT_CODE
fi

echo "Pythonスクリプト2の実行が完了しました。"

echo "Pythonスクリプト3を実行します..."
${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_3}" 
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "エラー: Pythonスクリプト3の実行に失敗しました (終了コード: $PYTHON_EXIT_CODE)。"
    exit $PYTHON_EXIT_CODE
fi

echo "Pythonスクリプト3の実行が完了しました。"
