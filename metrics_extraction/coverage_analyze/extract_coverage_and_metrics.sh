#!/bin/bash
# Check if project.txt exists
if [[ ! -f projects.txt ]]; then
  echo "projects.txt not found!"
  exit 1
fi
target_count=$1


# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

# Pythonのバージョンを指定 (pyenvでインストール済みのバージョン)
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"


PYTHON_SCRIPT_PATH_1="/work/riku-ka/metrics_culculator/coverage_zip/decide_fuzzer.py"


BASE_PATH="/work/riku-ka/metrics_culculator/coverage_zip/output"

count=1
while IFS= read -r repo_identifier; do # projects.txt から読み込むのは識別子 (例: owner/repo)
  echo "--$count-----------------------"
  if [[ -n "$target_count" && $count -ne $target_count ]]; then # $target_count と一致しない場合はスキップ (ne に注意)
    echo "SKIP: $repo_identifier (Target: $target_count, Current: $count)"
    ((count++))
    continue
  fi

  echo "Processing repository identifier: $repo_identifier"
  
  # GitHub識別子からローカルのディレクトリ名を取得 (例: "owner/repo" -> "repo")
  local_repo_dir_name=$(basename "$repo_identifier")
  echo "Local repository directory name: $local_repo_dir_name"

  # 1番目のスクリプト
  echo "Running first Python script: ${PYTHON_SCRIPT_PATH_1}"
  if ! ${PYTHON_EXEC} "${PYTHON_SCRIPT_PATH_1}" "${local_repo_dir_name}" "${BASE_PATH}"; then
    echo "Failed to run first script for ${repo_identifier}. Skipping."
    continue
  fi


  ((count++)) # 処理が完了したら（成功・失敗問わず）カウントアップ
done < projects.txt