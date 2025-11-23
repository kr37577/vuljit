#!/usr/bin/env bash
set -euo pipefail

vuljit_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXEC="${PYTHON_EXEC:-python3}"  # 必要に応じて上書き

# oss-fuzz リポジトリの clone/fetch/checkout
repo_dir="${vuljit_dir}/datasets/raw/oss-fuzz"
repo_url="https://github.com/google/oss-fuzz.git"
repo_commit="96cf5fd552e90b1879d5d3c9bf6d9cdb95e6f122"

mkdir -p "$(dirname "${repo_dir}")"
if [[ -d "${repo_dir}/.git" ]]; then
  echo "[info] oss-fuzz already cloned; fetching and checking out ${repo_commit}"
  git -C "${repo_dir}" fetch --all --prune
else
  echo "[info] cloning oss-fuzz into ${repo_dir}"
  git clone "${repo_url}" "${repo_dir}"
fi
git -C "${repo_dir}" checkout "${repo_commit}"

# メタデータ抽出
PYTHON_SCRIPT_PATH_1="${vuljit_dir}/scripts/project_mapping/oss_fuzz_project_info.py"
"${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}"

# プロジェクトごとの clone/処理（runner/テンプレートは要相談）
bash "${vuljit_dir}/scripts/project_mapping/clone_and_run_projects.sh"
