#!/bin/bash
# テストようにproject.txtをproject_test.txtにしている

# Check if projects_test.txt exists
if [[ ! -f projects.txt ]]; then
  echo "projects.txt not found!"
  exit 1
fi
target_count=$1
# Create a directory to clone repositories
mkdir -p cloned_repos


# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

# Pythonのバージョンを指定 (pyenvでインストール済みのバージョン)
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"


PYTHON_SCRIPT_PATH_1="text_code_metrics/repo_commit_processor_test.py"
# PYTHON_SCRIPT_PATH_2="text_code_metrics/text_metrics_calculator.py"
# PYTHON_SCRIPT_PATH_3="text_code_metrics/text_raw_convert_metrics.py"
PYTHON_SCRIPT_VULN_LABELER="text_code_metrics/label.py" 

count=1
while IFS= read -r repo_identifier; do # projects.txt から読み込むのは GitHub上の識別子 (例: owner/repo)
  echo "--$count-----------------------"
  if [[ -n "$target_count" && $count -ne $target_count ]]; then # $target_count と一致しない場合はスキップ (ne に注意)
    echo "SKIP: $repo_identifier (Target: $target_count, Current: $count)"
    ((count++))
    continue
  fi

  echo "Processing repository identifier: $repo_identifier"
  
  # GitHub識別子からローカルのディレクトリ名を取得 (例: "owner/repo" -> "repo")
  local_repo_dir_name=$(basename "$repo_identifier")
  # クローン先のローカルリポジトリパス
  local_repo_actual_path="cloned_repos/${local_repo_dir_name}"

  # カレントディレクトリは ~/metrics_culculator/ のはず
  # pwd

  # リポジトリのクローン処理
  if [[ -d "$local_repo_actual_path" ]]; then
    echo "  Repository already exists at $local_repo_actual_path"
    # 必要であればここで git pull などを行う
  else
    mkdir -p cloned_repos # cloned_repos ディレクトリがなければ作成
    echo "  Cloning https://github.com/${repo_identifier} into ${local_repo_actual_path}"
    if ! git clone "https://github.com/${repo_identifier}" "$local_repo_actual_path"; then
      echo "  Failed to clone ${repo_identifier}. Skipping."
      ((count++))
      continue
    fi
  fi

  # --- 1番目のスクリプト (repo_commit_processor_test.py) の呼び出し ---
  # SCRIPT_1 の出力先ディレクトリ (リポジトリごとのサブディレクトリを output 直下に作成)
  output_dir_script_1="output_0802/${local_repo_dir_name}"
  mkdir -p "$output_dir_script_1" # 出力ディレクトリを確実に作成

  echo "  Running SCRIPT_1 (repo_commit_processor) for local path: ${local_repo_actual_path}"
  echo "    Output directory for SCRIPT_1: ${output_dir_script_1}"
 
  # SCRIPT_1 を呼び出し。引数としてローカルリポジトリのパスと出力ディレクトリを渡す
  if ! "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_1}" "${local_repo_actual_path}" -o "${output_dir_script_1}"; then
    echo "  Error running SCRIPT_1 for ${repo_identifier}. Skipping further processing for this repo."
    ((count++))
    continue
  fi
  
  
  # SCRIPT_1 が生成する中間CSVファイルのフルパス (SCRIPT_1内のファイル名生成ロジックに合わせる)
  # SCRIPT_1 の修正案では、渡されたリポジトリパスの basename を使うので local_repo_dir_name を使用
  intermediate_csv_path="${output_dir_script_1}/${local_repo_dir_name}_commit_metrics_output_per_commit.csv"

  # 中間CSVが実際に生成されたか確認
  if [[ ! -f "${intermediate_csv_path}" ]]; then
    echo "  Error: SCRIPT_1 did not produce the expected CSV: ${intermediate_csv_path}. Skipping SCRIPT_2."
    ((count++))
    continue
  fi


# --- 変更したPythonスクリプト (Vulnerability Labeler) の呼び出し ---
  echo "  Vulnerability Labeler (SCRIPT_VULN_LABELER) をパッケージ: ${local_repo_dir_name} で実行中"
  echo "    入力メトリクスCSV: ${intermediate_csv_path}"
  echo "    出力ディレクトリ: ${output_dir_script_1}"

  # 変更したPythonスクリプトを呼び出す
  # 引数: package_name, metrics_file_path, -o output_dir
  if ! "${PYTHON_EXEC}" "${PYTHON_SCRIPT_VULN_LABELER}" "${local_repo_dir_name}" "${intermediate_csv_path}" -o "${output_dir_script_1}"; then
    echo "  ${local_repo_dir_name} のVulnerability Labeler実行中にエラーが発生しました。SCRIPT_2をスキップします。"
    ((count++))
    continue # または必要に応じてエラー処理
  fi

  # Vulnerability Labeler によって生成されるCSVファイルへのパス
  labeled_csv_path="${output_dir_script_1}/${local_repo_dir_name}_commit_metrics_with_vulnerability_label.csv"

  if [[ ! -f "${labeled_csv_path}" ]]; then
    echo "  エラー: Vulnerability Labeler が期待されるラベル付きCSV (${labeled_csv_path}) を生成しませんでした。SCRIPT_2をスキップします。"
    ((count++))
    continue
  fi


  # --- 2番目のスクリプト (text_metrics_calculator.py) の呼び出し ---
  # SCRIPT_2 の出力先ディレクトリ (SCRIPT_1 と同じ場所に出力する場合)
  output_dir_script_2="${output_dir_script_1}"
  # mkdir -p "$output_dir_script_2" # output_dir_script_1 と同じなら不要

  echo "  Running SCRIPT_2 (text_metrics_calculator) using input CSV: ${intermediate_csv_path}"
  echo "    Output directory for SCRIPT_2: ${output_dir_script_2}"

  # SCRIPT_2 を呼び出し。引数として中間CSVのパスと出力ディレクトリを渡す
  if ! "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_2}" "${intermediate_csv_path}" "${local_repo_dir_name}"  -o "${output_dir_script_2}"; then
    echo "  Error running SCRIPT_2 for ${repo_identifier}."
    # エラーがあっても次のリポジトリに進むために count はインクリメント
  fi

  # # --- 3番目のスクリプト (text_raw_convert_metrics.py) の呼び出し ---
  echo "  Running SCRIPT_3 (text_raw_convert_metrics) using input CSV: ${script2_output_path}"
  
  # SCRIPT_3 を呼び出し。引数として SCRIPT_2 の出力ファイルを渡す
  # 出力ファイルは自動生成されるので、-o は指定しない
  if ! "${PYTHON_EXEC}" "${PYTHON_SCRIPT_PATH_3}" "${script2_output_path}"; then
    echo "  Error running SCRIPT_3 for ${repo_identifier}."
  fi

  ((count++)) # 処理が完了したら（成功・失敗問わず）カウントアップ
done < projects.txt


echo "All specified repositories processed."
