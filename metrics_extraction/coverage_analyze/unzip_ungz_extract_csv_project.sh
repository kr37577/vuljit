#!/bin/bash

# スクリプトの引数が指定されているかチェック
if [ -z "$1" ]; then
  echo "使用法: $0 <解凍するZIPファイル>"
  exit 1
fi

ZIP_FILE="$1"

# ZIPファイルが存在するかチェック
if [ ! -f "$ZIP_FILE" ]; then
  echo "エラー: ファイル '$ZIP_FILE' が見つかりません。"
  exit 1
fi

# 解凍先のディレクトリ名を作成 (例: archive.zip -> archive)
DEST_DIR="${ZIP_FILE%.zip}"

# スクリプトの絶対パスを取得
SCRIPT_PATH_UNZIP=$(realpath "$0")
# スクリプトが配置されているディレクトリを取得
SCRIPT_DIR_UNZIP=$(dirname "$SCRIPT_PATH_UNZIP")

# python パス
# --- Python環境の準備 (pyenv を使用) ---
echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"

# Pythonのバージョンを指定 (pyenvでインストール済みのバージョン)
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"
# test_process_coverage.py のパスをスクリプトの場所を基準に設定
PYTHON_SCRIPT_TO_RUN="$SCRIPT_DIR_UNZIP/process_coverage_project.py"


# 解凍先ディレクトリを作成 (既に存在する場合は何もしない)
mkdir -p "$DEST_DIR"

# ZIPファイルを指定ディレクトリに解凍
echo "'$ZIP_FILE' を '$DEST_DIR/' へ解凍しています..."
unzip "$ZIP_FILE" -d "$DEST_DIR"

# unzipコマンドの終了ステータスをチェック
if [ $? -eq 0 ]; then
  echo "解凍が完了しました: $DEST_DIR/"
else
  echo "エラー: 解凍中に問題が発生しました。"
  exit 1
fi


# 解凍先ディレクトリ内の .gz ファイルを再帰的に検索して解凍
echo "'$DEST_DIR' 内の .gz ファイルを解凍しています..."
find "$DEST_DIR" -type f -name "*.gz" -exec gunzip {} \;
# gunzipコマンドの終了ステータスをチェック
if [ $? -eq 0 ]; then
  echo ".gz ファイルの解凍が完了しました。"
else
  echo "エラー: .gz ファイルの解凍中に問題が発生しました。"
  exit 1
fi


# pythonスクリプトを実行してCSVファイルを処理
echo "Pythonスクリプトを実行: ${PYTHON_SCRIPT_TO_RUN} $DEST_DIR"
"$PYTHON_EXEC" "$PYTHON_SCRIPT_TO_RUN" "$DEST_DIR"
if [ $? -eq 0 ]; then
  echo "CSVファイルの処理が完了しました。"
else
  echo "エラー: CSVファイルの処理中に問題が発生しました。"
  exit 1
fi




# 解凍先のディレクトリを削除
echo "解凍先ディレクトリ '$DEST_DIR' を削除しています..."
rm -rf "$DEST_DIR"
if [ $? -eq 0 ]; then
  echo "解凍先ディレクトリの削除が完了しました。"
else
  echo "エラー: 解凍先ディレクトリの削除中に問題が発生しました。"
  exit 1
fi


# 処理が成功した場合は終了
echo "すべての処理が正常に完了しました。"

exit 0