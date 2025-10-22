#!/bin/bash

# スクリプトの引数チェックを更新
if [ -z "$1" ] || ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ] || [ -z "$2" ]; then
  echo "使用法: $0 <処理するファイル数 (正の整数)> <ZIPファイルのあるディレクトリのパス>"
  echo "例: $0 10 /path/to/zip_files"
  exit 1
fi

NUM_TO_PROCESS=$1
ZIP_FILES_DIR="$2" # ZIPファイルが格納されているディレクトリ

# スクリプトの絶対パスを取得
SCRIPT_PATH=$(realpath "$0")
# スクリプトが配置されているディレクトリを取得
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

FILE_LIST_PATH="$SCRIPT_DIR/list_of_zip_files_fullpath_project.txt"
UNZIP_SCRIPT_PATH="$SCRIPT_DIR/unzip_ungz_extract_csv_project.sh"

# リストファイルが存在するかチェック
if [ ! -f "$FILE_LIST_PATH" ]; then
  echo "エラー: リストファイル '$FILE_LIST_PATH' が見つかりません。"
  exit 1
fi
# unzipスクリプトが存在するかチェック
if [ ! -f "$UNZIP_SCRIPT_PATH" ]; then
  echo "エラー: unzipスクリプト '$UNZIP_SCRIPT_PATH' が見つからない"
  exit 1
fi
# ファイル全体を配列に読み込む
ORIGINAL_FILES=()
while IFS= read -r line || [[ -n "$line" ]]; do
    ORIGINAL_FILES+=("$line")
done < "$FILE_LIST_PATH"

FILES_ATTEMPTED_THIS_RUN=0
SUCCESSFULLY_REMOVED_COUNT=0
NEW_FILES_LIST=()

echo "処理を開始します。対象ファイル数: $NUM_TO_PROCESS"
echo "ZIPファイルディレクトリ: $ZIP_FILES_DIR"

for i in "${!ORIGINAL_FILES[@]}"; do
    relative_zip_file_path="${ORIGINAL_FILES[$i]}" # list_of_zip_files_fullpath.txt から読み込んだパス

    # 空行や不正な行をスキップ
    if [ -z "$relative_zip_file_path" ]; then
        NEW_FILES_LIST+=("$relative_zip_file_path")
        continue
    fi

    # ZIPファイルのフルパスを組み立てる
    # list_of_zip_files_fullpath.txt のパスが "./file.zip" のような形式の場合の対応
    if [[ "$relative_zip_file_path" == "./"* ]]; then
        # "./" を削除して結合
        # ZIP_FILES_DIR が "." の場合、 "file.zip" のようになる
        # ZIP_FILES_DIR が "/path/to" の場合、 "/path/to/file.zip" のようになる
        zip_file_to_process="$ZIP_FILES_DIR/${relative_zip_file_path:2}"
    else
        # そのまま結合 (例: "file.zip" or "subdir/file.zip")
        zip_file_to_process="$ZIP_FILES_DIR/$relative_zip_file_path"
    fi
    # 以下の行はディレクトリ構造を破壊するため削除またはコメントアウト
    # actual_zip_file_name=$(basename "$relative_zip_file_path")
    # zip_file_to_process="$ZIP_FILES_DIR/$actual_zip_file_name"
    echo "処理対象のZIPファイル: $zip_file_to_process (元リストエントリ: $relative_zip_file_path)"
    # ただし、list_of_zip_files_fullpath.txt の内容が "./" で始まっているため、
    # ZIP_FILES_DIR とそのまま結合すると "/path/to/zip_files/./file.zip" のようになる。
    # realpath で正規化するか、文字列操作で "./" を除去する。
    # ここでは、ZIP_FILES_DIR の中で relative_zip_file_path が解決されると仮定し、
    # カレントディレクトリを変更する方法も考えられるが、パス結合がより安全。

    # # list_of_zip_files_fullpath.txt のパスが "./file.zip" のような形式の場合の対応
    # if [[ "$relative_zip_file_path" == "./"* ]]; then
    #     # "./" を削除して結合
    #     zip_file_to_process="$ZIP_FILES_DIR/${relative_zip_file_path:2}"
    # else
    #     # そのまま結合 (例: "file.zip" or "subdir/file.zip")
    #     zip_file_to_process="$ZIP_FILES_DIR/$relative_zip_file_path"
    # fi


    if [ "$FILES_ATTEMPTED_THIS_RUN" -lt "$NUM_TO_PROCESS" ]; then
        FILES_ATTEMPTED_THIS_RUN=$((FILES_ATTEMPTED_THIS_RUN + 1))
        echo "--------------------------------------------------"
        echo "処理中 ($FILES_ATTEMPTED_THIS_RUN/$NUM_TO_PROCESS): $zip_file_to_process (元リストエントリ: $relative_zip_file_path)"

        # unzip_ungz_extract_csv.sh を実行。ZIPファイルのフルパスを渡す
        bash "$UNZIP_SCRIPT_PATH" "$zip_file_to_process"
        exit_status=$?

        if [ $exit_status -eq 0 ]; then
            echo "処理成功: $zip_file_to_process"
            # ZIPファイルを削除
            if [ -f "$zip_file_to_process" ]; then
                rm -f "$zip_file_to_process"
                if [ $? -eq 0 ]; then
                    echo "ファイル削除成功: $zip_file_to_process"
                    SUCCESSFULLY_REMOVED_COUNT=$((SUCCESSFULLY_REMOVED_COUNT + 1))
                    # 成功したので NEW_FILES_LIST には追加しない (元の相対パスエントリを)
                else
                    echo "警告: ファイル削除失敗: $zip_file_to_process。リストには残します。"
                    NEW_FILES_LIST+=("$relative_zip_file_path") # 元のリストエントリを保持
                fi
            else
                echo "警告: 削除対象のファイル '$zip_file_to_process' が見つかりませんでしたが、処理は成功として扱います。"
                SUCCESSFULLY_REMOVED_COUNT=$((SUCCESSFULLY_REMOVED_COUNT + 1))
            fi
        else
            echo "処理失敗: $zip_file_to_process (終了コード: $exit_status)。リストには残します。"
            NEW_FILES_LIST+=("$relative_zip_file_path") # 元のリストエントリを保持
        fi
    else
        # 処理上限に達したので、残りのファイルはそのまま新しいリストに追加
        NEW_FILES_LIST+=("$relative_zip_file_path") # 元のリストエントリを保持
    fi
done


# 新しいリストの内容をファイルに書き出す
# 配列が空の場合でもファイルは空になる（期待通り）
printf "%s\n" "${NEW_FILES_LIST[@]}" > "$FILE_LIST_PATH"

echo "--------------------------------------------------"
echo "処理が完了しました。"
echo "合計 $SUCCESSFULLY_REMOVED_COUNT 個のZIPファイルを処理し、リストから削除しました。"
echo "$FILES_ATTEMPTED_THIS_RUN 個のファイルを処理しようとしました。"

exit 0
