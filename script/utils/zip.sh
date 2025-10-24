#!/bin/bash
#SBATCH --job-name=metrics_extract
#SBATCH --output=logs/metrics_extract_%A_%a.out
#SBATCH --error=errors/metrics_extract_%A_%a.err
#SBATCH --array=1
#SBATCH --time=04:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=50
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL

# javaモジュールをロード
module load java/21

# Java用の一時ディレクトリを作成し、場所を指定する
# ${SLURM_JOB_ID} はSlurmが提供する変数で、ジョブごとに固有のIDが入ります。
# これにより、同時に複数のジョブを実行しても一時ディレクトリが衝突しません。
JAVA_TMP_DIR="$3/java_tmp_${SLURM_JOB_ID}" # $3は3番目の引数（出力先ディレクトリ）
mkdir -p "$JAVA_TMP_DIR"
export _JAVA_OPTIONS="-Djava.io.tmpdir=$JAVA_TMP_DIR"


# 使用方法: ./script_name.sh <親ディレクトリ> <zip化する数> [出力先ディレクトリ]
# 例: ./script_name.sh "/Users/yourname/Projects" 3 "/Users/yourname/Archives"

SOURCE_PARENT_DIR="$1"
NUM_TO_ZIP="$2"
OUTPUT_ZIP_DIR="$3" # オプション

# 引数のチェック (簡易版)
if [ -z "$SOURCE_PARENT_DIR" ] || [ -z "$NUM_TO_ZIP" ]; then
  echo "エラー: 親ディレクトリとzip化する数を指定してください。"
  echo "使用方法: $0 <親ディレクトリ> <zip化する数> [出力先ディレクトリ]"
  exit 1
fi

if ! [ "$NUM_TO_ZIP" -eq "$NUM_TO_ZIP" ] 2>/dev/null || [ "$NUM_TO_ZIP" -le 0 ]; then
  echo "エラー: zip化する数は正の整数で指定してください。"
  exit 1
fi

if [ ! -d "$SOURCE_PARENT_DIR" ]; then
  echo "エラー: 親ディレクトリ '$SOURCE_PARENT_DIR' が見つかりません。"
  exit 1
fi

# 出力先ディレクトリが指定されていなければ、親ディレクトリと同じ場所にする
if [ -z "$OUTPUT_ZIP_DIR" ]; then
  OUTPUT_ZIP_DIR="$SOURCE_PARENT_DIR"
fi

# 出力先ディレクトリが存在しなければ作成 (エラー時も考慮)
mkdir -p "$OUTPUT_ZIP_DIR" || { echo "エラー: 出力先ディレクトリ '$OUTPUT_ZIP_DIR' の作成に失敗しました。"; exit 1; }
OUTPUT_ZIP_DIR_ABS=$(cd "$OUTPUT_ZIP_DIR" && pwd) # 絶対パスに変換

echo "情報: 親ディレクトリ: $SOURCE_PARENT_DIR"
echo "情報: zip化するディレクトリ数 (名前順先頭から): $NUM_TO_ZIP"
echo "情報: 出力先: $OUTPUT_ZIP_DIR_ABS"
echo ""

# ディレクトリ一覧を取得し、ソートし、先頭N個を選択
# find ... -print0 | sort -z | head -z -n "$NUM_TO_ZIP" | xargs -0 -I {} ... の方が
# ファイル名に特殊文字が含まれる場合に堅牢ですが、ここでは可読性のため簡易な方法で示します。
# macOS の head に -z オプションはないかもしれません。
# find の結果を配列に入れるのが安全です。
target_dirs=()
while IFS= read -d $'\0' dir_name; do
  target_dirs+=("$dir_name")
done < <(find "$SOURCE_PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
# sort -z もGNU拡張の可能性あり。macOSなら `sort` のみで改行区切りで処理後、配列に入れるのが無難か。

# macOSでのfindとsortの安全な組み合わせ（ディレクトリ名に改行がない前提）
selected_dirs_str=$(find "$SOURCE_PARENT_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | head -n "$NUM_TO_ZIP")

if [ -z "$selected_dirs_str" ]; then
  echo "情報: zip化対象のディレクトリが見つかりませんでした。"
  exit 0
fi

echo "以下のディレクトリを個別にzip化します:"
echo "$selected_dirs_str"
echo ""

# 元のディレクトリを記憶し、処理対象の親ディレクトリに移動
# これにより、jarコマンドのパス指定が簡潔になる
ORIGINAL_PWD=$(pwd)
cd "$SOURCE_PARENT_DIR" || { echo "エラー: 親ディレクトリ '$SOURCE_PARENT_DIR' への移動に失敗しました。"; exit 1; }

# 選択された各ディレクトリをzip化
echo "$selected_dirs_str" | while IFS= read -r dir_name; do
  if [ -n "$dir_name" ]; then # 空行でないことを確認
    echo "処理中: '$dir_name' をzip化しています..."
    # jar -c (create) -f (file) でzipファイルを作成
    # 出力先のパスは絶対パスで指定
    if jar -cf "$OUTPUT_ZIP_DIR_ABS/$dir_name.zip" "$dir_name"; then
      echo "成功: '$OUTPUT_ZIP_DIR_ABS/$dir_name.zip' が作成されました。"
    else
      echo "エラー: '$dir_name' のzip化に失敗しました。"
    fi
  fi
done

# 元のディレクトリに戻る
cd "$ORIGINAL_PWD"

# 作成した一時ディレクトリを削除して後片付け
rm -rf "$JAVA_TMP_DIR"

echo ""
echo "処理が完了しました。"