#!/bin/bash
#SBATCH --job-name=single_zip                # ジョブの名前
#SBATCH --output=logs/zip_%j.out             # 標準出力ログの保存先 (%j はジョブIDに置き換わります)
#SBATCH --error=errors/zip_%j.err            # 標準エラーログの保存先
#SBATCH --time=24:00:00                      # 最大実行時間 (24時間)
#SBATCH --partition=cluster_low              # ジョブを投入するパーティション (環境に合わせて変更してください)
#SBATCH --ntasks=1                           # 使用するタスク数
#SBATCH --mem=150G                             # 要求するメモリ量 
#SBATCH --cpus-per-task=50                    # 1タスクあたりのCPUコア数
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL

# 事前に logs と errors ディレクトリを作成しておいてください
# mkdir -p logs errors

# --- 使用方法 ---
# sbatch single_zip.sh <圧縮したいディレクトリのパス> <ZIPファイルの出力先ディレクトリ>
#
# --- 実行例 ---
# sbatch single_zip.sh /path/to/my_project /path/to/archives

# javaモジュールをロード
module load java/21

# --- 引数のチェック ---
if [ "$#" -ne 2 ]; then
  echo "エラー: 2つの引数（圧縮元ディレクトリと出力先ディレクトリ）を必ず指定してください。"
  echo "使用方法: sbatch $0 <圧縮元ディレクトリ> <出力先ディレクトリ>"
  exit 1
fi

SOURCE_DIR="$1"
OUTPUT_DIR="$2"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "エラー: 圧縮元ディレクトリ '$SOURCE_DIR' が見つかりません。"
  exit 1
fi

# Java用の一時ディレクトリを作成し、場所を指定する
# ${SLURM_JOB_ID} はSlurmが提供する変数で、ジョブごとに固有のIDが入ります。
JAVA_TMP_DIR="/tmp/java_tmp_${SLURM_JOB_ID}"
mkdir -p "$JAVA_TMP_DIR"
export _JAVA_OPTIONS="-Djava.io.tmpdir=$JAVA_TMP_DIR"

# --- 準備 ---
# 出力先ディレクトリが存在しなければ作成
mkdir -p "$OUTPUT_DIR" || { echo "エラー: 出力先ディレクトリ '$OUTPUT_DIR' の作成に失敗しました。"; exit 1; }

# 各パスを絶対パスに変換して、どこから実行しても安全なようにする
SOURCE_DIR_ABS=$(cd "$SOURCE_DIR" && pwd)
OUTPUT_DIR_ABS=$(cd "$OUTPUT_DIR" && pwd)

# 圧縮元ディレクトリの親ディレクトリと、圧縮対象のディレクトリ名を取得
SOURCE_PARENT_DIR=$(dirname "$SOURCE_DIR_ABS")
SOURCE_BASENAME=$(basename "$SOURCE_DIR_ABS")

# 出力するZIPファイル名を決定
OUTPUT_ZIP_FILE="$OUTPUT_DIR_ABS/$SOURCE_BASENAME.zip"

echo "--- 処理開始 ---"
echo "圧縮元: $SOURCE_DIR_ABS"
echo "出力先: $OUTPUT_ZIP_FILE"
echo "----------------"

# --- ZIP圧縮の実行 (jarを使用) ---
# -c: 新規アーカイブを作成
# -f: 作成するファイル名を指定
echo "処理中: '$SOURCE_BASENAME' をZIP化しています..."

# 親ディレクトリに移動してからjarコマンドを実行することで、zip内のパス構造をクリーンにする
# ( ) で囲むことでサブシェルで実行され、cdの影響がこの行の中だけに限定される
if (cd "$SOURCE_PARENT_DIR" && jar -cf "$OUTPUT_ZIP_FILE" "$SOURCE_BASENAME"); then
  echo "成功: '$OUTPUT_ZIP_FILE' が作成されました。"
else
  echo "エラー: '$SOURCE_BASENAME' のZIP化に失敗しました。"
  rm -rf "$JAVA_TMP_DIR" # エラー時も一時ファイルを削除
  exit 1 # エラーで終了
fi

# 作成した一時ディレクトリを削除して後片付け
rm -rf "$JAVA_TMP_DIR"

echo "--- 処理完了 ---"