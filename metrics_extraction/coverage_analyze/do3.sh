#!/bin/bash

#SBATCH --job-name=coverage_oss_fuzz
#SBATCH --output=logs/coverage_oss_fuzz_%A_%a.out
#SBATCH --error=errors/coverage_oss_fuzz_%A_%a.err
#SBATCH --array=1
#SBATCH --time=1000:00:00
#SBATCH --partition=cluster_low
#SBATCH --ntasks=1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=52
#SBATCH --mail-user=kato.riku.ks5@naist.ac.jp
#SBATCH --mail-type=END,FAIL 


# メモリ最低限を100GB以上に設定する必要あり
mkdir -p errors
mkdir -p logs

# スクリプトの絶対パス／ディレクトリ
SCRIPT_DIR="/work/riku-ka/metrics_culculator/coverage_zip"

# カウント対象ディレクトリ
zip_dir="/work/riku-ka/metrics_culculator/coverage_zip/ossfuzz_downloaded_coverage_20250802_zip"

# 出力するリストファイル（スクリプトと同じディレクトリ）
LIST_FILE="$SCRIPT_DIR/list_of_zip_files_fullpath_project.txt"

# ZIP の一覧を生成（zip_dir からの相対パスを出力し、先頭に ./ を付与）
# サブディレクトリも含める場合は -maxdepth を使わない（ここでは再帰的に取得）
# GNU find の %P を利用して zip_dir からの相対パスを出力
find "$zip_dir" -type f -name '*.zip' -printf '%P\n' | sed 's|^|./|' > "$LIST_FILE"

# 行数をカウント（存在しない/空ファイルは 0）
if [ -f "$LIST_FILE" ]; then
  count=$(wc -l < "$LIST_FILE" | tr -d ' ')
else
  count=0
fi

echo "Found $count zip files in $zip_dir"
echo "Wrote list to $LIST_FILE"

# 生成した件数とディレクトリを渡して処理を開始
bash "$SCRIPT_DIR/all_list_do_project.sh" "$count" "$zip_dir"
