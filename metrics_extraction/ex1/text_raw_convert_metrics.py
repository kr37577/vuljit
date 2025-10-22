import pandas as pd
import json
import sys
import os
import argparse


def expand_term_frequencies(input_filepath, output_filepath):
    """
    CSVファイルを読み込み、そのJSON列を展開して新しいCSVファイルに保存します。

    Args:
        input_filepath (str): 入力CSVファイルのパス。
        output_filepath (str): 結果を保存する出力CSVファイルのパス。
    """
    print(f"'{input_filepath}' を読み込んでいます...")
    try:
        df = pd.read_csv(input_filepath)
        print(f"読み込み完了。{len(df)}件のレコードを処理します。")
    except FileNotFoundError:
        print(f"エラー: ファイル '{input_filepath}' が見つかりません。")
        return
    except Exception as e:
        print(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
        return

    processed_records = []
    total_rows = len(df)
    for index, row in df.iterrows():
        # 処理の進捗を表示
        if (index + 1) % 200 == 0 or index + 1 == total_rows:
            print(f"レコード {index + 1}/{total_rows} を処理中...")

        record = {'commit_hash': row.get('commit_hash', '')}

        # 'files_term_freq' を展開
        try:
            files_freq_str = row.get('files_term_freq')
            if isinstance(files_freq_str, str) and files_freq_str.strip():
                files_freq = json.loads(files_freq_str)
                for term, count in files_freq.items():
                    # 不適切な文字を列名から除外
                    clean_term = "".join(
                        c if c.isalnum() else '_' for c in term)
                    record[f'text_{clean_term}_file'] = count
        except (json.JSONDecodeError, TypeError):
            pass  # JSONが空、または不正な形式の場合はスキップ

        # 'patch_term_diff' を展開
        try:
            patch_diff_str = row.get('patch_term_diff')
            if isinstance(patch_diff_str, str) and patch_diff_str.strip():
                # JSON文字列内の二重引用符問題を修正
                if patch_diff_str.startswith('"') and patch_diff_str.endswith('"'):
                    patch_diff_str = patch_diff_str[1:-1].replace('""', '"')

                patch_diff = json.loads(patch_diff_str)
                for term, diff in patch_diff.items():
                    clean_term = "".join(
                        c if c.isalnum() else '_' for c in term)
                    record[f'text_{clean_term}_patch'] = diff
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass  # JSONが空、または不正な形式の場合はスキップ

        processed_records.append(record)

    if not processed_records:
        print("処理対象のデータがありませんでした。")
        return

    print("データフレームを生成しています...")
    expanded_df = pd.DataFrame(processed_records).fillna(0)

    # commit_hash以外の列を整数型に変換
    for col in expanded_df.columns:
        if col != 'commit_hash':
            expanded_df[col] = expanded_df[col].astype(int)

    try:
        expanded_df.to_csv(output_filepath, index=False)
        print("\n処理が完了しました。")
        print(f"結果は '{output_filepath}' に保存されました。")
        print(f"新しいテーブルは {len(expanded_df)} 行、{len(expanded_df.columns)} 列です。")
    except Exception as e:
        print(f"出力ファイルの保存中にエラーが発生しました: {e}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='CSVファイルのJSON列を展開して新しいCSVファイルに保存します。') 
    
    
    parser.add_argument('input_file', type=str, help='入力CSVファイルのパス')
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help='出力CSVファイルのパス（省略時は入力ファイル名に "_expanded" を付加）')
    
    args = parser.parse_args()
    
    input_filepath = args.input_file
    output_filepath = args.output_file
    
    # 出力ファイルパスが指定されていない場合の処理
    if not output_filepath:
        # 入力ファイルのディレクトリとベース名を取得
        input_dir = os.path.dirname(input_filepath)
        base_name = os.path.basename(input_filepath)
        # 拡張子を除いたファイル名を取得
        name_without_ext = os.path.splitext(base_name)[0]
        # 新しい出力ファイル名を生成
        output_filename = f"{name_without_ext}_expanded.csv"
        output_filepath = os.path.join(input_dir, output_filename)

    expand_term_frequencies(input_filepath, output_filepath)