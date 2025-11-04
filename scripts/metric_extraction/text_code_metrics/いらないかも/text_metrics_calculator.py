import re
import re
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
from get_feature_commit_func import CODE_FILE_EXTENSIONS
from typing import Optional
import json
import pickle
import ast
import sys
import argparse
# -*- coding: utf-8 -*-

# --- グローバル定数 ---
# デフォルトのトークンパターン
DEFAULT_TOKEN_PATTERN = r"(?u)\b[a-zA-Z]\w+\b"

# 論文で述べられているフィルタリング閾値
MIN_DF_PAPER = 0.05  # ドキュメントの5%未満に出現する単語を除去
MAX_DF_PAPER = 0.80  # ドキュメントの80%以上に出現する単語を除去

# ロガー設定 (必要に応じて詳細設定)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_cleaned_file_contents(files_content_map: Dict[str, str]) -> List[str]:
    """
    ファイル内容の辞書から，実際のコンテンツ文字列のリストを抽出する
    プレースホルダーやエラーメッセージは除外する
    """
    contents = []
    if not files_content_map:
        logging.warning("No file contents provided.")
        return contents

    for file_path, content in files_content_map.items():
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in CODE_FILE_EXTENSIONS:
            continue

        if content and content != "[DELETED IN THIS COMMIT]" and \
           not content.startswith("ファイルの読み取りエラー") and \
           content != "[NO CONTENT BLOB (e.g., submodule)]":
            contents.append(content)
    return contents


def parse_patch_to_added_deleted_strings(patch_text: List[str]) -> Tuple[str, str]:
    """
    パッチテキストを解析し，追加された行全体と削除された行全体をそれぞれ1つの文字列として返す
    """
    added_lines_list = []
    deleted_lines_list = []

    if patch_text:
        for line in patch_text:
            if line.startswith('+') and not line.startswith('+++'):
                added_lines_list.append(line[1:].strip())

            elif line.startswith('-') and not line.startswith('---'):
                deleted_lines_list.append(line[1:].strip())

    added_string = " ".join(added_lines_list)
    deleted_string = " ".join(deleted_lines_list)
    return added_string, deleted_string


def learn_global_vocabulary(corpus: List[str], token_pattern: str, min_df: float, max_df: float) -> Optional[CountVectorizer]:
    """
    与えられたコーパス全体から，フィルタリングを適用したCountVectorizer（語彙学習済み）を返す．
    """

    if not corpus:
        logging.warning("Empty corpus provided for vocabulary learning.")
        return None

    try:
        vectorizer = CountVectorizer(
            input='content',
            decode_error='ignore',
            token_pattern=token_pattern,
            min_df=min_df,
            max_df=max_df
        )
        vectorizer.fit(corpus)
        if not vectorizer.vocabulary_:
            logging.warning(
                f"フィルタリングの結果、語彙が空になりました。min_df={min_df}, max_df={max_df}")
            return None
        logging.info(f"グローバル語彙学習完了。語彙数: {len(vectorizer.vocabulary_)}")
        return vectorizer
    except ValueError as e:
        logging.error(f"グローバル語彙学習中にValueError: {e}")
        return None


def get_term_frequencies_from_vectorizer(texts: List[str], vectorizer: CountVectorizer) -> Dict[str, int]:
    """
    学習済みのCountVectorizerを使って，与えられたテキスト群から単語頻度を計算する．
    (Files Term Frequency用: 複数のファイル内容を結合した総頻度)
    """
    if not texts or not vectorizer or not vectorizer.vocabulary_:
        return {}
    try:
        td_matrix = vectorizer.transform(texts)
        total_term_counts = td_matrix.sum(axis=0).A1

        terms = vectorizer.get_feature_names_out()

        # フィルタリングで語彙が極端に少ない、またはtd_matrixが空の場合の対応
        if total_term_counts.shape[0] == 0 or terms.shape[0] == 0:
            return {}

        # termsとtotal_term_countsの長さが一致することを確認
        if len(terms) != len(total_term_counts):
            logging.warning(
                f"用語リストとカウントの長さが不一致。terms: {len(terms)}, counts: {len(total_term_counts)}")
            # この場合、エラーとするか、可能な範囲で処理を試みるか。ここでは空辞書を返す。
            return {}

        # NumPyの数値型をPythonのint型に変換
        return {term: int(count) for term, count in zip(terms, total_term_counts)}
    except Exception as e:
        logging.error(f"単語頻度計算中にエラー: {e}")
        return {}


def get_patch_diff_term_frequencies_from_vectorizer(
    added_text: str,
    deleted_text: str,
    vectorizer: CountVectorizer
) -> Dict[str, int]:
    """
    学習済みのCountVectorizerを使って，追加行と削除行の単語頻度の差分を計算する．
    """
    if not vectorizer or not vectorizer.vocabulary_:
        return {}

    feature_names = vectorizer.get_feature_names_out()
    if feature_names.shape[0] == 0:  # 語彙が空なら処理終了
        return {}

    counts_added = pd.Series(0, index=feature_names, dtype=int)
    if added_text:
        try:
            matrix_added = vectorizer.transform([added_text])
            temp_counts_added = pd.Series(
                matrix_added.toarray().flatten(), index=feature_names, dtype=int)
            counts_added = counts_added.add(temp_counts_added, fill_value=0)
        except Exception as e:
            logging.warning(f"パッチ追加部分のベクトル化中にエラー: {e}")

    counts_deleted = pd.Series(0, index=feature_names, dtype=int)
    if deleted_text:
        try:
            matrix_deleted = vectorizer.transform([deleted_text])
            temp_counts_deleted = pd.Series(
                matrix_deleted.toarray().flatten(), index=feature_names, dtype=int)
            counts_deleted = counts_deleted.add(
                temp_counts_deleted, fill_value=0)
        except Exception as e:
            logging.warning(f"パッチ削除部分のベクトル化中にエラー: {e}")

    diff_counts = (counts_added - counts_deleted).abs()
    # .to_dict() の結果の値をPythonのint型に変換
    return {k: int(v) for k, v in diff_counts[diff_counts > 0].to_dict().items()}


# --- メイン処理関数 ---
def calculate_text_metrics_for_all_commits(
    all_commits_data: List[Dict[str, Any]],
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
    min_df_files: float = MIN_DF_PAPER,
    max_df_files: float = MAX_DF_PAPER,
    min_df_patches: float = MIN_DF_PAPER,
    max_df_patches: float = MAX_DF_PAPER
) -> List[Dict[str, Any]]:
    """
    全コミットデータに対して，論文仕様のテキストメトリクスを計算する．

    Args:
        all_commits_data: コミットデータのリスト。各要素は辞書で、
                          'commit_hash', 'files_content_map', 'patch_text' をキーとして含む想定。
        token_pattern: 単語分割用の正規表現パターン。
        min_df_files, max_df_files: Files Term Frequency 用のフィルタリング閾値。
        min_df_patches, max_df_patches: Patches Term Frequency 用のフィルタリング閾値。

    Returns:
        各コミットのテキストメトリクスを含む辞書のリスト。
        各要素は 'commit_hash', 'files_term_freq', 'patch_term_diff' をキーとして持つ。
    """
    results = []

    # --- 1. Files Term Frequency のためのグローバル語彙学習 ---
    logging.info("Files Term Frequency のためのグローバル語彙学習を開始...")
    all_files_corpus: List[str] = []
    for commit_data in all_commits_data:
        files_content_map = commit_data.get('files_content_map', {})
        # 各ファイルが1ドキュメントとしてコーパスに追加される
        all_files_corpus.extend(get_cleaned_file_contents(files_content_map))

    global_files_vectorizer = learn_global_vocabulary(
        all_files_corpus, token_pattern, min_df_files, max_df_files)

    # --- 2. Patches Term Frequency のためのグローバル語彙学習 ---
    logging.info("Patches Term Frequency のためのグローバル語彙学習を開始...")
    all_patch_docs_corpus: List[str] = []  # 追加行と削除行をそれぞれドキュメントとして扱う
    for commit_data in all_commits_data:
        patch_text = commit_data.get('filtered_patch_lines_list')

        added_str, deleted_str = parse_patch_to_added_deleted_strings(
            patch_text)
        if added_str:
            all_patch_docs_corpus.append(added_str)
        if deleted_str:
            all_patch_docs_corpus.append(deleted_str)

    global_patch_vectorizer = learn_global_vocabulary(
        all_patch_docs_corpus, token_pattern, min_df_patches, max_df_patches)

    # --- 3. 各コミットのメトリクス計算 ---
    logging.info("各コミットのテキストメトリクス計算を開始...")
    for i, commit_data in enumerate(all_commits_data):
        commit_hash = commit_data.get('commit_hash', f'unknown_commit_{i}')
        commit_result: Dict[str, Any] = {'commit_hash': commit_hash}

        # Files Term Frequency 計算
        files_term_freq: Dict[str, int] = {}
        if global_files_vectorizer:
            files_content_map = commit_data.get('files_content_map', {})

            # ここではコミット内の全ファイル内容を結合せず、各ファイルをドキュメントとしてベクトル化し、
            # その総和を求める。_get_cleaned_file_contents がリストを返すのでそれでよい。
            cleaned_contents_for_commit = get_cleaned_file_contents(
                files_content_map)
            if cleaned_contents_for_commit:  # このコミットに変更された有効なファイルがある場合のみ
                files_term_freq = get_term_frequencies_from_vectorizer(
                    cleaned_contents_for_commit,
                    global_files_vectorizer
                )
        commit_result['files_term_freq'] = json.dumps(files_term_freq)

        # Patches Term Frequency 計算
        patch_term_diff: Dict[str, int] = {}
        if global_patch_vectorizer:
            patch_text = commit_data.get('filtered_patch_lines_list', [])
            # ループ内で、現在のコミットのパッチを正しくパースする
            added_str, deleted_str = parse_patch_to_added_deleted_strings(patch_text)
            
            # 追加行または削除行が存在する場合のみ
            if added_str or deleted_str:
                patch_term_diff = get_patch_diff_term_frequencies_from_vectorizer(
                    added_str,
                    deleted_str,
                    global_patch_vectorizer
                )
        commit_result['patch_term_diff'] = json.dumps(patch_term_diff)

        results.append(commit_result)
        if (i + 1) % 100 == 0:  # 100コミットごとに進捗を表示
            logging.info(f"{i+1}/{len(all_commits_data)} コミットの処理完了。")

    print(f'{len(all_files_corpus)} files corpus')
    print(f'{len(all_patch_docs_corpus)} patch corpus')

    logging.info("全てのコミットのテキストメトリクス計算が完了しました。")
    return results


def commit_code_patch_only_diff(patch_as_single_string: str) -> List[str]:
    """
    コミットのコードパッチから，diff形式の行を抽出する
    """
    if not patch_as_single_string:
        return []

    diff_lines = []

    lines_in_patch = patch_as_single_string.splitlines()

    for line_content in lines_in_patch:

        stripped_line = line_content.lstrip()

        is_added_line = stripped_line.startswith(
            '+') and not stripped_line.startswith('+++')
        is_deleted_line = stripped_line.startswith(
            '-') and not stripped_line.startswith('---')

        if is_added_line or is_deleted_line:
            diff_lines.append(line_content)

    return diff_lines


def load_file_data_per_commit_from_csv(csv_filepath: str) -> List[Dict[str, Any]]:
    """
    csvファイルからコミット単位で，ファイルのテキスト(file_text)を読み込み，テキストメトリクス計算用の形式にする
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        logging.error(f"CSVファイルが見つかりません: {csv_filepath}")
        return []
    except Exception as e:
        logging.error(f"CSVファイル読み込み中にエラー: {e}")
        return []

    prepared_data = []

    commit_hash_col = 'commit_hash'
    file_text_col = 'file_text'
    code_patch_per_commit_col = 'commit_code_patch'
    legacy_patch_text_col = 'commit_patch'

    base_required_columns = [commit_hash_col]
    has_filtered_patch_col = code_patch_per_commit_col in df.columns
    has_legacy_patch_col = legacy_patch_text_col in df.columns
    has_file_text_col = file_text_col in df.columns

    patch_source_info_logged = False

    if has_filtered_patch_col:
        patch_source_info_logged = True
    elif has_legacy_patch_col:
        patch_source_info_logged = True

    missing_cols = [
        col for col in base_required_columns if col not in df.columns]

    if missing_cols:
        logging.error(f"基本的な必須列がCSVファイルに存在しません: {', '.join(missing_cols)}")
        return []

    for index, row in df.iterrows():
        commit_hash = row.get(commit_hash_col, f'unknown_commit_index_{index}')
        files_content_map: Dict[str, str] = {}

        if has_file_text_col:
            file_text_json_str = row.get(file_text_col)
            if pd.notna(file_text_json_str) and isinstance(file_text_json_str, str) and file_text_json_str.strip():
                try:
                    # JSON文字列をPythonの辞書としてパース
                    parsed_content = json.loads(file_text_json_str)
                    if isinstance(parsed_content, dict):
                        files_content_map = parsed_content
                    else:
                        logging.warning(
                            f"Commit {commit_hash}: 列 '{file_text_col}' はjson.loadsで辞書に変換できませんでした。型: {type(parsed_content)}。")
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"Commit {commit_hash}: 列 '{file_text_col}' のJSONパースに失敗 (json.loads)。内容抜粋: {str(file_text_json_str)[:100]}... Error: {e}")
                except Exception as e:  # その他の予期せぬエラー
                    logging.warning(
                        f"Commit {commit_hash}: 列 '{file_text_col}' の処理中に予期せぬエラー: {e}")
            # pandasが数値などを自動でdict型に変換する場合があるため、そのケースも考慮
            elif pd.notna(file_text_json_str) and isinstance(file_text_json_str, dict):
                files_content_map = file_text_json_str  # 既に辞書の場合
            elif pd.notna(file_text_json_str) and not file_text_json_str.strip():
                logging.debug(  # 空文字列の場合はデバッグレベルでログ
                    f"Commit {commit_hash}: 列 '{file_text_col}' は空または空白文字列です。")
            elif pd.notna(file_text_json_str):  # 文字列でも辞書でもない、かつ空でもない場合
                logging.warning(
                    f"Commit {commit_hash}: 列 '{file_text_col}' は予期せぬ型です: {type(file_text_json_str)}。内容は '{str(file_text_json_str)[:50]}...'")

        parsed_filtered_lines_list: Optional[List[str]] = None
        if has_filtered_patch_col:
            patch_text_as_lines_str = row.get(code_patch_per_commit_col)

            if patch_text_as_lines_str is not None and isinstance(patch_text_as_lines_str, str):
                try:
                    # list_of_patch_lines: List[str] = patch_text_as_single_string.splitlines(
                    # )

                    parsed_filtered_lines_list = commit_code_patch_only_diff(
                        patch_text_as_lines_str)

                except Exception as e:
                    logging.warning(
                        f"Commit {commit_hash}: 列 '{parsed_filtered_lines_list}' の処理中にエラー: {e}")

        commit_data = {
            'commit_hash': commit_hash,
            'files_content_map': files_content_map,
            'filtered_patch_lines_list': parsed_filtered_lines_list,
        }

        prepared_data.append(commit_data)

        if (index + 1) % 1000 == 0:
            logging.info(f"{index + 1} / {len(df)} 行のCSVデータ準備完了。")

    logging.info(f"CSVからのデータ準備完了。合計 {len(prepared_data)} コミット。")
    return prepared_data


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 引数処理 (argparseを使用) ---
    parser = argparse.ArgumentParser(
        description="Calculate text-based metrics (Files Term Frequency and Patch Term Difference) for each commit from a given CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to the input CSV file. \nThis CSV is expected to be generated by the preceding script (repo_commit_processor_test.py)."
    )
    parser.add_argument(
        "local_repo_name",
        type=str,
        help="Name of the local repository. \nThis is used to determine the output directory for the results."
    )
    
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=None,
        help="Path to the directory where the output CSV will be saved. \nIf not specified, the output file is saved in the same directory as the input CSV."
    )
    args = parser.parse_args()

    csv_file_path = args.input_csv
    local_repo_name = args.local_repo_name

    # --- 出力ディレクトリの決定 ---
    if args.output_dir:
        output_save_dir = args.output_dir
    else:
        # -oが指定されない場合、入力CSVと同じディレクトリに出力する
        output_save_dir = os.path.dirname(os.path.abspath(csv_file_path))

    # 出力ディレクトリが存在しない場合は作成
    if not os.path.isdir(output_save_dir):
        try:
            os.makedirs(output_save_dir, exist_ok=True)
            logging.info(f"出力ディレクトリを作成しました: {output_save_dir}")
        except OSError as e:
            logging.error(f"出力ディレクトリの作成に失敗しました: {e}")
            sys.exit(1)

    # 出力CSVファイルのフルパス
    
    output_csv_filename = f'{local_repo_name}_commit_text_metrics_results.csv'
    output_csv_path = os.path.join(output_save_dir, output_csv_filename)

    logging.info(f"入力CSVファイル: {csv_file_path}")
    logging.info(f"出力CSVファイル: {output_csv_path}")

    # --- メイン処理の実行 ---
    all_commits_input_data = load_file_data_per_commit_from_csv(csv_file_path)

    if all_commits_input_data:
        calculated_metrics_results = calculate_text_metrics_for_all_commits(
            all_commits_input_data,
            token_pattern=DEFAULT_TOKEN_PATTERN,
            min_df_files=MIN_DF_PAPER,
            max_df_files=MAX_DF_PAPER,
            min_df_patches=MIN_DF_PAPER,
            max_df_patches=MAX_DF_PAPER
        )

        if calculated_metrics_results:
            output_df = pd.DataFrame(calculated_metrics_results)
            try:
                output_df.to_csv(output_csv_path, index=False)
                logging.info(f"計算結果を {output_csv_path} に保存しました。")
            except IOError as e:
                logging.error(f"ファイル '{output_csv_path}' への書き込み中にエラー: {e}")
                sys.exit(1)
            except Exception as e:
                logging.error(f"予期せぬエラーによりファイル '{output_csv_path}' の書き込みに失敗: {e}")
                sys.exit(1)
        else:
            logging.error("メトリクス計算中にエラーが発生したか、結果が空でした。")
    else:
        logging.error(f"処理するデータがありませんでした。CSVファイル ({csv_file_path}) のパスや内容を確認してください。")
# ★★★ここまでが修正箇所★★★
