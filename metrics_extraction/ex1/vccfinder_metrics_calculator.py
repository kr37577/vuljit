# vccfinder_metrics_calculator.py

import logging
import os
import re
from typing import Dict, Any, List
import git


# VCCFinder論文由来のメトリクス名を定義 (commit_patch_to_code_metrics_vector の row の順序に対応)
VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER = [
    # Security-sensitive features (s1-s24)
    "VCC_s1_nb_added_sizeof",              # s1 = nb_added_sizeof
    "VCC_s2_nb_removed_sizeof",            # s2 = nb_removed_sizeof
    "VCC_s3_diff_sizeof",                  # s3 = nb_added_sizeof - nb_removed_sizeof
    "VCC_s4_sum_sizeof",                   # s4 = nb_added_sizeof + nb_removed_sizeof
    "VCC_s5_nb_added_continue",            # s5 = nb_added_continue
    "VCC_s6_nb_removed_continue",          # s6 = nb_removed_continue
    "VCC_s7_nb_added_break",               # s7 = nb_added_break
    "VCC_s8_nb_removed_break",             # s8 = nb_removed_break
    "VCC_s9_nb_added_INTMAX",              # s9 = nb_added_INTMAX
    "VCC_s10_nb_removed_INTMAX",           # s10 = nb_removed_INTMAX
    "VCC_s11_nb_added_goto",               # s11 = nb_added_goto
    "VCC_s12_nb_removed_goto",             # s12 = nb_removed_goto
    "VCC_s13_nb_added_define",             # s13 = nb_added_define
    "VCC_s14_nb_removed_define",           # s14 = nb_removed_define
    "VCC_s15_nb_added_struct",             # s15 = nb_added_struct
    "VCC_s16_nb_removed_struct",           # s16 = nb_removed_struct
    "VCC_s17_diff_struct",                 # s17 = nb_added_struct - nb_removed_struct
    "VCC_s18_sum_struct",                  # s18 = nb_added_struct + nb_removed_struct
    "VCC_s19_nb_added_offset",             # s19 = nb_added_offset
    "VCC_s20_nb_removed_offset",           # s20 = nb_removed_offset
    "VCC_s21_nb_added_void",               # s21 = nb_added_void
    "VCC_s22_nb_removed_void",             # s22 = nb_removed_void
    "VCC_s23_diff_void",                   # s23 = nb_added_void - nb_removed_void
    "VCC_s24_sum_void",                    # s24 = nb_added_void + nb_removed_void

    # Code-fix features (f1-f29)
    "VCC_f1_sum_file_change",              # f1 = nb_added_file + nb_removed_file
    "VCC_f2_nb_added_loop",                # f2 = nb_added_loop
    "VCC_f3_nb_removed_loop",              # f3 = nb_removed_loop
    "VCC_f4_diff_loop",                    # f4 = nb_added_loop - nb_removed_loop
    "VCC_f5_sum_loop",                     # f5 = nb_added_loop + nb_removed_loop
    "VCC_f6_nb_added_if",                  # f6 = nb_added_if
    "VCC_f7_nb_removed_if",                # f7 = nb_removed_if
    "VCC_f8_diff_if",                      # f8 = nb_added_if - nb_removed_if
    "VCC_f9_sum_if",                       # f9 = nb_added_if + nb_removed_if
    "VCC_f10_nb_added_line",               # f10 = nb_added_line
    "VCC_f11_nb_removed_line",             # f11 = nb_removed_line
    "VCC_f12_diff_line",                   # f12 = nb_added_line - nb_removed_line
    "VCC_f13_sum_line",                    # f13 = nb_added_line + nb_removed_line
    "VCC_f14_nb_added_paren",              # f14 = nb_added_paren
    "VCC_f15_nb_removed_paren",            # f15 = nb_removed_paren
    "VCC_f16_diff_paren",                  # f16 = nb_added_paren - nb_removed_paren
    "VCC_f17_sum_paren",                   # f17 = nb_added_paren + nb_removed_paren
    "VCC_f18_nb_added_bool",               # f18 = nb_added_bool
    "VCC_f19_nb_removed_bool",             # f19 = nb_removed_bool
    "VCC_f20_diff_bool",                   # f20 = nb_added_bool - nb_removed_bool
    "VCC_f21_sum_bool",                    # f21 = nb_added_bool + nb_removed_bool
    "VCC_f22_nb_added_assignement",        # f22 = nb_added_assignement
    "VCC_f23_nb_removed_assignement",      # f23 = nb_removed_assignement
    # f24 = nb_added_assignement - nb_removed_assignement
    "VCC_f24_diff_assignement",
    # f25 = nb_added_assignement + nb_removed_assignement
    "VCC_f25_sum_assignement",
    "VCC_f26_nb_added_function",           # f26 = nb_added_function
    "VCC_f27_nb_removed_function",         # f27 = nb_removed_function
    # f28 = nb_added_function - nb_removed_function
    "VCC_f28_diff_function",
    # f29 = nb_added_function + nb_removed_function
    "VCC_f29_sum_function",

    # Raw counts (appended to the row after sX and fX features)
    "VCC_nb_added_sizeof_raw",
    "VCC_nb_removed_sizeof_raw",
    "VCC_nb_added_continue_raw",
    "VCC_nb_removed_continue_raw",
    "VCC_nb_added_break_raw",
    "VCC_nb_removed_break_raw",
    "VCC_nb_added_INTMAX_raw",
    "VCC_nb_removed_INTMAX_raw",
    "VCC_nb_added_goto_raw",
    "VCC_nb_removed_goto_raw",
    "VCC_nb_added_define_raw",
    "VCC_nb_removed_define_raw",
    "VCC_nb_added_struct_raw",
    "VCC_nb_removed_struct_raw",
    "VCC_nb_added_void_raw",
    "VCC_nb_removed_void_raw",
    "VCC_nb_added_offset_raw",
    "VCC_nb_removed_offset_raw",
    "VCC_nb_added_line_raw",
    "VCC_nb_removed_line_raw",
    "VCC_nb_added_if_raw",
    "VCC_nb_removed_if_raw",
    "VCC_nb_added_loop_raw",
    "VCC_nb_removed_loop_raw",
    "VCC_nb_added_file_raw",
    "VCC_nb_removed_file_raw",
    "VCC_nb_added_function_raw",
    "VCC_nb_removed_function_raw",
    "VCC_nb_added_paren_raw",
    "VCC_nb_removed_paren_raw",
    "VCC_nb_added_bool_raw",
    "VCC_nb_removed_bool_raw",
    "VCC_nb_added_assignement_raw",
    "VCC_nb_removed_assignement_raw",
]

CODE_FILE_EXTENSIONS = [
    '.c', '.cc', '.cpp', '.cxx', '.c++',  # C/C++ ソースファイル
    '.h', '.hh', '.hpp', '.hxx', '.h++'   # C/C++ ヘッダーファイル
]


def is_code_file(extension_list: List[str]) -> bool:
    """
    Check if the file extension is in the list of code file extensions.
    コードファイルの拡張子がリストに含まれているか確認する
    """
    for ext in extension_list:
        # 拡張子は大文字・小文字を区別しないように小文字に変換して比較
        if ext.lower() in CODE_FILE_EXTENSIONS:
            return True
    return False


def filter_diff_for_code_files_from_gitpython(diff_items: List[git.diff.Diff]) -> List[str]:
    """
    GitPython の Diff オブジェクトのリストを入力として受け取ります
    コードファイルに対する変更のdiff行のみを連結したリストを返します
    これには、commit_patch_to_code_metrics_vector が期待する
    --- a/... および +++ b/... ヘッダー行が含まれます。
    """
    filtered_patch_lines: List[str] = []
    for diff_item in diff_items:
        # 拡張子チェック用のファイルパスを決定
        file_path_for_ext_check = diff_item.b_path if diff_item.b_path else diff_item.a_path
        if not file_path_for_ext_check:
            logging.debug("Skipping diff_item with no a_path or b_path.")
            continue

        _, extension = os.path.splitext(file_path_for_ext_check)
        if is_code_file([extension]):
            logging.debug(
                f"Processing code file: {file_path_for_ext_check}, Change type: {diff_item.change_type}")
            logging.debug(
                f"  a_path: {diff_item.a_path}, b_path: {diff_item.b_path}")
            logging.debug(
                f"  new_file: {diff_item.new_file}, deleted_file: {diff_item.deleted_file}, renamed_file: {diff_item.renamed_file}")
            logging.debug(
                f"  diff_item.diff is None: {diff_item.diff is None}")
            if diff_item.diff is not None:
                logging.debug(f"  diff_item.diff type: {type(diff_item.diff)}")
                try:
                    # diff内容の最初の数文字をログに出力（長すぎる場合を考慮）
                    diff_preview = diff_item.diff[:200] if isinstance(
                        diff_item.diff, str) else diff_item.diff.decode('utf-8', errors='replace')[:200]
                    logging.debug(
                        f"  diff_item.diff content preview: {diff_preview}...")
                except Exception as e:
                    logging.debug(f"  Error previewing diff_item.diff: {e}")

            # 標準的なdiffヘッダーを追加 (末尾に改行を追加)
            header_added = False
            if diff_item.new_file:
                filtered_patch_lines.append(f"--- a/{'/dev/null'}\n")
                if diff_item.b_path:  # b_path が None でないことを確認
                    filtered_patch_lines.append(f"+++ b/{diff_item.b_path}\n")
                    header_added = True
            elif diff_item.deleted_file:
                if diff_item.a_path:  # a_path が None でないことを確認
                    filtered_patch_lines.append(f"--- a/{diff_item.a_path}\n")
                    header_added = True
                filtered_patch_lines.append(f"+++ b/{'/dev/null'}\n")
            elif diff_item.renamed_file:  # リネームの場合、a_path と b_path 両方が存在するはず
                if diff_item.a_path and diff_item.b_path:
                    filtered_patch_lines.append(f"--- a/{diff_item.a_path}\n")
                    filtered_patch_lines.append(f"+++ b/{diff_item.b_path}\n")
                    header_added = True
            else:  # 通常の変更
                # a_path と b_path が同じ場合もある
                path_for_header = diff_item.a_path if diff_item.a_path else diff_item.b_path
                if path_for_header:  # path_for_header が None でないことを確認
                    filtered_patch_lines.append(f"--- a/{path_for_header}\n")
                    filtered_patch_lines.append(
                        f"+++ b/{path_for_header}\n")  # 通常の変更ではaとbは同じパス
                    header_added = True

            if not header_added and (diff_item.a_path or diff_item.b_path):
                logging.warning(
                    f"File headers might be incomplete for {file_path_for_ext_check} (new:{diff_item.new_file}, del:{diff_item.deleted_file}, ren:{diff_item.renamed_file})")

            if diff_item.diff is not None:
                try:
                    diff_text_content: str
                    if isinstance(diff_item.diff, bytes):
                        diff_text_content = diff_item.diff.decode(
                            'utf-8', errors='replace')
                    else:
                        diff_text_content = diff_item.diff  # str型と仮定

                    if diff_text_content:  # 空文字列でないことを確認
                        current_file_diff_lines = diff_text_content.splitlines(
                            True)
                        filtered_patch_lines.extend(current_file_diff_lines)
                    else:
                        logging.debug(
                            f"  diff_text_content was empty for {file_path_for_ext_check} after potential decode.")
                except UnicodeDecodeError:
                    logging.warning(
                        f"UnicodeDecodeError processing diff content for file {file_path_for_ext_check}. Skipping its diff content.")
                    continue
                except Exception as e:  # その他の予期せぬエラー
                    logging.error(
                        f"Error processing diff content for {file_path_for_ext_check}: {e}", exc_info=True)
                    continue
            else:
                # diff_item.diff が None の場合でも、ファイル追加/削除のヘッダーは上で追加されているはず
                logging.debug(
                    f"  diff_item.diff was None for {file_path_for_ext_check}. Only file headers (if applicable) are added.")
    return filtered_patch_lines


def commit_patch_to_code_metrics_vector(commit_patch: List[str]) -> List[int]:
    """
    Code_metrics
    returns ONE feature vector from ONE commit_patch
    EXPECTS: only one commit at a time (list of patch lines)
    """
    # (get_feature_commit_func.py から移動した commit_patch_to_code_metrics_vector 関数の実装をここに記述)
    # ... (関数の内容は前回の回答を参照) ...
    # 以下は関数の主要部分の再掲（実際のコードはこのコメントを置き換えてください）
    nb_added_if, nb_removed_if = 0, 0
    nb_added_loop, nb_removed_loop = 0, 0
    nb_added_file, nb_removed_file = 0, 0
    nb_added_function, nb_removed_function = 0, 0
    nb_added_paren, nb_removed_paren = 0, 0
    nb_added_bool, nb_removed_bool = 0, 0
    nb_added_assignement, nb_removed_assignement = 0, 0
    nb_added_break, nb_removed_break = 0, 0
    nb_added_sizeof, nb_removed_sizeof = 0, 0
    nb_added_return, nb_removed_return = 0, 0
    nb_added_continue, nb_removed_continue = 0, 0
    nb_added_INTMAX, nb_removed_INTMAX = 0, 0
    nb_added_goto, nb_removed_goto = 0, 0
    nb_added_define, nb_removed_define = 0, 0
    nb_added_struct, nb_removed_struct = 0, 0
    nb_added_void, nb_removed_void = 0, 0
    nb_added_offset, nb_removed_offset = 0, 0
    nb_added_line, nb_removed_line = 0, 0

    kbp, kbn = 0, 0

    # Log the received commit_patch content
    logging.debug(
        f"commit_patch_to_code_metrics_vector received {len(commit_patch)} lines.")
    if commit_patch:  # リストが空でない場合のみ最初の数行を表示
        logging.debug(f" commit_patch: {commit_patch}")

    for line_with_newline in commit_patch:  # commit_patch は改行を含む行のリスト
        line = line_with_newline.rstrip('\n\r')  # 末尾の改行を除去して処理
        if line.startswith("+") and not line.startswith("+++"):  # コード行の追加
            nb_added_line += 1
            if "if (" in line:
                nb_added_if += 1
            if ("for (" in line) or ("while (" in line):
                nb_added_loop += 1
            # 関数定義の判定
            if any(kw in line for kw in ["int ", "static int ", "void ", "char *", "float ", "double "]) and "(" in line and ")" in line and not line.strip().startswith("if") and not line.strip().startswith("for") and not line.strip().startswith("while"):
                # 関数定義の判定
                if "{" in line or line.endswith(";") == False:
                    nb_added_function += 1
            if ("(" in line) and (")" in line):
                nb_added_paren += 1
            if any(op in line for op in ["||", "&&", "!"]) and "!=" not in line:
                nb_added_bool += 1
            if "=" in line and "==" not in line and "!=" not in line and "<=" not in line and ">=" not in line:
                nb_added_assignement += line.count('=')
            if "sizeof" in line:
                nb_added_sizeof += 1
            if "break" in line:
                nb_added_break += 1
            if "return" in line:
                nb_added_return += 1
            if "continue" in line:
                nb_added_continue += 1
            if "INT_MAX" in line:
                nb_added_INTMAX += 1  # INT_MAX
            if "goto" in line:
                nb_added_goto += 1
            if "#define" in line:
                nb_added_define += 1
            if "struct" in line and "{" in line:
                nb_added_struct += 1  # 構造体定義
            if "void " in line and "(" in line:
                nb_added_void += 1

        elif line.startswith("-") and not line.startswith("---"):
            nb_removed_line += 1
            if "if (" in line:
                nb_removed_if += 1
            if ("for (" in line) or ("while (" in line):
                nb_removed_loop += 1
            if any(kw in line for kw in ["int ", "static int ", "void ", "char *", "float ", "double "]) and "(" in line and ")" in line and not line.strip().startswith("if") and not line.strip().startswith("for") and not line.strip().startswith("while"):
                if "{" in line or line.endswith(";") == False:
                    nb_removed_function += 1
            if ("(" in line) and (")" in line):
                nb_removed_paren += 1
            if any(op in line for op in ["||", "&&", "!"]) and "!=" not in line:
                nb_removed_bool += 1
            if "=" in line and "==" not in line and "!=" not in line and "<=" not in line and ">=" not in line:
                nb_removed_assignement += line.count('=')
            if "sizeof" in line:
                nb_removed_sizeof += 1
            if "break" in line:
                nb_removed_break += 1
            if "return" in line:
                nb_removed_return += 1
            if "continue" in line:
                nb_removed_continue += 1
            if "INT_MAX" in line:
                nb_removed_INTMAX += 1
            if "goto" in line:
                nb_removed_goto += 1
            if "#define" in line:
                nb_removed_define += 1
            if "struct" in line and "{" in line:
                nb_removed_struct += 1
            if "void " in line and "(" in line:
                nb_removed_void += 1

        # --- a/file や +++ b/file の行をカウント (nb_added_file, nb_removed_file)
        if line.startswith("--- a/"):
            nb_removed_file += 1
        if line.startswith("+++ b/"):
            nb_added_file += 1

        # offset のカウント
        if ("offset =" in line) or ("offset=" in line) or ("->offset" in line):
            if line.startswith("+") and not line.startswith("+++"):
                nb_added_offset += 1
            elif line.startswith("-") and not line.startswith("---"):
                nb_removed_offset += 1

    logging.debug(
        f"Calculated in commit_patch_to_code_metrics_vector: nb_added_line={nb_added_line}, nb_removed_line={nb_removed_line}")

    # security-sensitive features
    # sizeof
    s1 = nb_added_sizeof
    s2 = nb_removed_sizeof
    s3 = nb_added_sizeof - nb_removed_sizeof
    s4 = nb_added_sizeof + nb_removed_sizeof

    # continue
    s5 = nb_added_continue
    s6 = nb_removed_continue

    # break
    s7 = nb_added_break
    s8 = nb_removed_break

    # INT_MAX
    s9 = nb_added_INTMAX
    s10 = nb_removed_INTMAX

    # goto
    s11 = nb_added_goto
    s12 = nb_removed_goto

    # define
    s13 = nb_added_define
    s14 = nb_removed_define

    # struct
    s15 = nb_added_struct
    s16 = nb_removed_struct
    s17 = nb_added_struct - nb_removed_struct
    s18 = nb_added_struct + nb_removed_struct

    # offset
    s19 = nb_added_offset
    s20 = nb_removed_offset

    # void
    s21 = nb_added_void
    s22 = nb_removed_void
    s23 = nb_added_void - nb_removed_void
    s24 = nb_added_void + nb_removed_void

    # code-fix features
    # file change
    f1 = nb_added_file + nb_removed_file

    # loop
    f2 = nb_added_loop
    f3 = nb_removed_loop
    f4 = nb_added_loop - nb_removed_loop
    f5 = nb_added_loop + nb_removed_loop

    # if
    f6 = nb_added_if
    f7 = nb_removed_if
    f8 = nb_added_if - nb_removed_if
    f9 = nb_added_if + nb_removed_if

    # line
    f10 = nb_added_line
    f11 = nb_removed_line
    f12 = nb_added_line - nb_removed_line
    f13 = nb_added_line + nb_removed_line

    # parenthesis
    f14 = nb_added_paren
    f15 = nb_removed_paren
    f16 = nb_added_paren - nb_removed_paren
    f17 = nb_added_paren + nb_removed_paren

    # bool
    f18 = nb_added_bool
    f19 = nb_removed_bool
    f20 = nb_added_bool - nb_removed_bool
    f21 = nb_added_bool + nb_removed_bool

    # assignment
    f22 = nb_added_assignement
    f23 = nb_removed_assignement
    f24 = nb_added_assignement - nb_removed_assignement
    f25 = nb_added_assignement + nb_removed_assignement

    # function call
    f26 = nb_added_function
    f27 = nb_removed_function
    f28 = nb_added_function - nb_removed_function
    f29 = nb_added_function + nb_removed_function

    # expressionはない

    row = [
        s1, s2, s3, s4,
        s5, s6,
        s7, s8,
        s9, s10,
        s11, s12,
        s13, s14,
        s15, s16, s17, s18,
        s19, s20,
        s21, s22, s23, s24,
        f1,
        f2, f3, f4, f5,
        f6, f7, f8, f9,
        f10, f11, f12, f13,
        f14, f15, f16, f17,
        f18, f19, f20, f21,
        f22, f23, f24, f25,
        f26, f27, f28, f29,
        nb_added_sizeof, nb_removed_sizeof,
        nb_added_continue, nb_removed_continue,
        nb_added_break, nb_removed_break,
        nb_added_INTMAX, nb_removed_INTMAX,
        nb_added_goto, nb_removed_goto,
        nb_added_define, nb_removed_define,
        nb_added_struct, nb_removed_struct,
        nb_added_void, nb_removed_void,
        nb_added_offset, nb_removed_offset,
        nb_added_line, nb_removed_line,
        nb_added_if, nb_removed_if,
        nb_added_loop, nb_removed_loop,
        nb_added_file, nb_removed_file,
        nb_added_function, nb_removed_function,
        nb_added_paren, nb_removed_paren,
        nb_added_bool, nb_removed_bool,
        nb_added_assignement, nb_removed_assignement,
    ]
    return row


def calculate_and_populate_vccfinder_metrics(
    metrics_dict: Dict[str, Any],
    diff_commit_obj: git.diff.DiffIndex,  # GitPython DiffIndex object
    commit_hash_for_logging: str  # ログ出力用
) -> None:
    """
    VCCFinder関連のメトリクスを計算し，提供された辞書を更新する
    diff_commit_obj は親コミットと現コミットの差分オブジェクト
    """
    # 1. コードファイルのみのパッチを含んだ全ての行を取得
    #    diff_commit_obj は DiffIndex であり、イテレートすると Diff オブジェクトが得られる
    patch_lines_for_code_files = filter_diff_for_code_files_from_gitpython(
        list(diff_commit_obj))

    if patch_lines_for_code_files:
        try:
            vccfinder_metrics_values = commit_patch_to_code_metrics_vector(
                patch_lines_for_code_files)

            if len(vccfinder_metrics_values) == len(VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER):
                for i, metric_name in enumerate(VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER):
                    metrics_dict[metric_name] = vccfinder_metrics_values[i]
            else:
                logging.warning(f"Mismatch in VCCFinder metrics count for {commit_hash_for_logging}. "
                                f"Expected {len(VCCFINDER_METRIC_NAMES_FROM_SOURCE_ORDER)}, "
                                f"got {len(vccfinder_metrics_values)}. VCC metrics will be default.")
        except Exception as e:
            logging.error(
                f"Error calculating VCCFinder metrics for {commit_hash_for_logging}: {e}", exc_info=True)
            # エラー時は metrics_dict の VCCFinder関連メトリクスは初期値のまま
    else:
        logging.info(
            f"No code file changes found for VCCFinder metrics in {commit_hash_for_logging}.")
        # コードファイルの変更がない場合も、VCCFinder関連メトリクスは初期値のまま


def main():
    # test
    repo_path = "clone/apache___arrow"
    repo = git.Repo(repo_path)
    commit_hash = "071ea1bc245446f6ee257bb1ed7d056c46c2868e"
    commit = repo.commit(commit_hash)
    diff_commit_obj = commit.parents[0].diff(commit, create_patch=True)

    metrics_dict = {}
    commit_hash_for_logging = commit.hexsha
    calculate_and_populate_vccfinder_metrics(
        metrics_dict, diff_commit_obj, commit_hash_for_logging)
    print(metrics_dict)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
    main()
