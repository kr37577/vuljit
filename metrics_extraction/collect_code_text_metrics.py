import pandas as pd
import sys
import os
import argparse

def load_and_merge_metrics(
    df_code: pd.DataFrame,
    df_text: pd.DataFrame,
    project_name: str
) -> pd.DataFrame:
    """
    与えられたデータフレームから、単一のプロジェクトでフィルタリング・マージを行う。
    """
    # --- 1. プロジェクト名でフィルタリング ---
    print(f"  -> プロジェクト '{project_name}' のデータを抽出中...")
    df_code_filtered = df_code[df_code['repo_path'].str.contains(project_name, na=False)].copy()

    if df_code_filtered.empty:
        return pd.DataFrame()

    # --- 2. データのマージ ---
    print("  -> メトリクスをマージ中...")
    merged_df = pd.merge(df_code_filtered, df_text, on='commit_hash', how='inner')

    return merged_df

# --- メインの実行部分 ---
if __name__ == '__main__':
    # =================================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ コマンドライン引数の設定 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # =================================================================
    parser = argparse.ArgumentParser(
        description="プロジェクトのコードメトリクスとテキストメトリクスをマージし、分析用のメトリクスのみを抽出してCSVファイルに追記保存します。"
    )
    # (引数の設定は変更なし)
    parser.add_argument('-p', '--project', type=str, required=True, help='処理対象のプロジェクト名を指定します。(必須)')
    parser.add_argument('-c', '--code_metrics', type=str, required=True, help='コードメトリクスCSVファイルのパスを指定します。(必須)')
    parser.add_argument('-t', '--text_metrics', type=str, required=True, help='テキストメトリクスCSVファイルのパスを指定します。(必須)')
    parser.add_argument('-o', '--output', type=str, default='merged_metrics.csv', help='出力ファイル名を指定します。(デフォルト: merged_metrics.csv)')
    args = parser.parse_args()

    # --- 最初に一度だけ全データを読み込む ---
    try:
        print("🔄 全メトリクスデータを読み込みます...")
        df_text_all = pd.read_csv(args.text_metrics)
        df_code_all = pd.read_csv(args.code_metrics)
        print("✅ 全データの読み込み完了。")
    except FileNotFoundError as e:
        print(f"❌ エラー: ファイル '{e.filename}' が見つかりません。パスを確認してください。", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ エラー: CSVファイルの読み込み中に問題が発生しました: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 指定された単一プロジェクトの処理を実行 ---
    print(f"\n▶️  プロジェクト '{args.project}' の処理を開始します...")
    project_dataframe = load_and_merge_metrics(
        df_code=df_code_all,
        df_text=df_text_all,
        project_name=args.project
    )

    # --- 結果の書き込み ---
    if not project_dataframe.empty:
        # =================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ メトリクスのみを抽出 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # =================================================================
        print("\n🔍 分析用のメトリクスと主要な識別子のみを抽出します...")

        # 保持したい主要な識別子（キー）のリスト
        # このリストを編集すれば、残す非数値列をカスタマイズできます。
        keys_to_keep = [
            'commit_hash',
            'repo_path',
            'commit_datetime',
            'is_vcc',
            'commit_change_file_path_filetered',
            # kamei
            'subsystems_changed',
            'directories_changed',
            'files_changed',
            'total_lines_changed',
            'lines_added',
            'lines_deleted',
            'total_prev_loc',
            'is_bug_fix',
            'past_bug_fixes',
            'entropy',
            'ndev',
            'age',
            'nuc',
            'exp',
            'rexp',
            'sexp',
            # 
            # 'mean_days_since_creation',
            # 'mean_past_changes',
            # 'past_different_authors',
            # 'author_past_contributions',
            # 'author_past_contributions_ratio',
            # 'author_30days_past_contributions',
            # 'author_30days_past_contributions_ratio',
            # 'author_workload',
            # 'days_after_creation',
            # 'touched_files',
            # 'number_of_hunks',
            # revising vcc
            'VCC_s1_nb_added_sizeof',
            'VCC_s2_nb_removed_sizeof',
            'VCC_s3_diff_sizeof',
            'VCC_s4_sum_sizeof',
            'VCC_s5_nb_added_continue',
            'VCC_s6_nb_removed_continue',
            'VCC_s7_nb_added_break',
            'VCC_s8_nb_removed_break',
            'VCC_s9_nb_added_INTMAX',
            'VCC_s10_nb_removed_INTMAX',
            'VCC_s11_nb_added_goto',
            'VCC_s12_nb_removed_goto',
            'VCC_s13_nb_added_define',
            'VCC_s14_nb_removed_define',
            'VCC_s15_nb_added_struct',
            'VCC_s16_nb_removed_struct',
            'VCC_s17_diff_struct',
            'VCC_s18_sum_struct',
            'VCC_s19_nb_added_offset',
            'VCC_s20_nb_removed_offset',
            'VCC_s21_nb_added_void',
            'VCC_s22_nb_removed_void',
            'VCC_s23_diff_void',
            'VCC_s24_sum_void',
            'VCC_f1_sum_file_change',
            'VCC_f2_nb_added_loop',
            'VCC_f3_nb_removed_loop',
            'VCC_f4_diff_loop',
            'VCC_f5_sum_loop',
            'VCC_f6_nb_added_if',
            'VCC_f7_nb_removed_if',
            'VCC_f8_diff_if',
            'VCC_f9_sum_if',
            'VCC_f10_nb_added_line',
            'VCC_f11_nb_removed_line',
            'VCC_f12_diff_line',
            'VCC_f13_sum_line',
            'VCC_f14_nb_added_paren',
            'VCC_f15_nb_removed_paren',
            'VCC_f16_diff_paren',
            'VCC_f17_sum_paren',
            'VCC_f18_nb_added_bool',
            'VCC_f19_nb_removed_bool',
            'VCC_f20_diff_bool',
            'VCC_f21_sum_bool',
            'VCC_f22_nb_added_assignement',
            'VCC_f23_nb_removed_assignement',
            'VCC_f24_diff_assignement',
            'VCC_f25_sum_assignement',
            'VCC_f26_nb_added_function',
            'VCC_f27_nb_removed_function',
            'VCC_f28_diff_function',
            'VCC_f29_sum_function',      
                ]

        # project_dataframeに存在するキーのみを対象にする
        existing_keys = [col for col in keys_to_keep if col in project_dataframe.columns]

        # データフレームから数値型(int, floatなど)の列名リストを自動で取得
        numeric_cols = project_dataframe.select_dtypes(include='number').columns.tolist()

        # 最終的に保存する列のリスト = 識別子リスト + 数値メトリクスリスト
        final_cols_to_save = existing_keys + numeric_cols
        
        # 重複する列があれば削除しつつ、順序を保持
        final_cols_to_save = list(dict.fromkeys(final_cols_to_save))

        # 抽出した列のみで新しいデータフレームを作成
        metrics_only_dataframe = project_dataframe[final_cols_to_save]
        
        print(f"  -> {len(project_dataframe.columns)}列から{len(metrics_only_dataframe.columns)}列に絞り込みました。")
        # =================================================================
        
        output_file = args.output
        
        output_file_exists = os.path.exists(output_file)
        print(f"\n🔄 ファイル '{output_file}' への書き込み準備中...")
        if output_file_exists:
            print("  -> ファイルが存在するため、末尾に追記します。")
        else:
            print("  -> ファイルが存在しないため、新規に作成します。")

        try:
            # 抽出後のデータフレームをCSVに保存
            metrics_only_dataframe.to_csv(
                output_file,
                mode='w',
                header=True,
                index=False,
                encoding='utf-8-sig'
            )
            action = "追記" if output_file_exists else "新規作成"
            print(f"✅ プロジェクト '{args.project}' の結果 ({len(metrics_only_dataframe)}件) を '{output_file}' に{action}しました。")
        except Exception as e:
            print(f"❌ エラー: ファイル保存中に問題が発生しました: {e}", file=sys.stderr)
    else:
        print(f"ℹ️ プロジェクト '{args.project}' では条件に合うデータが見つからなかったため、ファイルへの書き込みは行いません。")

    print("\n🎉 処理が完了しました。")