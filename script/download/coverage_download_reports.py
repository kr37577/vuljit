import csv
from google.cloud import storage
import os
from datetime import datetime
import concurrent.futures
import threading
import gzip  # gzipモジュールをインポート

# スレッドセーフなカウンタクラス


def get_zip_filenames_from_directory(directory_path: str) -> set[str]:
    """
    指定されたディレクトリ内にあるZIPファイルの名前（拡張子付き）をセットで取得します。
    この関数は指定されたディレクトリの直下のみを検索し、サブディレクトリ内は検索しません。

    Args:
        directory_path (str): 検索対象のディレクトリのパス。

    Returns:
        set[str]: ディレクトリ内に見つかったZIPファイル名のセット。
                  ファイルが見つからない場合や、指定されたパスが
                  ディレクトリでない、または存在しない場合は空のセットを返します。
    """
    zip_filenames = set()
    if not os.path.isdir(directory_path):
        print(f"エラー: 指定されたパス '{directory_path}' はディレクトリではありません、または存在しません。")
        return zip_filenames

    try:
        for item_name in os.listdir(directory_path):
            # アイテムへのフルパスを作成
            item_path = os.path.join(directory_path, item_name)
            # それがファイルであり、かつ拡張子が .zip (大文字・小文字を問わない) かどうかをチェック
            if os.path.isfile(item_path) and item_name.lower().endswith('.zip'):
                zip_filenames.add(item_name)
    except OSError as e:
        # ディレクトリの読み取り中にエラーが発生した場合など
        print(f"エラー: ディレクトリ '{directory_path}' のスキャン中にエラーが発生しました: {e}")
        # エラーが発生した場合、それまでに見つかったファイル名のセットを返すか、
        # 空のセットを返すか選択できます。ここでは、それまでの結果を返します。

    return zip_filenames


class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, amount=1):
        with self._lock:
            self._value += amount

    def get_value(self):
        with self._lock:
            return self._value


def download_target_files_from_gcs_period(csv_data, start_date_str, end_date_str, target_filenames, save_directory, max_workers=10):
    """
    CSVから重複を除いたpackage_nameを読み込み、指定期間内のGCSレポートディレクトリ下の
    {target_name}/linux サブディレクトリを探索し、指定されたファイル名リストに一致するファイルを
    ダウンロードして .gz 形式で圧縮し、階層構造で保存する (並列処理・ターゲットディレクトリ対応・圧縮対応版)。

    Args:
        csv_data (str): CSV形式の文字列データ。
        start_date_str (str): ダウンロード対象期間の開始日 ('YYYYMMDD'形式)。
        end_date_str (str): ダウンロード対象期間の終了日 ('YYYYMMDD'形式)。
        target_filenames (list[str]): ダウンロードしたいファイル名のリスト (例: ['summary.json'])。
        save_directory (str): 保存先ルートディレクトリパス。
                                この下に {package_name}/{date}/{target_name}/linux/{filename}.gz が作成される。
        max_workers (int): 並列ダウンロードに使用する最大ワーカースレッド数。
    """
    # target_filenames が None の場合や空の場合のチェック（これは前回同様のものです）
    if target_filenames is None:
        target_filenames = []
    if not target_filenames:
        print("警告: ダウンロード対象のファイル名リスト (target_filenames) が空またはNoneです。")
        # return # 必要に応じて処理を中断

    os.makedirs(save_directory, exist_ok=True)

    try:
        start_date = datetime.strptime(start_date_str, '%Y%m%d')
        end_date = datetime.strptime(end_date_str, '%Y%m%d')
        if start_date > end_date:
            print("エラー: 開始日が終了日より後になっています。")
            return
        print(
            f"指定された期間: {start_date.strftime('%Y-%m-%d')} から {end_date.strftime('%Y-%m-%d')}")
        print(
            f"ダウンロード対象ファイル (各ターゲットの linux ディレクトリ内、.gzで圧縮): {target_filenames}")
        print(f"最大並列ワーカー数: {max_workers}")
    except ValueError:
        print("エラー: 開始日または終了日の形式が不正です ('YYYYMMDD'形式で指定してください)。")
        return

    try:
        storage_client = storage.Client.create_anonymous_client()
        print("GCS匿名クライアントの初期化に成功しました。")
    except Exception as e:
        print(f"エラー: GCSクライアントの初期化に失敗しました: {e}")
        print("認証情報が設定されているか、または gcloud auth application-default login を実行してみてください。")
        return

    bucket_name = "oss-fuzz-coverage"
    try:
        bucket = storage_client.bucket(bucket_name)
        print(f"GCSバケット '{bucket_name}' へのアクセスを確認しました。")
    except Exception as e:
        print(f"エラー: GCSバケット '{bucket_name}' の取得またはアクセスに失敗しました: {e}")
        return

    unique_package_names = set()
    try:
        if csv_data.startswith('\ufeff'):
            csv_data = csv_data[1:]
        reader = csv.reader(csv_data.strip().splitlines())
        header = next(reader)
        try:
            package_name_index = header.index('package_name')
        except ValueError:
            print("エラー: CSVヘッダーに 'package_name' 列が見つかりません。")
            return

        processed_rows = 0
        skipped_rows = 0
        for i, row in enumerate(reader):
            if not row:
                skipped_rows += 1
                continue
            if len(row) > package_name_index:
                package_name = row[package_name_index].strip()
                if package_name:
                    unique_package_names.add(package_name)
                    processed_rows += 1
                else:
                    skipped_rows += 1
            else:
                skipped_rows += 1

        if not unique_package_names:
            print("エラー: CSVから有効なパッケージ名を抽出できませんでした。")
            return
        print(
            f"CSVから {len(unique_package_names)} 個の一意なパッケージ名を抽出しました。(処理行数: {processed_rows}, スキップ行数: {skipped_rows})")

    except StopIteration:
        print("エラー: CSVデータが空か、ヘッダー行のみです。")
        return
    except Exception as e:
        print(f"エラー: CSVデータの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # --- ここから修正 ---
    # unique_package_names から .zip で終わるパッケージ名を除外する処理
    original_package_names_count = len(unique_package_names)

    # save_directory の直下にある .zip ファイル名を取得
    # (例: {"curl.zip", "LibXml2.ZIP", "otherfile.zip"})
    # get_zip_filenames_from_directory 関数は既にファイルの上部で定義されていますね。
    marker_zip_files = get_zip_filenames_from_directory(save_directory)

    # マーカーファイル名から拡張子 .zip を取り除き、小文字に変換したベース名のセットを作成
    # (例: {"curl", "libxml2", "otherfile"})
    # これらが「既に処理済み」とみなすパッケージのベース名となる
    processed_package_basenames = set()
    if marker_zip_files:  # セットが空でない場合のみ処理
        for zip_file_name in marker_zip_files:
            # ファイル名が実際に .zip で終わるか確認し、拡張子を除去
            if zip_file_name.lower().endswith('.zip'):
                # basename = zip_file_name[:-len('.zip')] # ".zip" の長さを引く方法
                # より安全なのは、Python 3.9+ なら removesuffix
                try:
                    # Python 3.9+ の場合
                    basename = zip_file_name.lower().removesuffix('.zip')
                except AttributeError:
                    # Python 3.9 未満の場合の代替
                    if zip_file_name.lower().endswith('.zip'):
                        # 最後の4文字(.zip)を削除
                        basename = zip_file_name[:-4].lower()
                    else:
                        basename = zip_file_name.lower()  # .zipで終わらない場合はそのまま（通常はここに来ないはず）
                processed_package_basenames.add(basename)
            # else: .zip で終わらないファイル名は無視（get_zip_filenames_from_directory の仕様上、基本的には.zipのはず）

    # unique_package_names (CSVからのパッケージ名) のうち、
    # processed_package_basenames に含まれていないものだけを処理対象とする
    # (unique_package_names の各要素も小文字に変換して比較することで、大文字・小文字の違いを吸収)
    if processed_package_basenames:  # 処理済みリストが空でなければフィルタリング
        filtered_package_names = {
            pkg_name for pkg_name in unique_package_names
            if pkg_name.lower() not in processed_package_basenames
        }
    else:
        # 処理済みを示すマーカーファイルが一つもなければ、全てのパッケージが対象
        filtered_package_names = unique_package_names  # 元のセットをそのまま使う

    # 正しい removed_count の計算
    removed_count = original_package_names_count - len(filtered_package_names)

    if removed_count > 0:
        print(
            f"情報: 処理対象のパッケージリストから {removed_count} 個の末尾が '.zip' のパッケージ名を除外しました。")

    unique_package_names = filtered_package_names  # フィルタリング後のセットで上書き

    if not unique_package_names:
        print("警告: .zip パッケージ名を除外した結果、処理対象のパッケージがなくなりました。")
        # この時点でダウンロード対象がなければ、処理を終了することも可能です。
        # print("処理を中断します。")
        # return

    print(f"フィルタリング後、処理対象となる一意なパッケージ名は {len(unique_package_names)} 個です。")
    # --- 修正ここまで ---

    total_downloaded_counter = ThreadSafeCounter()

    def process_package(package_name):
        """
        単一パッケージの処理を効率化したバージョン。
        GCSへのAPI呼び出しを1回に集約し、ローカルでフィルタリングとダウンロードを行う。
        2種類のパス構造に対応:
        1. [proj]/reports/[date]/[target]/linux/[file]
        2. [proj]/reports/[date]/linux/[file]
        """
        package_local_download_count = 0
        print(f"[{package_name}] 処理開始...")

        report_base_prefix = f"{package_name}/reports/"
        all_blobs_for_package = []
        try:
            # GCSへのAPI呼び出しを1回に集約
            # prefixに一致するすべてのblobを再帰的に取得
            blobs_iterator = storage_client.list_blobs(bucket_name, prefix=report_base_prefix)
            all_blobs_for_package = list(blobs_iterator)
            if not all_blobs_for_package:
                print(f"[{package_name}] 完了: GCS上にレポートファイルが見つかりませんでした。")
                return 0
        except Exception as e:
            print(f"[{package_name}] エラー: GCSのファイルリスト一括取得中にエラーが発生しました ({report_base_prefix}): {e}")
            return 0

        # ローカルでフィルタリングとダウンロード処理
        for blob in all_blobs_for_package:
            try:
                # blob.nameからパスの各部分を抽出
                path_parts = blob.name.split('/')
                
                date_str, target_name, file_name, local_save_dir = "", "", "", ""

                # 構造1: [proj]/reports/[date]/[target]/linux/[file]
                if len(path_parts) == 6 and path_parts[4] == 'linux':
                    date_str = path_parts[2]
                    target_name = path_parts[3]
                    file_name = path_parts[5]
                    local_save_dir = os.path.join(save_directory, package_name, date_str, target_name, "linux")
                # 構造2: [proj]/reports/[date]/linux/[file]
                elif len(path_parts) == 5 and path_parts[3] == 'linux':
                    date_str = path_parts[2]
                    target_name = "no_target"  # ターゲット名がない場合のプレースホルダー
                    file_name = path_parts[4]
                    local_save_dir = os.path.join(save_directory, package_name, date_str, "linux")
                else:
                    # どちらの構造にも一致しない場合はスキップ
                    continue

                # ファイル名がダウンロード対象かチェック
                if file_name not in target_filenames:
                    continue

                # 日付が指定期間内かチェック
                if len(date_str) == 8 and date_str.isdigit():
                    current_date = datetime.strptime(date_str, '%Y%m%d')
                    if not (start_date <= current_date <= end_date):
                        continue
                else:
                    continue # 日付形式でないものはスキップ

                # 保存パスを構築し、ダウンロード処理
                compressed_file_path = os.path.join(local_save_dir, f"{file_name}.gz")

                # 既にファイルが存在する場合はスキップ
                if os.path.exists(compressed_file_path):
                    continue

                print_target = target_name if target_name != "no_target" else "(no target)"
                print(f"[{package_name}/{date_str}/{print_target}] ダウンロード＆圧縮中 [{file_name}.gz]")
                
                os.makedirs(local_save_dir, exist_ok=True)
                
                content = blob.download_as_bytes()
                with gzip.open(compressed_file_path, 'wb') as f_gz:
                    f_gz.write(content)
                
                package_local_download_count += 1

            except Exception as download_err:
                print(f"[{package_name}] エラー: ファイル処理中にエラーが発生しました ({blob.name}): {download_err}")
                # エラーが発生した場合、中途半端に作成されたファイルを削除する
                if 'compressed_file_path' in locals() and os.path.exists(compressed_file_path):
                    try:
                        os.remove(compressed_file_path)
                    except OSError:
                        pass # 削除失敗は無視
                continue

        if package_local_download_count > 0:
            print(f"[{package_name}] 完了: 合計 {package_local_download_count} 個の新規ファイルを圧縮・保存しました。")
        else:
            print(f"[{package_name}] 完了: 新規にダウンロードする対象ファイルはありませんでした（調査済みまたは期間外）。")
            
        return package_local_download_count

    if not unique_package_names:  # フィルタリングの結果、処理対象がなければここでメッセージを出して終了
        print("処理対象となるパッケージ名がありません。スクリプトを終了します。")
        return  # 関数を抜ける

    print(f"\n--- {len(unique_package_names)} 個のパッケージの並列処理を開始 ---")
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pkg_name in sorted(list(unique_package_names)):
            futures.append(executor.submit(process_package, pkg_name))

        for future in concurrent.futures.as_completed(futures):
            try:
                downloaded_count_for_package = future.result()
                total_downloaded_counter.increment(
                    downloaded_count_for_package)
            except Exception as exc:
                print(f"エラー: 並列処理タスクの実行中に予期せぬエラーが発生しました: {exc}")

    print(f"\n--- すべての処理が完了しました ---")
    print(f"合計 {total_downloaded_counter.get_value()} 個のファイルを圧縮して保存しました。")
    print(f"データは {save_directory} に保存されています。")


# --- スクリプト実行部分 ---
if __name__ == "__main__":
    import argparse

    # Resolve repo root (this file is under <repo>/vuljit/download_gcs)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, '..'))

    parser = argparse.ArgumentParser(description="Download oss-fuzz coverage report files (gz) from GCS over a period.")
    parser.add_argument('--csv', dest='csv_file_path', default=os.environ.get('VULJIT_VUL_CSV', os.path.join(repo_root, 'data', 'oss_fuzz_vulns_0802.csv')),
                        help='Path to CSV file which contains package_name column (default: env VULJIT_VUL_CSV or vuljit/data/oss_fuzz_vulns_0802.csv)')
    parser.add_argument('--start', dest='start_date', default=os.environ.get('VULJIT_START_DATE', '20160101'),
                        help='Start date YYYYMMDD (default: env VULJIT_START_DATE or 20160101)')
    parser.add_argument('--end', dest='end_date', default=os.environ.get('VULJIT_END_DATE', '20250802'),
                        help='End date YYYYMMDD (default: env VULJIT_END_DATE or 20250802)')
    parser.add_argument('--out', dest='dest_dir', default=os.environ.get('VULJIT_COVERAGE_DIR', os.path.join(repo_root, 'data', 'coverage_gz')),
                        help='Destination root directory (default: env VULJIT_COVERAGE_DIR or vuljit/data/coverage_gz)')
    parser.add_argument('--workers', dest='num_workers', type=int, default=int(os.environ.get('VULJIT_WORKERS', '8')),
                        help='Parallel workers (default: env VULJIT_WORKERS or 8)')
    parser.add_argument('--file', dest='files', action='append', default=None,
                        help='Target file name to download (repeatable), e.g., --file summary.json')
    parser.add_argument('--files', dest='files_csv', default=os.environ.get('VULJIT_COVERAGE_FILES', 'summary.json'),
                        help='Comma separated target file names (used if --file not provided).')

    args = parser.parse_args()

    # Target files resolution
    if args.files:
        target_files_to_download = args.files
    else:
        target_files_to_download = [s.strip() for s in (args.files_csv or '').split(',') if s.strip()]
        if not target_files_to_download:
            target_files_to_download = ['summary.json']

    # Read CSV content
    try:
        with open(args.csv_file_path, 'r', encoding='utf-8-sig') as f:
            csv_input_string = f.read()
        print(f"CSVファイルを読み込みました: {args.csv_file_path}")
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {args.csv_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        sys.exit(1)

    download_target_files_from_gcs_period(
        csv_input_string,
        args.start_date,
        args.end_date,
        target_files_to_download,
        args.dest_dir,
        max_workers=args.num_workers,
    )

    print("\nスクリプトの実行が終了しました。")
