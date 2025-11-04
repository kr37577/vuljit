import csv
from google.cloud import storage
import os
import shutil  
from datetime import datetime
import concurrent.futures
import threading

# スレッドセーフなカウンタクラス (変更なし)


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


# ★★★ ZIP機能に関する引数を追加 ★★★
def download_and_zip_reports_from_gcs(csv_data, save_directory, zip_output_directory, max_workers=10, cleanup_after_zip=False):
    """
    CSVからpackage_nameを読み込み、GCSからcovreportファイルをダウンロードする。
    ★★★ プロジェクトごとにダウンロードが完了したら、その内容をZIPファイルに圧縮する。 ★★★

    Args:
        csv_data (str): CSV形式の文字列データ。
        save_directory (str): ダウンロードファイルの一時保存先ルートディレクトリ。
        zip_output_directory (str): 作成したZIPファイルの保存先ディレクトリ。
        max_workers (int): 並列ダウンロードに使用する最大ワーカースレッド数。
        cleanup_after_zip (bool): Trueの場合、ZIP化後に元ディレクトリを削除する。
    """
    # 各ディレクトリが存在しない場合は作成
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(zip_output_directory, exist_ok=True)  # ★★★ ZIP保存先ディレクトリも作成 ★★★

    print("ダウンロード対象ファイル (textcov_reports/{date} ディレクトリ内): *.covreport")
    print("期間指定なし: 利用可能なすべての textcov_reports をダウンロードします。")
    print(f"最大並列ワーカー数: {max_workers}")
    print(f"ZIPファイル保存先: {zip_output_directory}")
    print(f"ZIP化後の元ディレクトリ削除: {'はい' if cleanup_after_zip else 'いいえ'}")

    # GCSクライアントを初期化 (変更なし)
    try:
        storage_client = storage.Client.create_anonymous_client()
    except Exception as e:
        print(f"エラー: GCSクライアントの初期化に失敗しました: {e}")
        return

    bucket_name = "oss-fuzz-coverage"
    try:
        bucket = storage_client.bucket(bucket_name)
    except Exception as e:
        print(f"エラー: GCSバケット '{bucket_name}' の取得に失敗しました: {e}")
        return

    # CSVから重複を除いたpackage_nameを取得 (変更なし)
    unique_package_names = set()
    try:
        if csv_data.startswith('\ufeff'):
            csv_data = csv_data[1:]
        reader = csv.reader(csv_data.strip().splitlines())
        header = next(reader)
        package_name_index = header.index('package_name')
        for row in reader:
            if len(row) > package_name_index:
                package_name = row[package_name_index].strip()
                if package_name:
                    unique_package_names.add(package_name)
    except Exception as e:
        print(f"エラー: CSVデータの読み込み中に予期せぬエラーが発生しました: {e}")
        return
    if not unique_package_names:
        print("エラー: CSVから有効なパッケージ名を抽出できませんでした。")
        return
    print(f"CSVから {len(unique_package_names)} 個の一意なパッケージ名を抽出しました。")

    total_downloaded_counter = ThreadSafeCounter()

    # --- パッケージごとの処理を並列化 ---
    def process_package(package_name):
        package_local_download_count = 0
        found_reports_for_package = False

        print(f"[{package_name}] 処理開始...")
        print(f"[{package_name}] textcov_reports ディレクトリ下のすべての日付ディレクトリを検索中...")

        # (この部分はダウンロード処理なので変更なし)
        textcov_base_prefix = f"{package_name}/textcov_reports/"
        blobs_iterator = storage_client.list_blobs(
            bucket_name, prefix=textcov_base_prefix, delimiter='/')
        date_dirs = []
        try:
            for page in blobs_iterator.pages:
                date_dirs.extend(page.prefixes)
        except Exception as e:
            print(f"[{package_name}] エラー: GCSの日付ディレクトリリスト取得中にエラーが発生しました: {e}")
            return 0
        if not date_dirs:
            print(f"[{package_name}] 情報: textcov_reports 日付ディレクトリが見つかりません。")
            return 0
        for dir_prefix in date_dirs:
            try:
                date_str = dir_prefix.rstrip('/').split('/')[-1]
                if len(date_str) == 8 and date_str.isdigit():
                    files_in_date_dir_iterator = storage_client.list_blobs(
                        bucket_name, prefix=dir_prefix)
                    for blob in files_in_date_dir_iterator:
                        if blob.name.endswith('/'):
                            continue
                        file_name = os.path.basename(blob.name)
                        if file_name.endswith('.covreport'):
                            found_reports_for_package = True
                            package_date_dir = os.path.join(
                                save_directory, package_name, date_str)
                            local_file_path = os.path.join(
                                package_date_dir, file_name)
                            try:
                                os.makedirs(package_date_dir, exist_ok=True)
                                if os.path.exists(local_file_path):
                                    continue
                                # print(f"[{package_name}/{date_str}] ダウンロード中 [{file_name}]")
                                blob.download_to_filename(local_file_path)
                                package_local_download_count += 1
                            except Exception as download_err:
                                print(
                                    f"[{package_name}/{date_str}] エラー: ファイル処理中にエラーが発生しました ({blob.name}): {download_err}")
            except Exception as date_proc_err:
                print(
                    f"[{package_name}] 警告: 日付ディレクトリ '{dir_prefix}' の処理中に予期せぬエラー: {date_proc_err}")
                continue

        # パッケージごとの結果サマリ
        if package_local_download_count > 0:
            print(
                f"[{package_name}] 完了: 合計 {package_local_download_count} 個のファイルをダウンロードしました。")
        elif found_reports_for_package:
            print(f"[{package_name}] 完了: 新規にダウンロードするファイルはありませんでした。")
        else:
            print(f"[{package_name}] 完了: '.covreport' ファイルが見つかりませんでした。")

        # ★★★ ここからZIP圧縮とクリーンアップ処理 ★★★
        if found_reports_for_package:
            source_dir_to_zip = os.path.join(save_directory, package_name)
            zip_file_base_path = os.path.join(
                zip_output_directory, package_name)

            # ZIP圧縮
            try:
                print(
                    f"[{package_name}] ダウンロードしたファイルをZIPに圧縮中 -> {zip_file_base_path}.zip")
                shutil.make_archive(zip_file_base_path,
                                    'zip', source_dir_to_zip)
                print(f"[{package_name}] 圧縮完了。")

                # クリーンアップ
                if cleanup_after_zip:
                    try:
                        print(
                            f"[{package_name}] 圧縮元のディレクトリを削除中: {source_dir_to_zip}")
                        shutil.rmtree(source_dir_to_zip)
                        print(f"[{package_name}] ディレクトリを削除しました。")
                    except Exception as e:
                        print(f"[{package_name}] エラー: ディレクトリの削除に失敗しました: {e}")

            except Exception as e:
                print(f"[{package_name}] エラー: ZIPファイルの作成に失敗しました: {e}")
        # ★★★ ZIP処理ここまで ★★★

        return package_local_download_count

    # --- ThreadPoolExecutorを使用してパッケージ処理を並列実行 --- (変更なし)
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
    print(f"合計 {total_downloaded_counter.get_value()} 個のファイルをダウンロードしました。")
    print(f"ZIPファイルは '{zip_output_directory}' に保存されています。")


# --- 実行例 ---
if __name__ == '__main__':
    # CSVファイルのパスを指定
    csv_file_path = '/work/riku-ka/fuzz_introspector/oss_fuzz_vulns.csv'

    try:
        with open(csv_file_path, 'r', encoding='utf-8-sig') as f:
            csv_input_string = f.read()
    except FileNotFoundError:
        print(f"エラー: CSVファイルが見つかりません: {csv_file_path}")
        exit()
    except Exception as e:
        print(f"エラー: CSVファイルの読み込み中にエラーが発生しました: {e}")
        exit()

    # --- 設定 ---
    # ダウンロードファイルの一時保存先ディレクトリ
    # cleanup_after_zip=Trueの場合、この中身は最終的に空になります。
    destination_directory = './ossfuzz_downloaded_textcov'

    # ★★★ 作成したZIPファイルの保存先ディレクトリ ★★★
    zip_directory = './ossfuzz_zips'

    # ★★★ ZIP化後に元ディレクトリを削除するかどうか (True/False) ★★★
    # Falseにすると、ダウンロードしたファイルとZIPファイルの両方が残ります。
    # Trueにすると、ZIPファイルのみが残ります。
    cleanup = False

    # 並列処理のワーカー数を指定 (CPUコア数を自動設定)
    num_workers = 10

    # --- 関数実行 ---
    download_and_zip_reports_from_gcs(
        csv_input_string,
        destination_directory,
        zip_directory,  # ★★★ 引数を追加 ★★★
        max_workers=num_workers,
        cleanup_after_zip=cleanup  # ★★★ 引数を追加 ★★★
    )
