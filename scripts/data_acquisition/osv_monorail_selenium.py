import time
import re
import csv
import random
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

# --- 設定 ---
# プロジェクトルートを基点にパスを計算（相対パス運用）
BASE_DIR = Path(__file__).resolve().parents[2]
data_dir = BASE_DIR / "datasets" / "derived_artifacts" / "vulnerability_reports"
raw_data_dir = BASE_DIR / "datasets" / "raw"
redirect_dir = BASE_DIR / "datasets" / "derived_artifacts" / "issue_redirect_mapping_selenium"

# ★★★ 読み込むCSVファイル名を指定してください ★★★
input_csv_filename = data_dir / "oss_fuzz_vulnerabilities.csv"

# HTMLファイルを保存するディレクトリ名（Rawデータ格納ディレクトリ）
output_dir_html = raw_data_dir

# イシュー番号とリダイレクト先URLのマッピングを保存するCSVファイル名
output_csv_redirect_mapping = redirect_dir / "issue_redirect_mapping_selenium.csv"

# アクセス間隔（秒） - サーバー負荷軽減のため
min_sleep_time = 3  # 最短待機時間
max_sleep_time = 7  # 最長待機時間 (Seleniumは動作が遅めなので少し長めに)

# Selenium WebDriver設定
chrome_options = ChromeOptions()
chrome_options.add_argument("--headless")  # ヘッドレスモード（ブラウザ表示なし）で実行する場合
chrome_options.add_argument("--disable-gpu")  # ヘッドレスモードで推奨
chrome_options.add_argument("--no-sandbox")  # サンドボックス無効（環境による）
chrome_options.add_argument("--disable-dev-shm-usage")  # /dev/shm使用無効（環境による）
chrome_options.add_argument(
    # UserAgent設定

    'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

# ページの読み込みタイムアウト（秒）
page_load_timeout = 45  # 長めに設定
# 要素が表示されるまでの最大待機時間（秒）
element_wait_timeout = 30  # 長めに設定

# --- WebDriverの初期化 ---
try:
    print("WebDriverを初期化中...")
    # webdriver-manager を使ってChromeDriverのパスを自動取得・設定
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(page_load_timeout)
    print("WebDriverの初期化完了。")
except WebDriverException as e:
    print(f"エラー: WebDriverの初期化に失敗しました。{e}")
    print("ChromeDriverがインストールされているか、またはパスが通っているか確認してください。")
    print("webdriver-managerが正しく動作しない場合は、手動でWebDriverのパスを指定する必要があります。")
    exit()
except Exception as e:
    print(f"エラー: WebDriverの初期化中に予期せぬエラーが発生しました。{e}")
    exit()


# --- HTML保存先ディレクトリを作成 ---
try:
    output_dir_html.mkdir(parents=True, exist_ok=True)
    print(f"HTMLファイルは '{output_dir_html}' ディレクトリに保存されます。")
except OSError as e:
    print(f"エラー: ディレクトリ '{output_dir_html}' の作成に失敗しました。{e}")
    driver.quit()
    exit()

# --- リダイレクトマッピングCSVファイルの準備 ---
try:
    output_csv_redirect_mapping.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not output_csv_redirect_mapping.exists()) or (
        output_csv_redirect_mapping.stat().st_size == 0
    )
    with output_csv_redirect_mapping.open('a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['issue_id', 'final_url',
                      'reported_date']  # reported_date を追加
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
    print(f"イシュー番号と最終URLのマッピングは '{output_csv_redirect_mapping}' に保存/追記されます。")
except IOError as e:
    print(f"エラー: CSVファイル '{output_csv_redirect_mapping}' の準備中にエラーが発生しました。{e}")
    driver.quit()
    exit()

# --- 既に処理済みのIDを読み込む ---
processed_ids = set()
if output_csv_redirect_mapping.exists():
    with output_csv_redirect_mapping.open('r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                processed_ids.add(int(row['issue_id']))
            except (KeyError, ValueError):
                continue

# --- 入力CSVファイルを読み込み、各IDを処理 ---
processed_ids_count = 0
try:
    with input_csv_filename.open('r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        if 'monorail_id' not in reader.fieldnames:
            print(
                f"エラー: 入力CSVファイル '{input_csv_filename}' に 'monorail_id' 列が見つかりません。")
            driver.quit()
            exit()

        print(
            f"入力CSVファイル '{input_csv_filename}' から 'monorail_id' を読み込んで処理を開始します...")

        for row in reader:
            issue_id_str = row.get('monorail_id')
            if not issue_id_str or not issue_id_str.strip():
                print("  -> スキップ: 'monorail_id' が空です。")
                continue

            issue_id = int(issue_id_str.strip())
            # --- 既に処理済みのIDをスキップ ---
            if issue_id in processed_ids:
                print(f"  -> スキップ: Issue ID {issue_id} は既に処理済みです。")
                continue

            initial_url = f'https://bugs.chromium.org/p/oss-fuzz/issues/detail?id={issue_id}'
            print(f"Processing Issue ID: {issue_id} (URL: {initial_url})")

            try:
                # --- Seleniumでページにアクセス ---
                driver.get(initial_url)

                # --- リダイレクト後のページの特定の要素が表示されるまで待機 ---
                # 例として、Issue Trackerページの主要コンテナ要素を待つ
                # 注意: このセレクタは実際のページの構造に合わせて調整が必要な場合があります
                # 開発者ツール (F12) で確認してください
                wait = WebDriverWait(driver, element_wait_timeout)
                # issue-page-container や issue-title-header など、確実に表示される要素を指定
                # 例1: issues.oss-fuzz.com の主要なコンテナ要素 (もし存在すれば)
                # wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "b-issue-page")))
                # 例2: タイトルが表示される要素 (CSSクラスは仮)
                wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "body > header > div.top-bar-wrapper > b-top-bar-outlet > b-app-top-bar > b-app-logo > app-logo > a > h1")))
                # 必要に応じて他の要素や条件で待機を追加
                # time.sleep(2) # 必要であれば追加の固定待機

                print(f"  -> ページ読み込み完了 (待機要素確認済)")

                # --- 最終的なURLを取得 ---
                final_url = driver.current_url
                print(f"  -> Final URL: {final_url}")

                # --- Descriptionの投稿日時をUTCで取得 ---
                try:
                    # Descriptionセクション(#comment1)内のtime要素からdatetime属性を取得
                    time_element_selector = "#comment1 b-formatted-date-time time"
                    time_element = WebDriverWait(driver, element_wait_timeout).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, time_element_selector))
                    )
                    reported_date = time_element.get_attribute('datetime')

                except Exception as e:
                    reported_date = "N/A"
                    print(
                        f"  -> 警告: Descriptionの投稿日時(UTC)の取得に失敗しました (Issue ID: {issue_id}): {e}")

                # --- CSVにマッピングを記録 ---
                try:
                    with output_csv_redirect_mapping.open('a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=[
                                                'issue_id', 'final_url', 'reported_date'])
                        writer.writerow(
                            {'issue_id': issue_id, 'final_url': final_url, 'reported_date': reported_date})
                except IOError as e:
                    print(
                        f"  -> エラー: リダイレクトマッピングCSV '{output_csv_redirect_mapping}' への書き込み中にエラー: {e}")

                # --- HTMLを取得して保存 ---
                html_content = driver.page_source
                file_name = f'issue_{issue_id}.html'
                file_path = output_dir_html / file_name
                try:
                    with file_path.open('w', encoding='utf-8') as f:  # エンコーディングはUTF-8推奨
                        f.write(html_content)
                    print(f"    -> HTMLを保存しました: {file_path}")
                    processed_ids_count += 1
                except IOError as e:
                    print(f"    -> エラー: ファイル '{file_path}' の書き込み中にエラー: {e}")
                except Exception as e:
                    print(
                        f"    -> エラー: ファイル '{file_path}' の書き込み中に予期せぬエラー: {e}")

            # --- Selenium処理のエラーハンドリング ---
            except TimeoutException:
                print(
                    f"  -> エラー: ページの読み込みまたは要素の待機中にタイムアウトしました (Issue ID: {issue_id}, URL: {initial_url} -> {driver.current_url if 'driver' in locals() and driver else 'N/A'})")
            except NoSuchElementException:
                print(
                    f"  -> エラー: ページ内で待機対象の要素が見つかりませんでした (Issue ID: {issue_id}, URL: {driver.current_url if 'driver' in locals() and driver else 'N/A'})")
            except WebDriverException as e:
                print(
                    f"  -> エラー: WebDriverエラーが発生しました (Issue ID: {issue_id}): {e}")
                # 必要であればここで処理を中断またはWebDriverを再起動するロジックを追加
            except Exception as e:
                print(f"  -> エラー: 予期せぬエラーが発生しました (Issue ID: {issue_id}): {e}")

            # --- 次のリクエストまでの待機 ---
            sleep_time = random.randint(min_sleep_time, max_sleep_time)
            print(f"    ... {sleep_time}秒待機 ...")
            time.sleep(sleep_time)

except FileNotFoundError:
    print(f"エラー: 入力CSVファイル '{input_csv_filename}' が見つかりません。")
except IOError as e:
    print(f"エラー: 入力CSVファイル '{input_csv_filename}' の読み込み中にエラーが発生しました。{e}")
except Exception as e:
    print(f"エラー: CSVファイルの処理中に予期せぬエラーが発生しました: {e}")
finally:
    # --- WebDriverを終了 ---
    if 'driver' in locals() and driver:
        driver.quit()
        print("WebDriverを終了しました。")

print(f"\n処理完了。{processed_ids_count} 件のイシューのHTMLを取得しました。")
