# scripts配下のデータ保存先まとめ

この資料は `vuljit/scripts` 内の代表的なスクリプトがどこにデータを保存するかを整理したものです。各パイプラインは環境変数で出力先を上書きできる設計になっており、未設定の場合はリポジトリ内の `datasets/` 系ディレクトリに保存されます。

## data_acquisition 配下スクリプトの出力先

### coverage_download_reports.py
- **役割**: OSS-Fuzz の GCS カバレッジレポートから `summary.json` など指定ファイルを取得し、パッケージ/日付/ターゲットごとの階層で保存する。
- **既定の保存先**: `datasets/raw/coverage_report/<package>/<YYYYMMDD>/<target|no_target>/linux/<filename>`
  - ルートは環境変数 `VULJIT_COVERAGE_DIR` または引数 `--out` で変更可能（未指定時は `datasets/raw/coverage_report`）。
- **主な入力**: `datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv`（環境変数 `VULJIT_VUL_CSV` または `--csv` で上書き可）。
- **関連環境変数**: `VULJIT_START_DATE` / `VULJIT_END_DATE`（期間指定）、`VULJIT_COVERAGE_FILES`（ダウンロード対象ファイルリスト）、`VULJIT_WORKERS`（並列数）。
- **補足**: 既に同名ファイルが存在する場合はスキップし、階層ディレクトリは自動作成される。

### coverage_download_reports.sh
- **役割**: 上記 Python スクリプトを SLURM ジョブとして実行し、必要な環境変数を設定するジョブスクリプト。
- **保存先の扱い**:
  - `VULJIT_COVERAGE_DIR` を `datasets/raw/coverage_report` に初期化してエクスポート。
  - 追加生成物として `scripts/data_acquisition/logs/` と `scripts/data_acquisition/errors/` を作成（SLURM 標準出力/標準エラー置き場）。
  - 将来的な zip 化の設定項目 (`VULJIT_COVERAGE_ZIP_DIR` → `datasets/derived_artifacts/coverage_zip`) がコメントとして残されている。

### download_srcmap.py
- **役割**: OSS-Fuzz カバレッジバケット `oss-fuzz-coverage` から `<date>.json` のソースマップを取得する。
- **既定の保存先**: `datasets/raw/srcmap_json/<package>/json/<YYYYMMDD>.json`
  - ルートは環境変数 `VULJIT_SRCDOWN_DIR` または引数 `--dir` で変更可能。
- **主な入力**: プロジェクトリスト CSV（既定は `datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv`、`VULJIT_VUL_CSV` で上書き可能）。
- **関連環境変数**: `VULJIT_START_DATE` / `VULJIT_END_DATE`（取得期間）、`VULJIT_WORKERS`（並列数）。`--csv-column` で列番号指定も可能。
- **補足**: 取得対象が存在しない場合はファイルを作成しない。パッケージ別サブディレクトリは自動作成される。

### download_srcmap.sh
- **役割**: `download_srcmap.py` を SLURM で実行するラッパースクリプト。
- **保存先の扱い**: `VULJIT_SRCDOWN_DIR` を `datasets/raw/srcmap_json` に初期化して Python 側に引き渡す。ジョブログ用に `scripts/data_acquisition/logs/` / `errors/` を利用。

### ossfuzz_vulnerability_issue_report_extraction.py
- **役割**: OSV のアーカイブ (GCS) を取得し、OSS-Fuzz 脆弱性レポートを CSV 化する。
- **既定の保存先**: `datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv`
  - 引数 `--out` または環境変数 `VULJIT_VUL_CSV` で出力パスを指定可能。ディレクトリが存在しない場合は自動作成する。
- **関連オプション**:
  - `--cache-dir` を指定するとアーカイブをローカルキャッシュし、以降の実行で再利用。
  - `VULJIT_OSV_ARCHIVE_URL` を設定するとダウンロード元 ZIP の URL を明示的に切り替えられる。
- **補足**: 出力 CSV はモノレール ID 昇順でソートされ、UTF-8 で保存される。

## metric_extraction 配下スクリプトの出力先

### build_commit_metrics_pipeline.py
- **役割**: Git リポジトリからコミットメトリクスを抽出し、脆弱性ラベルと TF-IDF 特徴量を付与して CSV 化する。
- **既定の保存先**: `datasets/derived_artifacts/commit_metrics/<project>/<project>_commit_metrics_with_tfidf.csv`
  - ルートは `--metrics-dir` 引数または `VULJIT_METRICS_DIR` で変更可能。ディレクトリは自動作成。
- **主な入力**: 脆弱性 CSV (`datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv`、`VULJIT_VUL_CSV` で上書き可) とローカル Git リポジトリ。
- **補足**: 既存ファイルがある場合は未処理コミットのみを追記し、`--force` で再計算。ラベル列は `NEW_LABEL_COLUMN` の有無で管理。

### cluster.sh
- **役割**: `build_commit_metrics_pipeline.py` を SLURM で実行するジョブスクリプト。
- **保存先の扱い**: 既定で `datasets/derived_artifacts/commit_metrics/` に保存。ジョブログは `scripts/metric_extraction/logs/` / `errors/` に出力。
- **補足**: `--since`/`--until` のデフォルトをオプションで差し込む仕組みがある。

### coverage_aggregation/process_coverage_project.py
- **役割**: OSS-Fuzz カバレッジ JSON を集計し、ファイル単位 (`*_and_date.csv`) と日次トータル (`*_total_and_date.csv`) を生成。
- **既定の保存先**: `VULJIT_COVERAGE_METRICS_DIR/<project>/` → 未設定なら `VULJIT_BASE_DATA_DIR/coverage_metrics/<project>/`、いずれも未設定時は `datasets/coverage_metrics/<project>/`。
- **補足**: 既存 CSV が無い場合でもヘッダーのみの空ファイルを生成し、出力ディレクトリは自動で作成する。

### coverage_aggregation/run_process_coverage_project.sh
- **役割**: 単体/一括処理で `process_coverage_project.py` を呼び出すユーティリティ。
- **保存先の扱い**: `VULJIT_COVERAGE_METRICS_DIR` を `datasets/derived_artifacts/coverage_metrics` に初期化して Python に渡す。ジョブログは `coverage_aggregation/logs/` / `errors/`。
- **補足**: `--all` 指定で `datasets/raw/coverage_report` 以下の全プロジェクトを処理。

### coverage_aggregation/cluster_coverage.sh
- **役割**: 上記ラッパーを SLURM で多重実行するジョブスクリプト。
- **保存先の扱い**: `run_process_coverage_project.sh --all /work/riku-ka/vuljit/datasets/raw/coverage_report` を固定実行し、結果を `datasets/derived_artifacts/coverage_metrics/` に蓄積。ログ/エラーは `coverage_aggregation/logs/` / `errors/`。

### patch_coverage_pipeline/prepare_patch_coverage_inputs.py
- **役割**: `srcmap_json` から `revisions_<project>.csv` を生成し、Git リポジトリからコミット日時を付与して `revisions_with_commit_date_<project>.csv` として集約。
- **既定の保存先**: コミット日時付与後の成果物は `datasets/derived_artifacts/patch_coverage_inputs/`（`--commit-out` または `VULJIT_PATCH_COVERAGE_INPUTS_DIR` で変更）。
- **主な入力**:
  - `--srcmap-root` 未指定時は `datasets/raw/srcmap_json/`（`VULJIT_SRCDOWN_DIR` があればそれを優先）。
  - クローン済みリポジトリは `data/intermediate/cloned_repos/` または `VULJIT_CLONED_REPOS_DIR`。
- **補足**: `--skip-revisions` / `--skip-commit` で段階的に処理を省略でき、一時ディレクトリ利用時は処理後にクリーンアップ。

### patch_coverage_pipeline/create_project_csvs_from_srcmap.py
- **役割**: `srcmap_json` の解析から `revisions_<project>.csv` を生成する単体スクリプト。
- **既定の保存先**: `data/intermediate/patch_coverage/csv_results/`（`VULJIT_INTERMEDIATE_DIR` 経由で変更可）。
- **補足**: `prepare_patch_coverage_inputs.sh` から呼ばれる際は `datasets/derived_artifacts/patch_coverage_inputs/` へ出力を向ける。

### patch_coverage_pipeline/revision_with_date.py
- **役割**: `revisions_<project>.csv` にコミット日時を追加して `revisions_with_commit_date_<project>.csv` を作成。
- **既定の保存先**: `datasets/derived_artifacts/patch_coverage_inputs/`（`VULJIT_PATCH_COVERAGE_INPUTS_DIR`）。
- **補足**: Git リポジトリのベースは `data/intermediate/cloned_repos/`（`VULJIT_CLONED_REPOS_DIR`）を想定。

### patch_coverage_pipeline/create_daily_diff.py
- **役割**: `revisions_with_commit_date_<project>.csv` からコミット間の差分ファイルとパッチを生成。
- **既定の保存先**: `data/intermediate/patch_coverage/daily_diffs/<project>/`
  - 各日付のファイル一覧 `YYYYMMDD.csv` と対応する `YYYYMMDD_patches/` 以下の `.patch` 群を出力。
  - ルートは `--out` または `VULJIT_PATCH_COVERAGE_DIFFS_DIR` で変更可能。
- **主な入力**: `datasets/derived_artifacts/patch_coverage_inputs/`（`--input` / `VULJIT_PATCH_COVERAGE_INPUTS_DIR`）、`datasets/intermediate/cloned_repos/`（`--repos` / `VULJIT_CLONED_REPOS_DIR`）。
- **補足**: 途中再開に備えて `.progress` と既存ファイルの検出ロジックを持つ。

### patch_coverage_pipeline/calculate_patch_coverage_per_project.py
- **役割**: 日別差分と GCS カバレッジ HTML を突き合わせ、追加行のカバレッジ率を算出して CSV に蓄積。
- **既定の保存先**: `datasets/derived_artifacts/patch_coverage_metrics/<project>/<project>_patch_coverage.csv`
  - ルートは `--out` または `VULJIT_PATCH_COVERAGE_RESULTS_DIR` で変更可。
- **解析ログ**: HTML 解析のキャッシュ JSON は `data/intermediate/patch_coverage/parsing_results/`（`--parsing-out` / `VULJIT_PATCH_COVERAGE_PARSING_DIR`）に保存。
- **補足**: 既存 CSV を読み込んで日付単位で追記し、GCS は匿名クライアントでアクセスする。

### patch_coverage_pipeline/run_culculate_patch_coverage_pipeline.py
- **役割**: `revisions_with_commit_date_<project>.csv` を直接読み込み、Git 差分と GCS カバレッジを突き合わせて最終 CSV を生成するインメモリ版パイプライン。
- **既定の保存先**: `datasets/derived_artifacts/patch_coverage_metrics/<project>_patch_coverage.csv`（`--coverage-out` / `VULJIT_PATCH_COVERAGE_RESULTS_DIR`）。
- **補足**: `--parsing-out` 指定時は HTML 解析 JSON を併せて保存。並列数は `--workers` で調整。

### patch_coverage_pipeline/prepare_patch_coverage_inputs.sh
- **役割**: 上記 Python パイプラインを既定のリポジトリ構成で実行。
- **保存先の扱い**: `DEFAULT_COMMIT_OUT` を `datasets/derived_artifacts/patch_coverage_inputs` に設定し、必要に応じて `DEFAULT_SRCMAP_ROOT`（`datasets/raw/srcmap_json`）と `DEFAULT_REPOS`（`data/intermediate/cloned_repos`）を明示。

### patch_coverage_pipeline/run_shell_for_patch_projects.sh
- **役割**: `revisions_with_commit_date_<project>.csv` を列挙し、プロジェクトごとに任意のシェルスクリプト（例: `submit_patch_coverage_jobs.sh`）を実行。
- **保存先の扱い**: 入力既定は `datasets/derived_artifacts/patch_coverage_inputs/`。指定スクリプト側で生成される `logs/` / `errors/` を利用する運用を前提としている。

### patch_coverage_pipeline/submit_patch_coverage_jobs.sh
- **役割**: `run_culculate_patch_coverage_pipeline.py` を SLURM ジョブとして起動。
- **保存先の扱い**: 結果は Python 側のデフォルトで `datasets/derived_artifacts/patch_coverage_metrics/` へ。ジョブログは `patch_coverage_pipeline/logs/` / `errors/`。

### patch_coverage_pipeline/calculate_patch_coverage.sh
- **役割**: 旧来のジョブアレイでパッチカバレッジを計算するスクリプト（外部ディレクトリを参照）。
- **保存先の扱い**: `/work/riku-ka/patch_coverage_culculater/daily_diffs` など外部パスを前提としており、リポジトリ内 `datasets/derived_artifacts/patch_coverage_metrics/` ではなく外部ディレクトリに書き出すレガシー構成。
- **補足**: 利用時は環境に合わせたパス修正が前提。

## modeling 配下スクリプトの出力先

### aggregate_metrics_pipeline.py
- **役割**: コミットメトリクスとカバレッジ指標を突き合わせ、日次集約特徴量 `*_daily_aggregated_metrics.csv` を生成する。
- **既定の入力**:
  - メトリクス: `--metrics` または `VULJIT_METRICS_DIR`（未指定時は `datasets/metric_inputs/<directory>/<directory>_commit_metrics_with_tfidf.csv` を探索）。
  - カバレッジ: `--coverage` または `VULJIT_COVERAGE_AGG_DIR`（未指定時は `datasets/derived_artifacts/metrics/coverage_aggregate/<project>/` 内の `*_and_date.csv` と `*_total_and_date.csv`）。
  - パッチカバレッジ: `--patch` または `VULJIT_PATCH_COV_DIR`（未指定時は `datasets/derived_artifacts/metrics/patch_coverage/<project>/<project>_patch_coverage.csv`）。
- **既定の保存先**: `--out` または `VULJIT_BASE_DATA_DIR`（未指定時は `datasets/derived_artifacts/aggregate/<project>/<project>_daily_aggregated_metrics.csv`）。出力ディレクトリはプロジェクトごとに自動作成され、少数クラス件数が `settings.MIN_SAMPLES_THRESHOLD` 未満の場合は保存をスキップする。
- **補足**: `datasets/reference_mappings/filtered_project_mapping.csv` に記載された `project_id,directory_name` のペアを想定し、ファイル内で過去 VCC 参照特徴量も付加する。

### aggregate_metrics_pipeline.sh
- **役割**: 上記 Python 集約スクリプトを SLURM で一括実行する。
- **保存先の扱い**: `VULJIT_BASE_DATA_DIR` を `/work/riku-ka/vuljit/datasets/derived_artifacts/aggregate` に固定し、ログは `scripts/modeling/logs/` と `scripts/modeling/errors/` に出力。
- **補足**: `METRICS_BASE_PATH` や `PATCH_COVERAGE_BASE_PATH` が外部ドライブにハードコードされているため、環境に応じて変更が必要。

### main_per_project.py
- **役割**: プロジェクト単位で学習・評価を実施し、評価結果と予測値を保存するメインエントリ。
- **既定の入力**: `settings.BASE_DATA_DIRECTORY`（`VULJIT_BASE_DATA_DIR` で変更可、未指定時は `datasets/`）配下の `<project>/<project>_daily_aggregated_metrics.csv`。
- **既定の保存先**: `settings.RESULTS_BASE_DIRECTORY`（`VULJIT_RESULTS_DIR` または `VULJIT_MODEL` で変更可、既定は `datasets/model_outputs/<selected_model>/<project>/`）。
  - `expX_metrics.json`, `expX_importances.csv`, `expX_per_fold_metrics.csv` をプロジェクト別に生成。
  - 予測確率を付与した `*_daily_aggregated_metrics_with_predictions.csv` を併せて保存し、`predicted_risk_<canonical>` / `predicted_label_<canonical>` 列を追加する。
- **補足**: 予測の二値化閾値は `settings.PREDICTION_LABEL_THRESHOLD`（`VULJIT_LABEL_THRESHOLD` 環境変数）で制御可能。

### evaluation.py
- **役割**: 交差検証処理とモデル保存処理を担当する内部モジュール。
- **保存先の扱い**:
  - `settings.SAVE_BEST_MODEL` が有効な場合、`settings.MODEL_OUTPUT_DIRECTORY/<project>/best_model_<run_tag>.joblib` と `model_metadata_<run_tag>.json` を作成。
  - `settings.SAVE_HYPERPARAM_RESULTS` が有効な場合、`settings.LOGS_DIRECTORY/<project>/cv_results_<run_tag>.pkl` と `metadata_<run_tag>.json` を保存。
- **補足**: 両フラグは既定で `False` のため、オンにする際は出力ディレクトリの容量と管理方法に注意する。

### settings.py
- **役割**: モデリング全体で使用するパスと実験設定を集約。
- **保存関連の主な設定**:
  - `VULJIT_BASE_DATA_DIR` で学習データの search ルート、`VULJIT_RESULTS_DIR` および `VULJIT_MODEL` で結果出力先を切り替え。
  - `MODEL_OUTPUT_DIRECTORY` / `LOGS_DIRECTORY` は前述の保存フラグに連動。
- **補足**: `SELECTED_MODEL` の値に応じて `datasets/model_outputs/<model>/` 以下のサブディレクトリが自動で変わる。

### predict_one_project.sh
- **役割**: orchestration の CLI (`scripts/orchestration/cli.py`) を介して単一プロジェクトのトレーニングを実行するための SLURM スクリプト。
- **保存先の扱い**: `VULJIT_BASE_DATA_DIR` を未設定時に `/work/riku-ka/vuljit/datasets/derived_artifacts/aggregate` へ初期化し、ジョブログは `scripts/modeling/logs/` / `errors/` に保存。
- **補足**: 実際のモデルや結果は CLI 側で `settings.py` の指定に従って保存される。

## project_mapping 配下スクリプトの出力先

### oss_fuzz_project_info.py
- **役割**: `oss-fuzz` リポジトリの `projects/<name>/project.yaml` から C/C++ プロジェクトのメタデータを収集する。
- **既定の入力/取得先**:
  - リポジトリ: `datasets/raw/oss-fuzz`（`--dest` または `VULJIT_OSSFUZZ_DEST` 相当で変更可、未存在時は `DEFAULT_REPO_URL` から clone）。
  - 脆弱性 CSV: `datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv`（`--vuln-csv` で差し替え可能）。
- **既定の保存先**:
  - メタデータ: `datasets/derived_artifacts/oss_fuzz_metadata/oss_fuzz_project_metadata.csv` に追記（`--out` で変更）。ディレクトリは自動生成し、ヘッダは初回のみ書き出す。
  - `--summarize-vulns` 指定時は `datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv`（`--summary-out` 変更可）へ脆弱性数サマリを保存。
- **補足**: `--count-unique-days` を併用するとコミットと日付の統計を計算するが、ファイルには書き出さず標準出力に表示する。

### clone_and_run_projects.sh
- **役割**: 上記サマリ CSV を基に C/C++ プロジェクトをローカル clone/更新し、任意コマンドをプロジェクト単位で実行する。
- **既定の入力**: `datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv`（`-c` で指定）。
- **既定の保存先**: `datasets/raw/cloned_c_cpp_projects/<project>/` に対象リポジトリを clone。既存なら `git fetch --all --prune` で更新。
- **コマンド実行**:
  - `-r`（runner）指定で `<runner> <project> <repo> [args...]` を直接実行。
  - `-C`（command template）指定で `{project}`, `{repo}`, `{runner}`, `{runner_args}` プレースホルダを展開し `bash -c` で実行。
  - `--since` / `--until` オプションは runner 引数に自動補完される（既に指定されていれば挿入しない）。
- **補足**: ランナーやテンプレートが生成するログ/成果物は各コマンドに依存するため、本スクリプト側では管理しない。
