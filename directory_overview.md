# vuljit Directory Overview

- リポジトリ位置: `/work/riku-ka/vuljit`（OSS-Fuzz 脆弱性研究向けのデータ取得・特徴抽出・モデリング・解析ツール群）
- キャッシュ/履歴系（`.git/`, `__pycache__/`, `.pytest_cache/`）は省略。

## Root
- `.env.example` — 環境変数ひな型。データパス、GCS/GitHub 認証、モデル設定を指定。`cp .env.example .env` で複製して値を埋める。
- `datasets/` — データ置き場（raw/derived/model_outputs/statistics）。スクリプトの既定参照先。
- `analysis/` — 研究用解析コード（RQ1/2/3、統計、pytest テスト）。
- `scripts/` — データ取得・メトリクス抽出・モデリング・プロジェクト情報取得・ユーティリティの実行スクリプト群。

## datasets/
- `raw/` — 取得データ（`coverage_report/`, `srcmap_json/`, cloned repos）。data_acquisition が生成。
- `derived_artifacts/` — 加工成果物（`commit_metrics/`, `coverage_metrics/`, `patch_coverage_metrics/`, `vulnerability_reports/`, `aggregate/` 日次集約）。
- `model_outputs/` — モデル別・プロジェクト別の結果（メトリクス、特徴量重要度、予測付き CSV）。
- `statistics/` — 集計・可視化用の出力置き場。

## scripts/
### data_acquisition
- `ossfuzz_vulnerability_issue_report_extraction.py` — OSV アーカイブから OSS-Fuzz 脆弱性 CSV を生成。`--out` / `VULJIT_VUL_CSV` で出力先指定。
- `coverage_download_reports.py` — GCS カバレッジレポートを日付別取得。`--out` / `VULJIT_COVERAGE_DIR`。`coverage_download_reports.sh` は Slurm ラッパ（ログ: `scripts/data_acquisition/{logs,errors}/`）。
- `download_srcmap.py` — coverage source-map JSON を取得。`--dir` / `VULJIT_SRCDOWN_DIR`。`download_srcmap.sh` は Slurm ラッパ。

### metric_extraction
- `build_commit_metrics_pipeline.py` — Git コミットから特徴量＋脆弱ラベルを生成し CSV 化。出力 `derived_artifacts/commit_metrics/`。`cluster.sh` は Slurm ラッパ（使い方はシェルコメント参照）。
- `coverage_aggregation/process_coverage_project.py` — coverage JSON を集計し CSV を生成。`run_process_coverage_project.sh` は単体/全件処理、`cluster_coverage.sh` は Slurm 多重実行（シェルに例あり）。
- `patch_coverage_pipeline/prepare_patch_coverage_inputs.py` — srcmap から revisions CSV を作成しコミット日時付与。`prepare_patch_coverage_inputs.sh` はデフォルトパス付きラッパ。
- `patch_coverage_pipeline/create_daily_diff.py` — コミット間差分とパッチを日付ごとに生成。`calculate_patch_coverage_per_project.py` / `run_culculate_patch_coverage_pipeline.py` — 差分と coverage HTML を突き合わせパッチカバレッジ CSV を作成。`submit_patch_coverage_jobs.sh` などのジョブラッパはシェルコメント参照。
- `text_code_metrics/vccfinder_metrics_calculator.py` ほか — テキスト/コード特徴量抽出。`repo_commit_processor_test.py` は処理テスト。`いらないかも/` 配下は未使用。
- `update_labels.sh`, `run_update_labels_all.sh` — ラベル更新用ヘルパー。

### modeling
- Python モジュール:
  - `settings.py` — データ/結果パス、乱数種、閾値など全体設定。
  - `model_definition.py` — RandomForest/XGBoost/ランダムベースラインのパイプラインとハイパーパラメータ空間。
  - `data_preparation.py` — 集約済み特徴量のロードと前処理。
  - `cross_project_data.py` — クロスプロジェクト用データ分割/準備。
  - `aggregate_metrics_pipeline.py` — メトリクス＋カバレッジを結合し日次集約 CSV を作成。
  - `main_per_project.py` — プロジェクト単位の学習・評価エントリ。
  - `evaluation.py` — 交差検証やモデル保存処理。
  - `reporting.py` — 結果整形と出力。
- Shell エントリポイント（詳細はシェル冒頭コメントの使用例を参照）:
  - `aggregate_metrics_pipeline.sh` — 集約パイプラインを Slurm で一括実行。
  - `prediction_cross.sh` — 1 プロジェクトのクロスプロジェクト予測（例: `bash scripts/modeling/prediction_cross.sh -p libxml2`）。
  - `prediction_cross_batch.sh` — 複数プロジェクトをバッチ投入。
  - `predict_one_project.sh` — 単一プロジェクトの通常予測。
  - `バッグアップ/` 配下 — 過去の実験用シェル/スクリプト（現行では非推奨）。
- 結果は既定で `datasets/model_outputs/<model>/<project>/` に出力。

### project_mapping
- `clone_and_run_projects.sh` — プロジェクトをクローンし処理を実行するユーティリティ（使用例はシェルコメント参照）。
- `oss_fuzz_project_info.py` — OSS-Fuzz プロジェクト情報取得。
- `いらないかも/` 配下は未使用の旧マッピングスクリプト。

### utilities
- `zip.sh`, `single_zip.sh` — データ/成果物を ZIP 化する簡易ユーティリティ（使い方はシェルコメント参照）。

### scripts/いらないかも (unused/archived)
- 未使用の実験用スクリプト置き場。`orchestration/`（Dockerfile, CLI など）やその他試行錯誤スクリプトが含まれるが現行パイプラインでは利用しない。

## analysis/
### research_question1_2
- `analyze_comparison.py`, `analyze_trends_comparison.py`, `a.sh` — RQ1/2 向け比較・トレンド分析。

### research_question3
- Core: `core/`（io, metrics, baseline, scheduling, simulation, plotting, predictions, timeline）。
- CLI: `cli/*.py`（タイムラインや追加ビルドシミュレーションを CLI 化）。
- ワークフロー: `run_all.sh`, `run_prepare_RQ3.sh`, `rq3.sh`, `rq3_cluster.sh`, `minimal_simulation_wrapper.py`, `combine_strategy4_modes.py`, `measure_detection_time.py`, `threshold_precision_analysis.py`, `extract_build_counts.py`, `simulate_additional_builds.py`, `additional_build_strategies.py`, `analysis/generate_strategy_table.py`, `rq3_result.py`。
- Tests: `tests/` 配下の pytest スイート（`TEST_CATALOG.md` 参照）。
- Legacy: `いらないかも/` は未使用のアイデア置き場。

### statistic
- `fill_dataset_summary.py` と旧サマリースクリプト（`いらないかも/` は未使用）。

## Other notes
- Git メタデータあり（`.git/`）。Pytest キャッシュあり（`.pytest_cache/`）。
- ルートに README/Plan は別途作成予定（複製パッケージ用）。
