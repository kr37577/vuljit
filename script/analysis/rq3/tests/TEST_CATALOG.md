# RQ3 Test Catalog (Function-Level Detail)

このドキュメントでは、`pytest` 実行時に収集される各テスト関数について以下の観点でまとめています。

- **対象関数**: テストが直接/間接にカバーするプロダクション側の関数。
- **入力**: テスト内で与えている主な引数やフィクスチャ。
- **想定出力 / 振る舞い**: テストが期待するリターン値や副作用。
- **テスト内容**: テストコード内で具体的に検証している assertion の観点。

---

## `tests/test_core_baseline.py`

### `test_baseline_detection_metrics_computes_derived_columns`
- **対象関数**: `RQ3.core.baseline.baseline_detection_metrics`
- **入力**: 検出時間を 2 件含む `detection_df`、ビルド頻度を含む `build_counts_df`
- **想定出力**: ベースラインの中央値 (`baseline_detection_days`)、ビルド頻度 (`builds_per_day`)、積算ビルド数 (`baseline_detection_builds`)
- **テスト内容**: プロジェクト "alpha" の行を抽出し、中央値→1.5倍の積算ビルド値が 4.5 になることを検証

### `test_build_threshold_map_prefers_positive_candidates`
- **対象関数**: `RQ3.core.baseline.build_threshold_map`
- **入力**: 各プロジェクトで `baseline_detection_builds`/`baseline_detection_days`/`builds_per_day` に欠損を含む DataFrame
- **想定出力**: プロジェクトごとの閾値辞書 (`Dict[str, float]`)
- **テスト内容**:  
  - 既に `baseline_detection_build_number` がある場合はその値を採用  
  - days×builds で補完されるケース  
  - 適切な候補が無いケースでは `inf`

---

## `tests/test_core_io.py`

### `test_ensure_directory_creates_path`
- **対象関数**: `RQ3.core.io.ensure_directory`
- **入力**: `tmp_path` を使って存在しない多段ディレクトリを指定
- **想定出力**: 正規化されたディレクトリパス文字列
- **テスト内容**: 戻り値を `Path` 化し、実際にディレクトリが作成されることを確認

### `test_ensure_parent_directory_creates_parent`
- **対象関数**: `RQ3.core.io.ensure_parent_directory`
- **入力**: `tmp_path / "outputs" / "result.csv"`
- **想定出力**: ファイルパスの絶対文字列
- **テスト内容**: 親ディレクトリ (`outputs/`) が生成されているか確認

### `test_parse_build_counts_csv_reads_mapping`
- **対象関数**: `RQ3.core.io.parse_build_counts_csv`
- **入力**: `builds_per_day` 列を 2.2 / 3.9 とした CSV
- **想定出力**: `{"alpha": 2, "beta": 4}`
- **テスト内容**: 四捨五入＋下限 0 が働くことを検証

---

## `tests/test_core_metrics.py`

### `test_prepare_project_metrics_merges_baseline`
- **対象関数**: `RQ3.core.metrics.prepare_project_metrics`
- **入力**: 戦略 2 件のスケジュール辞書、ベースライン DataFrame
- **想定出力**: プロジェクト×戦略の DataFrame（トリガ件数、日数、積算値などを含む）
- **テスト内容**: "strategy1"+"alpha" の行を取り出し、`trigger_count=2`, `scheduled_builds=3`, `baseline_detection_builds=4` を検証

### `test_aggregate_strategy_metrics_summarises_values`
- **対象関数**: `RQ3.core.metrics.aggregate_strategy_metrics`
- **入力**: 上記の `prepare_project_metrics` 結果
- **想定出力**: 戦略単位の集計 DataFrame
- **テスト内容**: "strategy1" の行でプロジェクト数 2 件、 total_scheduled_builds=6 を確認

### `test_prepare_daily_totals_groups_by_day`
- **対象関数**: `RQ3.core.metrics.prepare_daily_totals`
- **入力**: 複数日付を含むスケジュール辞書
- **想定出力**: 戦略×プロジェクト×日付の DataFrame（追加ビルド数の合計）
- **テスト内容**: "strategy1"+"alpha" の日付 2 件が存在し、それぞれの `scheduled_additional_builds` が期待通りであることを確認

### `test_safe_ratio_handles_zero_denominator`
- **対象関数**: `RQ3.core.metrics.safe_ratio`
- **入力**: `(4, 2)` / `(1, 0)`
- **想定出力**: 2.0 / NaN
- **テスト内容**: 正常値確認とゼロ除算で NaN を返すことを検証

---

## `tests/test_core_plotting.py`

### `test_plot_additional_builds_boxplot_creates_file`
- **対象関数**: `RQ3.core.plotting.plot_additional_builds_boxplot`
- **入力**: `scheduled_builds` を含む DataFrame、出力ディレクトリ
- **想定出力**: `additional_builds_boxplot.png` のパス
- **テスト内容**: Matplotlib が利用可能な場合、関数がファイルを生成できるか（skip 条件付き）

### `test_plot_additional_builds_boxplot_returns_none_without_backend`
- **対象関数**: 同上
- **入力**: 空 DataFrame、`plotting.plt=None` に monkeypatch
- **想定出力**: `None`
- **テスト内容**: 描画バックエンドが無い環境で安全に終了することを確認

---

## `tests/test_core_predictions.py`

### `test_iter_prediction_files_finds_nested_csv`
- **対象関数**: `RQ3.core.predictions.iter_prediction_files`
- **入力**: サブディレクトリ内に `_daily_aggregated_metrics_with_predictions.csv` を配置
- **想定出力**: ファイルパスのイテレータ
- **テスト内容**: 発見したファイルが 1 件で、拡張子が定義どおりであること

### `test_load_project_predictions_filters_columns`
- **対象関数**: `RQ3.core.predictions.load_project_predictions`
- **入力**: 必須列（merge_date, is_vcc, risk 列）を持つ CSV
- **想定出力**: プロジェクト名が付与された DataFrame
- **テスト内容**: 列集合と `project` 列の値を確認

### `test_collect_predictions_concatenates_frames`
- **対象関数**: `RQ3.core.predictions.collect_predictions`
- **入力**: 2 プロジェクト分の CSV
- **想定出力**: 結合された DataFrame
- **テスト内容**: `project` 列のユニーク数が 2 であることを検証

---

## `tests/test_core_scheduling.py`

### `test_normalize_name_accepts_alias`
- **対象関数**: `RQ3.core.scheduling.normalize_name`
- **入力**: 別名 `"median"`
- **想定出力**: 正規化された `"strategy1_median"`
- **テスト内容**: 戻り値を直接確認

### `test_run_strategy_uses_registry`
- **対象関数**: `RQ3.core.scheduling.run_strategy`
- **入力**: Registry を monkeypatch で `DataFrame` を返すダミー関数に差し替え
- **想定出力**: 返ってくる `DataFrame` が期待どおり
- **テスト内容**: イコール判定で確認

### `test_iter_strategies_returns_sorted`
- **対象関数**: `RQ3.core.scheduling.iter_strategies`
- **入力**: なし
- **想定出力**: ソート済みの (name, func) イテレータ
- **テスト内容**: 戻り値の名前リストがソート順になっているか検証

---

## `tests/test_core_simulation.py`

### `test_prepare_schedule_for_waste_analysis_normalises_dates`
- **対象関数**: `RQ3.core.simulation.prepare_schedule_for_waste_analysis`
- **入力**: `merge_date_ts` に ISO 8601 を含む DataFrame
- **想定出力**: 正規化された `schedule_date` 列と数値化された `scheduled_additional_builds`
- **テスト内容**: 1 行目の `schedule_date`／`scheduled_additional_builds` を期待値と比較

### `test_summarize_wasted_builds_classifies_events`
- **対象関数**: `RQ3.core.simulation.summarize_wasted_builds`
- **入力**: 戦略 "strategy" のスケジュール DataFrame、閾値が無限大ではないベースライン
- **想定出力**: summary DataFrame と events DataFrame
- **テスト内容**: `success_triggers` などの集計値を確認し、イベント分類が `{"tp","fp"}` になることを検証

---

## `tests/test_core_simulation_run.py`

### `test_run_minimal_simulation_aggregates_strategies`
- **対象関数**: `RQ3.core.simulation.run_minimal_simulation`
- **入力**: `iter_strategies`/`run_strategy` をダミーに差し替えた状態での戦略実行
- **想定出力**: `SimulationResult`（summary DataFrame と schedules 辞書が保持される）
- **テスト内容**: summary の戦略リストと schedules のキー集合を確認

### `test_simulation_result_serialisation_helpers`
- **対象関数**: `RQ3.core.simulation.SimulationResult`
- **入力**: summary に DataFrame、metadata に辞書を渡して初期化
- **想定出力**: `summary.to_dict("records")` が期待どおり、metadata が保持されること
- **テスト内容**: DataFrame→dict の値、metadata の値をチェック

---

## `tests/test_core_timeline.py`

### `test_scan_daily_records_parses_dates`
- **対象関数**: `RQ3.core.timeline.scan_daily_records`
- **入力**: `merge_date` と `daily_commit_count` を含む CSV
- **想定出力**: 日付が昇順ソートされた `List[Dict]`
- **テスト内容**: レコード数とコミット数の値を確認

### `test_build_timeline_generates_cumulative_values`
- **対象関数**: `RQ3.core.timeline.build_timeline`
- **入力**: 2 日分の daily records、`builds_per_day=2`, `fuzz_multiplier=0.5`
- **想定出力**: 累積ビルド数 4
- **テスト内容**: 最終行の `cumulative_builds`を検証

### `test_summarise_project_timeline_handles_empty`
- **対象関数**: `RQ3.core.timeline.summarise_project_timeline`
- **入力**: 空のタイムライン、`builds_per_day=1`
- **想定出力**: 0 を初期値としたサマリー辞書
- **テスト内容**: `cumulative_builds` が 0、プロジェクト名保持を確認

### `test_write_timeline_csv_persists_rows`
- **対象関数**: `RQ3.core.timeline.write_timeline_csv`
- **入力**: 1 行分の辞書リスト、`tmp_path`
- **想定出力**: CSV ファイル（フィールド順が `FIELDNAMES` に一致）
- **テスト内容**: CSV のヘッダと内容を実際に読み取り確認
このカタログはテスト更新に合わせてメンテナンスしてください。特に新しいモジュールや CLI を追加した際は、テスト関数 / 対象関数 / 想定入出力 / assertion 観点を明示することで、今後のテスト整備とカバレッジ管理がスムーズになります。*** End Patch
