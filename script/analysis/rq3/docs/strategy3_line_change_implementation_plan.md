# Strategy3 実装計画（Fold検出日数比例版）

## 目標
- Strategy3 (`strategy3_line_change_proportional`) を Fold ごとの検出日数に比例した追加ビルド割当へ再実装する。
- ラインチャーンと補助メトリクスによる配分ロジックを明文化し、テストとログで追跡可能にする。
- 既存のデータセット生成・評価パイプラインへの影響を最小化しつつ、再現可能な実験環境を整備する。

## 前提整理
- 参照コード
  - `vuljit/RQ3/additional_build_strategies.py`
  - `vuljit/RQ3/tests/test_additional_build_strategies.py`
  - Fold メタデータ生成ヘルパー群（`_get_project_walkforward_metadata`, `_compute_project_fold_statistics`）
- 参照データ
  - WalkForward 予測結果（`predicted_*` シリーズ）
  - ラインチャーン集計（`_prepare_line_change_metrics`）
  - オプションの補助列（`daily_commit_count`, `builds_per_day`）
- 既存ドキュメント
  - `vuljit/RQ3/docs/strategy3_line_change_rework_plan.md`（仕様方針）

## 実装タスク詳細

### フェーズ0：環境点検
1. WalkForward メタデータと検出テーブルの最新スキーマを確認し、Fold ID の命名と日付解決ロジックを把握する。
2. 既存 Strategy3 のテストカバレッジを把握し、欠落している検証観点（Fold 有無、丸め影響など）をメモ化。
3. 必要な設定値（デフォルト `scaling_factor`, `rounding_mode`, バックアップ予算値）を決定し、設定ファイルの編集範囲を明示。

### フェーズ1：データ取得と整形
1. Strategy3 内で WalkForward メタデータを取得するユーティリティ（例：`_load_project_fold_metadata(project)`）を実装。
2. `_prepare_labelled_timeline` 呼び出し時に `walkforward_assignments` を常に渡すよう改修し、Fold 情報が陽性日に結合されることを確認。
3. ラインチャーンデータと補助列を統合したフレームに Fold ID を追加し、Fold→日別のグループ化ができる形へ整形。
4. 整形処理の単体テスト（小規模 DataFrame で Fold 結合が期待通りか）を追加。

### フェーズ2：Fold 予算ヘルパー実装
1. 新関数 `_compute_fold_budget(project, fold_id, fold_stats, label_count, *, fallback_config)` を作成。
   - 引数：Fold 統計辞書、陽性日件数、フォールバック設定。
   - 出力：`fold_budget`, `fold_sample_count`, `fold_budget_source`.
2. Fold 統計が欠損した場合にプロジェクト全体→グローバル統計→設定ファイルの定数へフォールバックするロジックを実装。
3. ユニットテストで各フォールバック経路を網羅（Fold 無し、サンプル不足、完全欠損）。
4. メタデータ列の値が仕様通りであることをテストで検証。

### フェーズ3：ライン重み計算モジュール化
1. `_resolve_line_churn_baseline(fold_rows, project_stats, global_stats)` のようなヘルパーを作成して中央値→フォールバックの順序を統一。
2. 陽性日ごとの `line_weight_raw` と `line_weight_share` を算出する関数を実装、Fold グループで呼び出す。
3. ベースラインや重みが NaN／0 の場合に安全側へ倒す処理（`max(value, ε)`）を追加。
4. ヘルパー単体テストで、陽性日が1件のケースや全てゼロのケースを確認。

### フェーズ4：Strategy3 本体改修
1. Fold ごとの陽性日サブフレームをループし、以下を順に実行：
   - Fold 予算を取得し、`fold_budget` を格納。
   - ライン重みと補助調整によって `expected_additional_builds_raw` を計算。
   - `scaling_factor` 適用後、丸め（デフォルト `ceil`）を実行。
   - 丸め後に予算超過した場合、超過分を小さい重みから減算して調整。処理結果を `fold_overflow_used` に記録。
2. 行単位で必要なメタデータ列を作成し、`rows` リストへ追加。
3. `project_overrides`（未設定可）により `scaling_factor` や補助係数を上書き可能にする仕組みを導入（辞書引数など）。
4. 既存の戻り DataFrame に新列（`line_churn_baseline`, `line_weight_share`, `fold_budget`, `fold_budget_source`, `expected_additional_builds_raw`, `scaling_factor_used`, `rounding_mode_used`, `aux_adjustment_applied`, `fold_overflow_used`）を追加。
5. docstring とコメントを更新し、新しい引数や挙動を説明。

### フェーズ5：補助メトリクス調整
1. `daily_commit_count` や `builds_per_day` が存在する場合に定数係数で重みを減衰させるロジックを実装。
2. 補助指標が欠損した場合のフォールバック（係数 1.0 を適用）とメタデータ記録を実装。
3. 係数を設定ファイルまたは関数引数から指定できるようにする。
4. テストで補助メトリクスによる減衰の有無を確認。

### フェーズ6：テスト拡充（1日）
1. Strategy3 用に新たなフィクスチャ DataFrame を用意し、複数 Fold・陽性日・補助指標あり／なしなどのケースを網羅。
2. 主要テスト項目：
   - Fold 予算が正しく計算され、行ごとの割当合計が予算を越えない。
   - 丸めモード切替（`ceil`, `round`）で期待する結果が得られる。
   - フォールバック経路（Fold なし、プロジェクトなし、グローバルのみ）が正しく反映される。
   - 補助メトリクス調整の働きとメタデータ列の値を検証。
3. 既存テストの期待値を更新し、Strategy3 の出力形式変更に対応。
4. `pytest -k strategy3` 実行でローカル確認。

### フェーズ7：評価ツール更新（0.5日）
1. バックテストスクリプト（例：`vuljit/RQ3/scripts/` 配下）を更新し、新メタデータ列を読み込めるようにする。
2. Fold 予算と割当結果の整合性をチェックする集計レポートを追加。
3. 感度分析用のテンプレート設定（複数 `scaling_factor` や補助係数の組合せ）を準備。

### フェーズ8：ドキュメント更新（0.5日）
1. `additional_build_strategies_detail.md` に今回のアルゴリズム変更とパラメータを追記。
2. 評価レポートテンプレート（`docs/reports/strategy3_rework_results.md`）を更新し、Fold 予算やメタデータの読み方を記載。


## 依存関係メモ
- Fold 統計生成が別タスクで改修される場合は、そのスキーマ確定後にフェーズ1以降を着手。
- 設定ファイルの変更が必要な場合、共用コンポーネントへの影響を確認してから PR を分割する。
- 評価スクリプトを複数人で共有しているなら、メタデータ列追加に伴う互換性調整を事前にコミュニケーションする。

## 完了判定
- Strategy3 が Fold 検出日数を基に割当を行い、ユニットテスト・バックテストで整合性が確認できる。
- 新しいメタデータ列が生成され、ドキュメントとテストに反映されている。
- 感度分析と再現実験が実施可能になっており、分析手順が記録されている。
