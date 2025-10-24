# Phase 6 ドキュメント

## 1. 目的
- 追加ビルドシミュレーションで実装した仕組みの再現性を確認し、閾値設定ロジックが期待通りに働くかをテストで担保する。
- 分析手順（Phase 4〜追加ビルドシミュレーション）と閾値選定理由を明文化し、レビュー時に参照できるようにする。
- Precision 分析結果と戦略比較の所感をまとめ、最終レポートのたたき台とする。

## 2. テスト実行手順
1. 仮想環境を有効化し、`requirements.txt` に記載のライブラリ（特に `pandas`, `pytest`）をインストールする。
2. ルートを `vuljit` とした上で、以下を実行する：
   ```bash
   PYTHONPATH=. pytest RQ3/tests/test_core_simulation_run.py
   ```
3. `tests/test_core_simulation_run.py` では以下を検証している：
   - `_build_threshold_map` が推定ビルド番号を優先し、欠損時は `baseline_detection_builds` や日数ベースのフォールバックを採用すること。
   - `_summarize_wasted_builds` がイベント単位で TP/FP/`fp_post_detection` を分類し、検出ウィンドウ指定時にベースライン寄与が制限されること。

## 3. 分析フロー概要
1. **Phase 4（`threshold_precision_analysis.py`）**
   - Precision/Recall 曲線を算出し、精度ターゲットごとのリスク閾値を `phase4_outputs/precision_thresholds.csv` に保存。
   - `strategy_precision_sweep.csv` で各戦略の追加ビルド量を比較。
2. **追加ビルドシミュレーション（`cli/additional_builds_cli.py`）**
   - 固定閾値で全戦略を再生し、`simulation_outputs/` 配下に以下を出力：
     - `strategy_summary.csv`（全体指標）
     - `strategy_project_metrics.csv`（プロジェクト別集計）
     - `strategy_overall_metrics.csv`（戦略別の平均・中央値）
     - `strategy_daily_totals.csv`（日単位の追加ビルド合計）
     - `strategy_wasted_builds.csv` / `strategy_wasted_build_events.csv`（累積閾値による成功・無駄判定）
     - `additional_builds_boxplot.png`（戦略ごとのビルド分布）

## 4. 閾値設定と精度に関する考察（要約）
- `baseline_detection_builds` を閾値として採用することで、「過去に検出まで要したビルド量」を疑似再現できる。欠損時は `baseline_detection_days × builds_per_day` を利用し、閾値が0以下の場合は検出不能として扱う。
- Precision ターゲットが高いほど追加ビルドが減り、再現率が下がる傾向が Phase 4 の結果から確認できる。累積ビルド方式では誤検知が続いても閾値に達しなければ検出扱いにならず、無駄ビルドがイベントログに蓄積される。
- 戦略1/2 は追加ビルドが閾値に達しやすく検出スピードを稼ぐ一方、戦略3/4 はビルド抑制が効くため、閾値到達回数が少なく遅延リスクが残る。Precision をどの水準に保つかで最適戦略が変わるため、プロジェクト特性ごとの再評価が推奨される。

## 5. 最終レポート（RQ3節）向けメモ
- Phase 4 で得た Precision 変化のグラフと、追加ビルドシミュレーションの `strategy_overall_metrics.csv` / `strategy_wasted_builds.csv` を組み合わせると、
  - 「どの戦略が最も早く閾値に到達するか」
  - 「無駄ビルド率がどこまで許容範囲か」
  を数値で示せる。
- `strategy_wasted_build_events.csv` の `detection_id` と `evaluation_baseline` / `evaluation_trial` を用いれば、閾値到達前後の累積値を可視化でき、Precision 調整の効果を説明しやすい。

本ドキュメントをレビュー後、必要に応じて最終レポート本文へ転記してください。
