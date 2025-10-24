# Additional-Build Simulation Review Checklist

## 1. 事前準備で確認する情報
- `rq3_dataset/detection_time_results.csv`
  - `project` もしくは `package_name` 列と `detection_time_days` が存在すること
  - 欠損値・異常値の扱い方（NaN/負値除去方針）が設計どおりか
- `rq3_dataset/project_build_counts.csv`
  - `project` 列と `builds_per_day` の型・値域（0以下の行がないか）
- 予測ファイル（`../outputs/results/xgboost/*_daily_aggregated_metrics_with_predictions.csv`）
  - `merge_date`/`merge_date_ts`, `predicted_risk_*`, `scheduled_additional_builds` 等がPhase3出力と整合しているか
- 実行環境の依存ライブラリ
  - `pandas`, `matplotlib` が利用できる環境であるか（CI/ローカル両方）

## 2. コードレビューの重点ポイント
- `cli/additional_builds_cli.py`
  - **入力整形**: `core.io.load_detection_table`, `core.io.load_build_counts` が列名違いや欠損を許容できるか
  - **メトリクス計算**: `_prepare_project_metrics` の平均/推定日数計算が負値にならないか、`builds_per_day` が0のケースを考慮しているか
  - **日次集計**: `_prepare_daily_totals` が日付列欠損時にスキップしているが、必要であれば警告やテストを追加すべきか
  - **可視化**: `_plot_additional_builds_boxplot` が `matplotlib` 依存である点と、外れ値非表示の意図を共有できているか
  - **CLI引数**: 既存ツールとの整合性、上書きポリシー、`--output-dir` 既存ディレクトリの扱い
- 生成物の妥当性
  - `simulation_outputs/strategy_project_metrics.csv` 等が期待通りに生成されるか（列名、値域）
  - `simulation_outputs/strategy_wasted_builds.csv` で無駄ビルド集計が正しく分類されているか
  - `simulation_outputs/strategy_wasted_build_events.csv` にトリガ詳細が揃っているか（累積・消費ビルド、検出ID、期限切れの扱い）
  - 箱ひげ図 `simulation_outputs/additional_builds_boxplot.png` のレイアウトや凡例が解釈しやすいか

## 3. 動作確認のフロー例
1. `python simulate_additional_builds.py --output-dir simulation_outputs_test --silent`
2. 生成されたCSVの列・件数を spot check（例: `head`, `describe`）
3. 箱ひげ図を開き、戦略ごとの傾向が可視化されているか確認
4. 最低1プロジェクトについて Phase3 出力との突合（`scheduled_additional_builds` の合計に差分がないか）

## 4. 未着手・フォローアップ
- フェーズ6で必要になるテスト/ドキュメントのTODOを洗い出し、別途課題管理へ
- 閾値を Precision 解析から自動取得する拡張要否の検討（Phase4との接続）
