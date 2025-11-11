# Strategy LOPO Reliability Plan

本ドキュメントは、クロスプロジェクト運用時に LOPO 統計が正しく計算・消費されるよう、追加ビルド戦略（Strategy1〜3 および Strategy4 の前処理）で必要な修正点をまとめた実装計画である。

## 背景

- `_load_detection_table` の日時取り込みが壊れており、`commit_date`/`reported_date` に `NaT` が入るため fold 統計や LOPO 統計が空になる。
- Strategy1/2/3 では `mode="cross_project"` でしかグローバル／LOPO フォールバックが有効にならず、デフォルトの `per_project` モードで fold 統計が欠けるとスケジュールが生成されない。
- Strategy3 の `global_budget` は fold ごとに同じ値を割り当てており、全体上限として機能していない。

## 対応方針

### 1. Fold／LOPO 統計の元データ修復

- 対象: `_load_detection_table`（`vuljit/analysis/research_question3/additional_build_strategies.py`）
- 変更:
  - `_coalesce_datetime` 呼び出し時に `"reported_date_utc"` → `("reported_date_utc",)` のようにシーケンスを渡す。
  - 併せて `"commit_date_utc"` など必要列を列挙し、優先順位どおりに値を埋める。
- 期待効果: `_compute_project_fold_statistics` で `commit_ts` が正しく生成され、fold ごとの検出サンプルおよび LOPO（対象除外）統計が算出可能になる。

### 2. Strategy1〜3 のフォールバック統一と LOPO 優先化

- 対象: `_resolve_median`（S1）、`_resolve_quartiles`（S2）、`_resolve_median_with_fallback`（S3）
- 変更:
  - `allow_project_fallback` / `allow_global_fallback` をモードに依存させず常に `True` とし、ドキュメントどおり `fold → project → LOPO → global` の順に探索する。
  - LOPO 用辞書が存在する場合はグローバル統計より先に参照するよう関数内の分岐を整理する。
- 期待効果: クロス/パーモード双方で fold 統計欠損時に必ず LOPO 又はグローバル統計に落ち、スケジュールが空になるケースを解消できる。

### 3. Strategy3 `global_budget` の全体上限化

- 対象: `strategy3_line_change_proportional` の fold ループ（行 1335 付近）
- 変更:
  - `fold_budget_target` を fold ごとにリセットするのではなく、全体予算を残量として保持し各 fold へ配分する。
  - 丸め処理後に総計が上限を超えないよう、余剰削減ロジックを fold から全体に拡張する（例: 各 fold の割当をリスト化し、超過分を順次減算）。
  - LOPO 由来の割当もこの残量管理に含め、`global_budget` の意図とドキュメント記述（「超過しない」）を一致させる。
- 期待効果: Cross モードで `global_budget` を指定した際、fold 数に比例してビルド総数が膨らむ問題を解消し、再現性のある予算制御を実現する。

## 実装・検証チェックリスト

- [ ] `_load_detection_table` 修正後、`_compute_project_fold_statistics` で fold / LOPO 統計が生成されることをユニットテストやデバッグ出力で確認する。
- [ ] Strategy1〜3 のフォールバックが fold → project → LOPO → global の順に機能することをテストし、fold 欠損ケースでもスケジュールが作成されることを検証する。
- [ ] Strategy3 の `global_budget` が総スケジュール量の上限として働くようになったか、複数 fold の入力で合計ビルド数が指定値以下に収まることを確認する。
- [ ] すべての戦略をクロスモードで再シミュレーションし、`strategy_project_metrics.csv` / `strategy_summary.csv` の主要メトリクスが期待どおりか差分をレビューする。

## リスクとフォローアップ

- 日付列の変換で新たな `NaT` が発生しうるため、列欠損時のフォールバック（`commit_date_utc` → `commit_date` など）をテストで担保する。
- フォールバック順序を変更することで既存メトリクスが変化する可能性があるため、リリース前に旧結果との比較レポートを作成する。
- `global_budget` の再配分に伴い計算コストが増す場合は、fold 数やイベント数を考慮した効率化（ヒープ利用など）を検討する。
