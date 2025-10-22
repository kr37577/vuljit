# Strategy3 修正計画

## 目的
- [x] Strategy3 (`strategy3_line_change_proportional`) が採用する各種定数・フォールバックの根拠を明確化し、論文で防衛可能な設計に改める。
- [x] 外れ値やデータ不足時の挙動を安定化させ、追加ビルド数の過大・過小割り当てを抑制する。
- [ ] 変更後の挙動を再現性のある実験と指標で評価し、改善効果を記録する。

## テスト計画
- [x] `vuljit/RQ3/tests/test_additional_build_strategies.py` に窓幅変更時の baseline 決定を検証するケースを追加する。
- [x] 同テストへ丸めモードと Fold 超過調整の挙動を確認するケースを追加する。
- [x] Fold 予算整合性とメタデータ列（`fold_budget`, `fold_budget_source` など）の一貫性を検証するアサーションを追加する。
- [x] Fold→プロジェクト→グローバルのフォールバック経路を WalkForward 割当モックで再現する追加テストを設計する。
- [x] 予算超過時の減算ロジックと Fold 未達時のレポート出力を検証する回帰テストを作成し、既存スナップショットとの差分を確認する。

## 対象コード
- [x] `vuljit/RQ3/additional_build_strategies.py:870-951` の改修対象を精査する。
- [x] `_prepare_line_change_metrics`（同ファイル:210-221）と `_prepare_labelled_timeline`（同ファイル:468-560）の依存を確認する。

## 修正方針

### 1. Fold ベースラインと予算導出
- [x] `_get_project_walkforward_metadata` から Fold ごとの総検出日数と陽性日数を取得するヘルパーを実装する。
- [x] `_compute_project_fold_statistics` を利用して Fold 平均検出日数を算出し、欠損値の扱いを整理する。
- [x] `fold_budget = (Fold 平均検出日数) × (Fold の陽性日数) × build_per_day` を計算するロジックを実装する。
- [x] Fold 平均欠損時にプロジェクト平均→全体平均へフォールバックし、`fold_budget_source` に粒度を記録する処理を追加する。
- [x] すべての統計が欠損の場合に Fold 配分をスキップし、固定値フォールバックを避けたログを出力する。
- [x] `fold_budget`, `fold_sample_count`, `fold_budget_source` を出力へ残し、テストで検証できる状態にする。

### 2. ラインチャーン比率の算出
- [x] Fold 内の陽性日データから `line_change_total` を抽出し、中央値を `line_churn_baseline` として算出する。
- [x] `line_churn_baseline` が 0 または欠損のときに均等重み（`line_weight_raw = 1`）へフォールバックし、`baseline_zero_fallback` をログする。
- [x] `line_weight_raw` を正規化して `line_weight_share` を計算する処理を追加する。
- [x] 陽性日が 1 件のみの場合に 100% 割り当てとなることをテストで確認する。

### 3. Fold 内スケーリングと丸め戦略
- [x] `expected_additional_builds = fold_budget * line_weight_share` を算出するロジックを追加する。
- [x] `rounding_mode` 引数を受け取りデフォルト `ceil` の丸め処理を実装する。
- [x] Fold 合計が予算を超過した場合に、ライン重みの小さい日から超過分を減算する調整ロジックを実装する。
- [x] 予算未達の余剰は Fold 内で報告のみとし、繰り越ししない方針をコードとドキュメントへ反映する。
- [x] `expected_additional_builds_raw`, `rounded_additional_builds`, `rounding_mode_used`, `fold_overflow_used`, `baseline_zero_fallback` 列を出力に追加する。

### 4. 補助メトリクスの統合
- [x] 初期リリースでは補助指標による定数係数調整を導入しない方針をコードへ反映する。
- [ ] 補助指標を導入しない理由と将来拡張の選択肢をドキュメントに整理する。
- [ ] 感度分析で利用する監視指標（追加ビルド逸脱率など）を記録・可視化できるよう準備する。

### 5. ロギングとトレーサビリティ
- [x] スケジュール出力に `line_churn_baseline`, `line_weight_share`, `fold_budget`, `fold_budget_source`, `expected_additional_builds_raw`, `rounded_additional_builds`, `rounding_mode_used`, `fold_overflow_used`, `baseline_zero_fallback` 列を追加する。
- [x] フォールバック経路と丸め調整が期待どおりに動作し Fold 予算を超過しないことをテストで確認する。

## 評価計画
- [ ] OSS-Fuzz WalkForward 期間に対し、Strategy3 の現行版と改訂版を同条件で実行し評価データを取得する。
- [ ] 追加ビルド総数・最大値・中央値を算出し、主要指標の表を更新する。
- [ ] `expected_additional_builds × detection_time` による検出遅延短縮の疑似指標を計算する。
- [ ] 追加ビルド / 基礎ビルド比を計算し、ビルドコスト評価を整理する。
- [ ] Fold 予算計算や丸め戦略のバリエーションで感度分析を行い、現行版との比較表・グラフを作成する。
- [ ] 入力データのバージョンとコミットハッシュを記録し、再現性メモを整備する。

## ドキュメント更新
- [ ] `vuljit/RQ3/docs/additional_build_strategies_detail.md` へ Strategy3 の処理詳細・選定値・評価結果を追記する。
- [ ] `docs/reports/strategy3_rework_results.md` を作成し、評価レポートをまとめる。

## リスクと課題
- [ ] データ不足プロジェクトでフォールバックが多発しないか監視し、差分分析を実施する。
- [ ] 窓幅や分位点の設定をプロジェクト単位で見直し、必要に応じてハイパーパラメータ調整を検討する。
- [ ] 感度分析の再計算コストを評価し、CI で扱うためのサンプリング方針や代表プロジェクト選定を確立する。

## オープンな論点
- [ ] 類似プロジェクトの判定指標（メトリクス距離、ビルド頻度など）を設計する。
- [ ] コミット頻度やファイル数など他メトリクスの重み付け方針を決める。
- [ ] 追加ビルドコスト上限（運用側キャパシティ）の制約方法を検討する。
