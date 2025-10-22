# 追加ビルド戦略の詳細まとめ

本ドキュメントでは `vuljit/RQ3/additional_build_strategies.py` に実装されている追加ビルド戦略と、その前提となるヘルパー処理について日本語で解説します。OSS-Fuzz のタイムラインや WalkForward 学習設定に基づき、JIT 予測結果から追加ビルドの配分量を算出する仕組みを理解するための参考資料です。

---

## 1. 入力データと共通ヘルパー

### 1.1 主要データソース
- **ビルドタイムライン**：`timeline_outputs/build_timelines/*.csv`。1 日単位の追加ビルド候補を保持し、`merge_date_ts`・`day_index`・`builds_per_day` などを含みます。
- **検出遅延テーブル**：`rq3_dataset/detection_time_results.csv`。脆弱性コミットの検出遅延（`detection_time_days`）を持つ基礎データです。
- **予測 CSV**：`*_daily_aggregated_metrics_with_predictions.csv`。WalkForward 学習設定（分割数・訓練ウィンドウ）と一貫させるため、ここから Fold 情報を構築します。

### 1.2 WalkForward メタデータ生成
`_get_project_walkforward_metadata()` が予測 CSV を読み込み、以下の情報をキャッシュします。

- `folds`: Fold ID ごとの `train_start`・`train_end`・`test_indices` など。
- `assignments`: タイムラインの日付へ `walkforward_fold` と訓練ウィンドウ境界 (`train_window_start` / `train_window_end`) をマージするためのテーブル。
- `config`: `N_SPLITS_TIMESERIES` と `USE_ONLY_RECENT_FOR_TRAINING` の実際の値（環境変数または設定に応じて解決）。

### 1.3 検出統計の計算
`_compute_project_fold_statistics()` がプロジェクト × Fold の遅延統計量（中央値・四分位数）を作成します。

1. `detection_time_results.csv` を読み込み、不正値（負値や `NaN`）を除外。
2. WalkForward の訓練期間に含まれるコミットをフィルタリング。
3. Fold → プロジェクト全体 → 全体（グローバル）という優先順位で `median` / `q1` / `q3` を求め、欠損時にはフォールバックを返す構造に整備。

### 1.4 ラベル付きタイムラインの準備
`_prepare_labelled_timeline()` はタイムラインと予測 CSV を日付で結合し、以下を付与します。

- 予測ラベル（`_strategy_label`）およびリスクスコア（`RISK_COLUMN` デフォルト：`predicted_risk_VCCFinder_Coverage`）。
- Fold 割り当て（`walkforward_fold`）、訓練ウィンドウ（`train_window_start` / `train_window_end`）などのメタ情報。
- ラベル由来（予測ラベル列か、しきい値判定か）を返り値で伝達。

---

## 2. 戦略ごとの処理フロー

### 2.1 Strategy 1: `strategy1_median_schedule`
**目的**：JIT 予測で脆弱と判断された日について、検出遅延の中央値日数ぶんのビルドを上乗せする。

1. ラベル付きタイムラインから `_strategy_label == True` の行のみを抽出。
2. `_compute_project_fold_statistics()` の結果から、行に対応する Fold → プロジェクト全体 → グローバルの順で中央値 (`median_detection_days`) を解決。
3. `builds_per_day` が正であれば `ceil(median_days * builds_per_day)`、そうでなければ `ceil(median_days)` を追加ビルド数とする。
4. 出力レコードには `walkforward_fold`, `train_window_start`, `train_window_end`, `median_source`（どの統計を使用したか）を含む。

**ポイント**：
- Fold 内に統計が存在しない場合は自動的にプロジェクト全体・全体値へフォールバック。
- 中央値が `NaN` になる場合は追加ビルドを生成しないため、データ不足 Fold の過剰なスケジュール化を防げます。

### 2.2 Strategy 2: `strategy2_random_within_median_range`
**目的**：Fold 単位の四分位範囲（IQR）から乱数で検出遅延をサンプルし、追加ビルド数に変換する。

1. Strategy 1 と同様にラベル付きタイムラインを基に対象行を抽出。
2. `_compute_project_fold_statistics()` から `q1` / `q3` を Fold → プロジェクト → グローバルの順に取得。値が取得できなければスキップ。
3. プロジェクト名・Fold ID・ユーザー指定の `random_seed` から BLAKE2s ハッシュで安定的なシードを生成し、`numpy.random.Generator(np.random.PCG64)` を初期化。
4. `[q1, q3]` から一様分布で `sampled_offset_days` を取得（上下限が同値の場合はその値に固定）。
5. Strategy 1 と同様の方法で `scheduled_additional_builds` を算出し、四分位情報と Fold メタデータを出力に記録。

**ポイント**：
- Fold ごとに決定的な乱数シードを利用するため、実行環境が異なっても結果が再現可能。
- IQR が欠損または逆転するケースは、そのままスキップしてフォールバックやグローバル値へ移行。

### 2.3 Strategy 3: `strategy3_line_change_proportional`
**目的**：日次のコード変更量に比例させて追加ビルド数を変動させる（ラベルが陽性の日のみ計画を生成）。

- `vuljit/RQ3/additional_build_strategies.py:870-884`  
  戦略関数の定義。タイムライン・メトリクス・予測 CSV のパスを解決し、結果行を格納する `rows` を初期化。
- `:887-892`  
  プロジェクトごとにタイムラインを走査し、空タイムラインを除外。`_prepare_line_change_metrics` で日次メトリクス（`lines_added`/`lines_deleted`）を読み込み、`line_change_total` 列を生成。
- `:893-902`  
  `_prepare_labelled_timeline` を呼び、ビルドタイムラインと予測データを `merge_date_ts` で結合。JIT ラベル列 `_strategy_label` と `builds_per_day` 等を取得。取得できなければ次のプロジェクトへ。
- `:903-908`  
  ラベル付きタイムラインとメトリクスを再度 `merge_date_ts` で左結合し、1 日単位の行に churn とラベルを集約。
- `:909-911`  
  `_strategy_label` が True の日を抽出し、陽性サンプルが1件もなければスキップ。
- `:912-918`  
  `line_change_total` を 0 埋め・float 化。陽性日の正値のみで中央値を取り、それ以外は全体中央値を利用。基準値 `baseline` が 0 以下または非有限なら最大値または 1.0 に置き換え。`baseline` で割った正規化値を `clip_max` 上限でクリップし、`scaling_factor` を掛けて期待追加ビルド数 `expected_extra` を算出。
- `:919`  
  `expected_extra.round().astype(int)` で `scheduled_extra` を整数化（銀行丸め）。
- `:920-926`  
  `daily_commit_count` で始まる列をまとめ、複数列があれば `combine_first` で欠損補完。存在しない場合は NaN の Series を用意。
- `:928-949`  
  `_strategy_label` が True の各日について `scheduled_extra` が 1 以上か確認。0 以下は除外し、残りはレコードを組み立てる。出力には `line_change_total`、正規化値、期待値、最終的な `scheduled_additional_builds`、ラベル情報、リスクスコアなどを含める。
- `:951`  
  蓄積した `rows` を DataFrame 化して返却。陽性日かつ正規化後の追加ビルド数が 1 以上の行のみが最終結果に入る。

**補足**：
- `baseline` はそのプロジェクトの「通常の churn 規模」を近似するための基準で、陽性日の中央値 → 全体中央値 → 最大値/1.0 の順にフォールバック。
- 丸めは pandas の銀行丸めを利用しており、`0.5` などが偶数方向に丸められる点に注意。
- `clip_max` を調整すると、大きな churn の日に割り当てる追加ビルド数の上限を制御できる（デフォルト 5.0、設定で変更可能）。

### 2.4 Strategy 4: `strategy4_cross_project_regression`
**目的**：観測された追加ビルド量（検出遅延 × ビルド頻度）を教師データとして線形回帰モデルを学習し、各日ごとの追加ビルド数を予測する。

1. `_build_regression_dataset()` がタイムライン・検出遅延・ビルド頻度・予測ラベルを組み合わせ、目的変数 `observed_additional_builds` を算出。
2. プロジェクト単位で学習・評価用に分割し、`np.linalg.lstsq` で単純な最小二乗線形モデルを学習。
3. 学習済みモデルで各日の追加ビルド数を推定し、負値は 0 にクリップ、`ceil` で整数化。
4. タイムラインと整列させつつ `run_strategy` などの呼び出し側が扱いやすい DataFrame を返す。

**ポイント**：
- ラベルが真の行のみを対象とするため、JIT 判定を前提にした予測モデル。
- 学習・評価指標（MAE など）は戻り値で併せて返却され、分析時にモデル精度を確認できます。

---

## 3. 出力レコードの共通項目

- `project`: OSS-Fuzz プロジェクト名。
- `strategy`: 戦略識別子（例：`median_label_trigger`, `median_iqr_random`）。
- `merge_date`: 追加ビルドを実行予定の日付（UTC）。
- `day_index`: タイムライン上の連番。
- `builds_per_day`: 基準となる日次ビルド頻度。
- `scheduled_additional_builds`: 戦略が提案する追加ビルド数（整数）。
- `label_source` / `label_threshold`: ラベル列やしきい値の情報。
- `walkforward_fold`, `train_window_start`, `train_window_end`: 該当 Fold と訓練ウィンドウのメタ情報（Strategy 1/2）。
- 戦略固有の列（`median_detection_days`, `sampled_offset_days`, `line_change_total`, `predicted_additional_builds` など）。

---

## 4. 再現性と設定依存の注意点

- WalkForward の設定値（`N_SPLITS_TIMESERIES`, `USE_ONLY_RECENT_FOR_TRAINING`）は予測生成時と追加ビルド戦略側で一致させる必要があります。環境変数 `RQ3_WALKFORWARD_SPLITS` / `RQ3_WALKFORWARD_USE_RECENT` が利用可能です。
- Strategy 2 は NumPy の PCG64（決定的 RNG）を用いるため、Python バージョン違いでも結果が安定します。
- Strategy 4 の回帰モデルは Fold 境界よりもプロジェクト単位のランダム分割に依存するため、`random_seed` を揃えることで学習結果の再現性を担保してください。

---

## 5. 参考：関連テスト

- `tests/test_additional_build_strategies.py` には各戦略と `_compute_project_fold_statistics` の挙動を検証するテストが実装されています。Fold メタデータの統合やフォールバック優先順位、決定論的乱数など重要な性質を確認する際の参考になります。

---

このドキュメントを参照することで、各戦略がどのように入力データを処理し、追加ビルド数を決定するのかを把握できます。分析や改修を進める際にご活用ください。
