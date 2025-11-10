# Strategy4 Simple/Multi Regression 実装計画

## 1. 背景とゴール

- **背景**: 現状の Strategy4 (`strategy4_cross_project_regression`) は複数特徴量を使った線形回帰のみを想定しており、論文で言及されている *Simple Regression*（行変更量のみ）と *Multi Regression*（RAISEF と同一特徴量）の切り替えを CLI から行えない。
- **ゴール**:
  1. ユーザーが CLI で Simple/Multi を明示的に選べるようにする。
  2. Simple モードでは `line_change_total` の単一特徴量でモデルを学習・推論する。
  3. Multi モードでは従来どおり Prediction 設定由来の複数特徴量を使用する。
  4. どちらのモードでも再現性ある学習・評価・スケジュール生成ができることをテストで担保する。

## 2. 前提・制約

- `line_change_total` は `_prepare_line_change_metrics()` で生成可能。Simple モードではこの列の存在を必須とする。
- CLI からの設定値は `run_strategy` → `core/scheduling` → `strategy4_cross_project_regression` の順に伝播させる必要がある。
- 追加オプション導入に伴い、`rq3.sh` や `run_prepare_RQ3.sh` などの既存スクリプトが壊れないようデフォルトを後方互換（Multi モード）にする。
- 評価指標（MAE など）や診断 JSON の構造は変えない。

## 3. 変更タスクリスト

### 3.1 CLI/設定面

| タスク | 詳細 | 成果物 |
| --- | --- | --- |
| `cli/additional_builds_cli.py` にオプション追加 | 例: `--strategy4-mode {multi,simple}`（デフォルト `multi`）、必要であれば `--strategy4-feature` カスタム指定も許可。 | CLI 引数、help 文字列 |
| 環境変数フォールバック | `RQ3_STRATEGY4_MODE` のような環境変数をチェックし、CI やシェルスクリプトから切り替えやすくする。 | `core/io.resolve_default` のような仕組みは不要だが、CLI→os.environ 読み込みを追加 |
| `simulate_additional_builds.py` / `minimal_simulation_cli.py` | 追加ビルド一括実行スクリプトからも Strategy4 モードを渡せるようにする。 | 引数・docstring 更新 |
| `run_prepare_RQ3.sh` / `rq3.sh` | オプションを渡す例・コメントを追加（実行時に簡単に切り替えられるようにする）。 | スクリプト更新 |

### 3.2 コアロジック

| タスク | 詳細 | 成果物 |
| --- | --- | --- |
| `core/scheduling.py` | Strategy4 ラッパーに `**kwargs` で `mode` や `feature_cols` を受け取り、`strategy4_cross_project_regression` へ透過的に渡す。 | `_strategy4_wrapper` 更新 |
| `additional_build_strategies.py` | - `SIMPLE_REGRESSION_FEATURES = ("line_change_total",)` を定義<br>- Strategy4 の引数に `mode: Literal["multi","simple"]` を追加<br>- `mode=="simple"` のとき `feature_cols` を強制的に Simple リストへ設定。<br>- Simple モードでは行変更メトリクスが不足している場合にエラー/警告を出す。 | 関数シグネチャ・内部処理 |
| `strategy4_cross_project_regression` の入力検証 | `feature_cols` をログに出す、`dataset` から列欠損時のハンドリングを強化（現在の 0.0 代入に加え、Simple モードで欠損が多い場合は警告）。 | ログ＋例外 |
| `_build_regression_dataset` | Simple モード時は line-change メトリクスを必ずロードするよう早期バリデーションを入れる（例: `data_dir` が無いときは `FileNotFoundError`）。 | 追加チェック |

### 3.3 テスト

| テスト項目 | 内容 |
| --- | --- |
| Simple モード単体テスト | `strategy4_cross_project_regression(mode="simple")` をモックデータで実行し、`feature_signature` が Simple 特徴量になっていること、スケジュールが生成されることを確認。 |
| CLI 経路テスト | `tests/test_core_scheduling.py` などで `run_strategy("strategy4_regression", strategy4_mode="simple")` が `feature_cols=("line_change_total",)` を渡していることを検証。 |
| 退行テスト | 既存の Strategy4 テストを `mode="multi"` 明示に変更し、従来結果が変わらないことを保証。 |

### 3.4 ドキュメント更新

- `docs/additional_build_strategies_detail.md` に Simple/Multi の説明・CLI 使い方を追加。
- `README.md` や `analysis/research_question3/docs/strategy4_cross_project_rework_plan.md` にも、今回の追加で満たされる TODO を明記。
- 必要であれば実験ノート（例: `phase6_results.md`）に再実行方法を追記。

## 4. 実装順序案

1. **CLI オプション追加**（既存動作を崩さないよう Multi をデフォルトに設定）。
2. **Strategy4 内部でモード判定**（Simple 特徴量プリセットや入力検証を導入）。
3. **テスト作成・更新**（Simple モード用 fixture / CLI 経路）。
4. **ドキュメント整備**（仕様説明、利用例、再現手順）。
5. **スクリプトへの反映・リグレッション確認**（`rq3.sh` を Simple で動かして問題ないか手動で確認）。

## 5. リスクと緩和策

- **行変更メトリクス欠損**: Simple モードではライン情報が必須になる。→ `_prepare_line_change_metrics` 失敗時に明確な例外を投げ、CLI で早期に気付けるようにする。
- **特徴量の順序・シグネチャずれ**: `feature_signature` が結果の識別に使われるため、Simple/Multi で別ハッシュになることを想定しドキュメント化。
- **CLI 互換性**: 既存スクリプトが新オプションを知らなくても Multi として動くよう、必須引数にはしない。
- **テストコスト**: Simple モード用にラインメトリクス CSV をモックする必要がある。→ フィクスチャで最小限の DataFrame を準備する。

## 6. 完了条件チェックリスト

- [ ] `strategy4_cross_project_regression` が `mode="simple"` で `line_change_total` 単一特徴量を使う。
- [ ] CLI/スクリプトから Simple/Multi を切り替えられる。
- [ ] Simple/Multi 双方のユニットテストが追加され、CI で成功する。
- [ ] ドキュメントに利用手順とモード説明が記載されている。

---

## 7. 追加計画: Strategy1〜3 の Fold 非依存クロスプロジェクト化

### 7.1 背景とゴール

- 現状の Strategy1〜3 は「対象プロジェクトの Fold 統計と行動量だけ」を参照するため、クロスプロジェクトなリソース配分ができない。
- 複数プロジェクト間で追加ビルド予算や統計値を共有できるモードを追加し、Fold 情報に依存しない評価を可能にしたい。
- 目標は以下の通り。
  1. Strategy1〜3 に `mode`（例: `per_project` / `cross_project`）を追加し、CLI から切り替えられるようにする。
  2. `cross_project` モードでは Fold 列を参照せず、対象プロジェクト以外の統計・行動量を積極的に利用した配分を行う。
  3. 既存の per-project 挙動は後方互換を維持する。

### 7.2 モード／設定タスク

| タスク | 詳細 | 成果物 |
| --- | --- | --- |
| 戦略ごとの `mode` 引数追加 | `strategy{1,2,3}_*` に `mode` と `global_budget`（必要に応じて）を追加し、デフォルトで `per_project`。 | 新パラメータ、docstring 更新 |
| CLI オプション | `additional_builds_cli.py` / `minimal_simulation_cli.py` に `--strategy{1,2,3}-mode`（および `--strategy3-global-budget` など）を追加し、`run_minimal_simulation` → `_strategyX_wrapper` に伝播。 | CLI 追加引数、ヘルプ |
| スクリプト/環境変数 | `rq3.sh` 系スクリプトや環境変数 (`RQ3_STRATEGY1_MODE` 等) でモードを上書きできるようにする。 | サンプル設定、コメント |

### 7.3 コアロジック変更

| タスク | 詳細 | 成果物 |
| --- | --- | --- |
| グローバル統計の再導入 | `_compute_project_fold_statistics` から Fold を集約したグローバル中央値/IQR を常に取得し、`mode=="cross_project"` のとき優先的に利用する。 | 追加戻り値／API 更新 |
| Positive 行のクロスプロジェクト結合 | `strategy1_median_schedule` / `strategy2_random_within_median_range` で全プロジェクトの positive 行を結合し、グローバル統計で `scheduled_additional_builds` を計算。必要なら総予算で正規化。 | 新しい配分関数 |
| Strategy3 の正規化配分 | `line_change_total` を全プロジェクトで正規化（z-score 等）し、`global_budget` をラインシェアで割り当てるロジックを追加。Fold 別 `fold_budget` 計算は `per_project` のみで実行。 | 新しいスケーリング処理・配分マップ |
| Fold 依存処理の無効化 | `cross_project` モードでは `_get_project_walkforward_metadata` や `walkforward_assignments` の読み込みをスキップし、不要な列を出力しない。 | 条件分岐の整理 |

### 7.4 シミュレーション／評価

| タスク | 詳細 |
| --- | --- |
| 予算シミュレーション | クロスプロジェクト配分時に一日あたりの追加ビルド上限や全体総量を指定できるようにし、結果を `SimulationResult.summary` に記録する。 |
| テストケース | `tests/test_additional_build_strategies.py` に複数プロジェクトを入力する新フィクスチャを追加し、グローバル中央値/IQR やラインシェアが既定どおり適用されることを確認。 |
| ドキュメント更新 | `additional_build_strategies_detail.md` にモード切り替え手順と想定シナリオを追記する。 |

### 7.5 実装順序案

1. `mode` 引数と CLI オプションを追加し、per-project デフォルトのまま配線を通す。
2. `_compute_project_fold_statistics` のグローバル統計 API を整備（Fold 情報を参照しないコードパスを追加）。
3. Strategy1/2 の positive 行を結合するユーティリティを実装し、`cross_project` モード用にグローバル中央値/IQR を使うルートを実装。
4. Strategy3 のラインシェア正規化＋グローバル予算配分を実装。
5. シミュレーション／テスト／ドキュメントを更新し、per-project モードとの回帰差分を確認。

### 7.6 リスクと緩和策

- **データ欠損**: 他プロジェクトのラインメトリクスや統計が不足するとグローバル値が歪む。→ 欠損率を監視し、一定以上で per-project にフォールバックするガードを入れる。
- **予算超過**: グローバル配分時に合計ビルド数が想定を超える可能性。→ `global_budget` を必須にし、余剰が出た場合は自動的に丸めるロジックを導入。
- **互換性**: 既存の per-project 挙動が変化しないよう、デフォルトモードとテストを維持する。
- **評価指標の比較困難**: モード間で `scheduled_additional_builds` 総量が異なると比較しづらい。→ シミュレーションレポートにモード名と予算パラメータを明記し、Fair comparison を支援する。
