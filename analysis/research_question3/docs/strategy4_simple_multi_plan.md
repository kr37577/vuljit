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

- [x] `strategy4_cross_project_regression` が `mode="simple"` で `line_change_total` 単一特徴量を使う。
- [x] CLI/スクリプトから Simple/Multi を切り替えられる。
- [x] Simple/Multi 双方のユニットテストが追加され、CI で成功する。
- [x] ドキュメントに利用手順とモード説明が記載されている。

---

## 7. Strategy1〜3 クロスプロジェクトモード実装ノート

- [x] `strategy1_median_schedule` / `strategy2_random_within_median_range` / `strategy3_line_change_proportional` に `mode` 引数を追加し、`per_project`（従来動作）と `cross_project` を切り替え可能にした。クロスモードでは Fold 列が無くても動作し、グローバル統計を必ず参照する。
- [x] `additional_builds_cli.py` / `minimal_simulation_cli.py` に `--strategy{1,2,3}-mode` を実装。環境変数 `RQ3_STRATEGY{1,2,3}_MODE` でデフォルトを制御でき、`run_minimal_simulation` へ overrides として伝播する。
- [x] Strategy3 では `--strategy3-global-budget` / `global_budget` 引数を追加し、クロスモード時に割り当てる総ビルド量を明示的に上書き可能にした（行変更量の正規化は行わず、生の値でシェア計算）。
- [x] `_compute_project_fold_statistics` のグローバル統計をクロスモードで積極的に利用し、欠損時はプロジェクト統計にフォールバック。Fold 情報を持たないタイムラインでもスケジュール生成ができるように例外処理を緩和した。
- [x] `tests/test_additional_build_strategies.py` にクロスモード専用テスト（Strategy1/2/3）を追加し、`strategy_mode` 列やグローバル統計の適用を検証。
- [x] `docs/additional_build_strategies_detail.md` に CLI フラグとクロスモードの説明を追記し、Simple/Multi と併せて利用方法を明文化。

リスクと対策:
- **データ欠損**: グローバル統計が欠ける場合はプロジェクト統計または既存挙動にフォールバックし、スケジュール生成をスキップして過剰計画を防止。
- **予算超過**: Strategy3 クロスモードでは丸め処理後の総量が `global_budget` を超えないよう再配分ロジックで調整。
- **互換性**: デフォルトは `per_project` のままなので既存シナリオは影響なし。クロスモードで追加された `strategy_mode` 列により、分析時に挙動を識別できる。
