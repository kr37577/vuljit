# Strategy4 修正計画

## 目的
- [ ] Strategy4 (`strategy4_cross_project_regression`) で脆弱性ラベル作成時の JIT モデルと同一の特徴量群・前処理を利用し、機械学習プロセスの整合性と防衛可能性を高める。
- [ ] 追加ビルド予測モデルの学習・評価手順を OSS-Fuzz WalkForward 設定と同期させ、再現性のある指標を提示する。
- [ ] データ漏洩や特徴量不整合を防ぎつつ、結果の解釈性と運用性を向上させる。

## テスト計画
- [ ] `vuljit/RQ3/tests/test_additional_build_strategies.py` に WalkForward 用モックデータを追加し、Fold 単位の学習・推論分割と特徴量リスト同期を検証する。
- [ ] 前処理ユーティリティの単体テストを `prediction` モジュールと共用し、特徴量欠損・列追加時のフォールバック挙動を確認する。
- [ ] 学習サンプル閾値未満や特異行列発生時のフォールバック（学習スキップ・既存戦略委譲）が安全に実行されるかをテストし、スケジュールメタデータの整合性を検証する。
- [ ] 回帰評価スクリプトのテストを用意し、Fold 別指標（MAE / RMSE / MAPE）と集計結果の再現性を確認する。

## 対象コード/資産
- [ ] `vuljit/RQ3/additional_build_strategies.py:954-1159` の改修対象を洗い出す。
- [ ] `/work/riku-ka/vuljit/prediction` パイプラインと Strategy4 の依存関係を整理する。
- [ ] `prediction/settings.py`, `data_preparation.py`, `model_definition.py`, `evaluation.py`, `main_per_project.py` の共通化ポイントを特定する。
- [ ] `rq3_dataset/*.csv`, `timeline_outputs/build_timelines/*.csv`, `prediction` ディレクトリ内の特徴量結合 CSV の入出力要件を確認する。

## 修正方針

### 1. WalkForward 整合データパイプライン
- [x] `prediction.main_per_project` と同一の WalkForward 設定（`settings.N_SPLITS_TIMESERIES`, `USE_ONLY_RECENT_FOR_TRAINING` など）を読み取り同期させる。
- [x] `_build_regression_dataset` を改修し、Fold ごとに学習用と検証用を分離し未来参照を防ぐフィルタを追加する。
- [x] WalkForward アサインメントと検出遅延・タイムラインを結合するヘルパーを実装し、Fold ID と学習ウィンドウ境界を各行に付与する。

### 2. 特徴量・前処理の共通化
- [x] `prediction/settings.py` の `predicted_risk_VCCFinder_Coverage` 特徴量集合（`KAMEI_FEATURES` + `VCCFINDER_FEATURES` + Coverage 系列 + ベース指標）をデフォルト採用する。
- [x] 設定モジュールから特徴量リストを動的取得し、Strategy4 でも同一リストを使用しつつ追加列（`daily_commit_count`, `builds_per_day` など）を共通プリプロセスで合成する。
- [x] `prediction/data_preparation.py` と同等の欠損補完・型変換・スケーリング処理を共通ユーティリティとして再利用できるようにする。

### 3. モデル学習と推論
- [x] WalkForward 各 Fold で線形回帰モデルを学習し、訓練データを Fold の学習ウィンドウに限定する。
- [x] 推論では Fold ごとにモデルを保存し、検証ウィンドウへ一括予測して負値クリップと `ceil` 丸めを行い、`label_flag` が True の行のみ追加ビルド候補に残す。

### 4. WalkForward 評価・可視化
- [x] 各 Fold の予測と実測を集計し、MAE・RMSE・MAPE を Fold 別と全体平均で算出する。

### 5. 再現性とログ
- [x] `settings.RANDOM_STATE`・`USE_HYPERPARAM_OPTIMIZATION` などの設定値を Strategy4 でも尊重し、乱数系列と構成値をログへ出力する。
- [x] Fold ごとのモデルハイパー設定・使用特徴量ハッシュ・WalkForward 分割境界を JSON で出力し `docs/reports/` 配下へ保存、スケジュール DataFrame に `walkforward_fold`, `train_window_start`, `train_window_end`, `model_version`, `feature_signature` を追加する。
- [x] 再評価時に `prediction` 成果物と突き合わせられる再現パッケージ（モデル保存パス・設定ファイルコピー等）を生成する。



## リスクと緩和策
- [ ] **特徴量セット変更への追従遅れ**：`prediction/settings.py` の更新を検知する同期テストと、CI での設定ハッシュ比較を導入する。
- [ ] **WalkForward 計算コストの増大**：代表プロジェクトに限定したスモークジョブとバッチ実行運用を整備し、キャッシュ済み特徴量の再利用を検討する。
- [ ] **Fold 別データ不足**：学習サンプル閾値を定義し、閾値未満 Fold をスキップして既存戦略へフォールバックする運用を明文化する。

## ドキュメント更新
- [ ] `vuljit/RQ3/docs/additional_build_strategies_detail.md` に Strategy4 の新プロセス・使用特徴量・評価結果を追記する。
- [ ] `docs/reports/strategy4_rework_results.md` を作成し詳細レポートをまとめる。
- [ ] `prediction` モジュール README/ドキュメントへ共通化した前処理・モデル仕様を追記する。

## オープンな論点
- [ ] 追加ビルド予測を回帰で維持するか分類方式へ切り替えるか最終判断を下す。
- [ ] 特徴量の正規化・スケーリング方法を整理し、複数モデル構成での整合性を検証する。
- [ ] 評価指標へ運用コスト（ビルド時間・インフラ制約）を組み込む方針を決定する。
