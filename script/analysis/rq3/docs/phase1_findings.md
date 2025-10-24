# フェーズ1 成果サマリ

## 1. 既存JIT予測パイプラインの処理フローと成果物
- **入力データ**: `vuljit/data/<project>/<project>_daily_aggregated_metrics.csv` に、RQ1/RQ2で整備した特徴量群・`is_vcc` ラベル・`vcc_commit_count` が日次粒度で格納されている。列構成は [vuljit/data/arrow/arrow_daily_aggregated_metrics.csv](../data/arrow/arrow_daily_aggregated_metrics.csv) を代表例とする。
- **前処理**: `vuljit/prediction/data_preparation.py` の `preprocess_dataframe_for_within_project` が `merge_date` でソートし、`is_vcc` を数値化したうえで特徴量列を抽出する ([data_preparation.py:6](../prediction/data_preparation.py)).
- **学習・評価**: `vuljit/prediction/main_per_project.py` がプロジェクトごとに CSV を走査し、特徴量セット別に繰り返し評価を実施する ([main_per_project.py:122](../prediction/main_per_project.py))。時系列分割や推論確率の出力は `evaluation.py`／`reporting.py` に集約。
- **モデル定義**: `vuljit/prediction/model_definition.py` で RandomForest/XGBoost/Random Baseline のパイプラインとサンプリング戦略を切り替える ([model_definition.py:6](../prediction/model_definition.py))。
- **成果物**: 予測確率を付与した日次 CSV (`*_daily_aggregated_metrics_with_predictions.csv`) と fold別指標 (`exp*_per_fold_metrics.csv`)／特徴量重要度などが `vuljit/outputs/results/<model>/<project>/` に保存される。集計用ユーティリティは `vuljit/prediction/summarize_final_dataset.py` に整理済み。

## 2. RQ1/RQ2脆弱性ラベルとビルド（fuzzing回数）の対応整理
- **ラベル基盤**: `is_vcc`・`vcc_commit_count` 列は RQ1/RQ2の成果物で、全プロジェクトの集計状況を `summarize_final_dataset.py` で確認できる。
- **ビルド頻度情報**: `vuljit/rq3_dataset/project_build_counts.csv` は OSS-Fuzz `project.yaml` から抽出した `builds_per_day` を保持し、1309件中1292件がデフォルト値1、欠損なしで整備済み。
- **参照整合性**: プロジェクトIDは OSS-Fuzz 名で統一されており、`data/` 配下のディレクトリ名と `project_build_counts.csv` の `project` 列が一致するため、日次ラベルとビルド頻度をプロジェクト単位で join 可能。
- **補助データ**: シミュレーションで用いるリスク予測 (`predicted_risk_*` 列) は `outputs/results/<model>/<project>/` の CSV、および RQ3向け抽出物 `vuljit/outputs/rq3_det_outputs/` に格納されている。

## 3. 脆弱性検出日・追加ビルド日の情報源とデータ品質
- **検出日データ**: `vuljit/rq3_dataset/detection_time_results.csv`（および `vuljit/prediction/detection_time_results.csv`）が`reported_date`（検出報告日）、`commit_date`（導入日）、`detection_time_days` を保持。全3674件のうち 958件で `detection_time_days` と `commit_date` が欠損、日付はISO8601(Utc)で一貫している。
- **ギャップ**: 実際の追加ビルド実施日を直接記録した生ログは確認できず、現状は `test_start` と `builds_per_day` からスケジュールを派生させる必要がある。必要なら OSS-Fuzz テレメトリ／ClusterFuzz API から生データの取得を検討。
