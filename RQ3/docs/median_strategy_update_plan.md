# プロジェクト別中央値対応 実装計画

## 目的
1. `strategy1_median_schedule` の追加ビルド規模を、WalkForward 学習ウィンドウと同期したプロジェクト別検出遅延中央値へ置き換える。  
2. `strategy2_random_within_median_range` の乱数範囲を、同一プロジェクト・同一 Fold の検出遅延に基づく四分位範囲（Q1〜Q3）へ更新する。  
3. CLI やコア API のシグネチャ変更を避けつつ、横断的な影響（シミュレーション・分析スクリプト・既存テスト）を最小限に抑える。  

## 背景
- 既存コードは `detection_time_results.csv` から全期間のプロジェクト中央値を算出しているが、WalkForward 学習の Fold 区切りと連動していない。  
- 追加ビルド戦略も同一の時間分割を参照した統計量を用いることで、模擬運用時刻に整合するパラメータ化が可能になる。  
- 実装は `additional_build_strategies.py` を中心に行い、`core` 配下のスケジューラや CLI には呼び出し互換性を維持したまま補助情報を伝搬させる。  

## 実装ステップ

### 0. WalkForward 分割メタデータの導出
- [x] 予測 CSV（`*_daily_aggregated_metrics_with_predictions.csv`）を読み込み、`settings.N_SPLITS_TIMESERIES`（既定 10）に基づきチャンクサイズ `ceil(len / splits)` を計算するユーティリティを実装。  
- [x] 時系列分割の仕様に合わせて、「最初のチャンク＝訓練専用」「以降のチャンク＝Fold 1..K のテスト対象」となる Fold 割り当てを決定。  
- [x] `settings.USE_ONLY_RECENT_FOR_TRAINING` を参照し、累積学習（0〜i-1 チャンク）と直前チャンクのみのスライディング方式の双方に対応する訓練ウィンドウ境界（`train_start`, `train_end`）を求める。  
- [x] プロジェクトごとに `{fold_id: {"train_start": ts or None, "train_end": ts, "train_indices": [...], "test_indices": [...]}}` 形式のキャッシュを構築し、再計算を避けるためにオプションでメモ化する。  
- [x] `_prepare_labelled_timeline` など追加ビルド戦略が参照する DataFrame に Fold ID・訓練期間メタ情報を結合できるよう、インデックスや日付キーの対応方法を整理する。  

### 1. プロジェクト別 Fold 統計ヘルパーの新設
- [x] `additional_build_strategies.py` に `_compute_project_fold_statistics` を追加し、0章で得た Fold メタデータと `detection_time_results.csv` を入力として扱えるようにする。  
- [x] `detection_time_days` を `float` に変換し、負値や欠損を除外。`commit_date` を UTC Datetime として正規化する。  
- [x] 各プロジェクト × Fold について、訓練期間内 (`train_start <= commit_date <= train_end` または `commit_date <= train_end`) に存在する検出遅延値を抽出し、中央値（Median）と四分位点（Q1, Q3）を計算。  
- [x] Fold 内に有効データがない場合は Median/Q1/Q3 を `float("nan")` に設定し、後段でプロジェクト全期間の統計量あるいは全プロジェクト共通フォールバックへ切り替えられるようにする。  
- [x] プロジェクト全期間の統計量（Median, Q1, Q3）と、必要に応じて「全プロジェクト横断」のフォールバック値を追加で算出する。  
- [x] 戻り値は `{"__global__": {"median": ..., "q1": ..., "q3": ...}, project: {"__overall__": {...}, fold_id: {"median": ..., "q1": ..., "q3": ...}}}` のようなネスト構造とし、過不足なく情報を取得できるようにする。  

### 2. `strategy1_median_schedule` の更新
- [x] 既存の `median_map` 生成ロジックを撤廃し、ステップ 1 で作成した Fold 統計キャッシュを利用する。  
- [x] `_prepare_labelled_timeline` が返す DataFrame に Fold ID・訓練期間終端（`walkforward_fold`, `train_window_end` など）を追加する。  
- [x] 行ごとの Fold ID をキーに、Fold Median → プロジェクト全体 Median → 全体フォールバック Median の順に値を決定し、すべて `NaN` の場合はその行をスキップする。  
- [x] 追加ビルド数は `ceil(median_days * builds_per_day)`（`builds_per_day <= 0` の場合は `ceil(median_days)`）で算出し、`median_detection_days` フィールドに Fold Median を格納する。  
- [x] 出力レコードに Fold ID や `train_window_end` などのメタ情報を併記し、後続分析で追跡できるようにする。  
- [x] Docstring とコメントを「同一プロジェクトの WalkForward 訓練ウィンドウで算出した中央値」に更新する。  

### 3. `strategy2_random_within_median_range` の更新
- [x] ステップ 1 の統計キャッシュから、対象プロジェクト × Fold の Q1/Q3 を取得。`NaN` 場合はプロジェクト全体 → 全体フォールバックの順に四分位情報を補完する。  
- [x] Fold 内で Q1/Q3 を決定できないときはその日のスケジュール生成をスキップする（中央値だけを頼りに乱数範囲を作らない）。  
- [x] 乱数範囲は下限を Q1、上限を Q3 とし、上限が下限未満になった場合は両者を同値扱いにする。  
- [x] Fold ID とユーザー指定の `random_seed` を連結した安定ハッシュ値からシード整数を作成し、`numpy.random.Generator(np.random.PCG64(seed))` を用いて `generator.uniform(Q1, Q3)` を呼び出すことで環境に依存しない決定的乱数を生成する。  
- [x] 追加ビルド数は `ceil(offset * builds_per_day)`（`builds_per_day <= 0` の場合は `ceil(offset)`）で算出し、出力レコードには以下を含める：`offset_days_q1`、`offset_days_q3`、`sampled_offset_days`、`walkforward_fold`、`train_window_end`。  
- [x] Docstring / コメントを「同一プロジェクトの WalkForward 訓練ウィンドウに基づく四分位範囲」に言及する内容へ更新する。  

### 4. 補助関数・インターフェース整備
- [x] `_prepare_labelled_timeline` が Fold ID などを付与できるよう、必要なら追加引数（例: `fold_metadata`, `merge_on="merge_date_ts"`）を導入する。ただし戻り値の基本構造（DataFrame, label名, threshold情報, bool）は維持する。  
- [x] 既存ヘルパー内の変数名を `project_median_days`、`project_q1_days` などへ揃え、Foldごとの値を扱っている箇所が読みやすくなるよう最低限のコメントを加える。  
- [x] 戦略関数のデフォルト引数に `walkforward_splits` や `use_recent_training` を追加する場合は、既存呼び出しとの後方互換性を確保（例: `None` 時は `settings` から値を取得）。  
- [x] 新ヘルパーを `__all__` に追加する必要があるか、またはモジュール内部 util として非公開に留めるかを検討する。  

### 5. テストと検証
- [x] 既存の `tests/test_core_simulation_run.py`、`tests/test_core_metrics.py` を実行してリグレッションが無いことを確認。  
- [x] `_compute_project_fold_statistics` の単体テストを追加し、以下を検証：  
  - Fold の訓練期間フィルタが正しく適用されること。  
  - Fold 内データ不足時にプロジェクト全体・全体フォールバックへ適切に切り替わること。  
  - Q1/Q3 が順序逆転しない（必要なら補正される）こと。  
- [x] `tests/test_additional_build_strategies.py` を強化し、`_compute_project_fold_statistics` のフォールバック挙動、戦略 1 が Fold Median → プロジェクト全体 → グローバルの優先順位を持つこと、戦略 2 が Fold ID を含む NumPy PCG64 ベースの決定的乱数で IQR をサンプリングすることを確認するテストを追加した。  
- [x] 必要に応じて追加の統合テスト（例: 2 Fold 以上を持つダミープロジェクト）を用意し、出力レコードのメタ情報（Fold ID, train_window_end）が予定通り設定されるかを確認する。  

### 6. 想定リスクとフォローアップ
- [x] Fold 内のデータが極端に少ない場合、Median/Q1/Q3 が不安定になるため、フォールバック値やスキップ判定のルールをコードコメントおよびドキュメントで明示する。  
- [x] `settings.N_SPLITS_TIMESERIES` や `USE_ONLY_RECENT_FOR_TRAINING` を変えると訓練期間境界が変動するため、追加ビルド側でも同じ設定を利用することを README 等で周知する。  
- [x] 将来的に中央値以外の統計量（平均、任意分位点、ロバスト指標など）へ拡張する可能性を考慮し、Fold統計キャッシュのデータ構造は汎用的に保つ。  

#### フォローアップメモ
- `additional_build_strategies.py` 内の Fold→プロジェクト→グローバルのフォールバック順序にコメントを追加し、データ不足時はスキップする旨を明記した。  
- `README.md#RQ3 シミュレーション` に WalkForward 設定（`N_SPLITS_TIMESERIES` と `USE_ONLY_RECENT_FOR_TRAINING`）を参照するよう追記し、CLI からの設定変更時に追加ビルド側も同じ値を渡すべきことを記載した。  
- Fold 統計キャッシュは `median/q1/q3` をキーとする辞書を維持しており、今後の統計値追加時もフィールドを拡張するだけで再利用できる。  

## 完了条件
- `strategy1_median_schedule` と `strategy2_random_within_median_range` が WalkForward 学習ウィンドウに合わせたプロジェクト別統計量（Median / Q1 / Q3）を利用する。  
- 既存テストに加えて新設テストがすべて成功し、乱数挙動も決定論的に再現できる。  
- ドキュメント・コードコメント・計画書が最新仕様（プロジェクト別四分位範囲）と整合している。  
