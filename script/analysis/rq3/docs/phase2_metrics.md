# フェーズ2 指標定義と閾値調整メモ

## 1. ビルド履歴整形スクリプト
- 実装場所: `vuljit/RQ3/build_timeline.py`。`project_build_counts.csv`（`vuljit/rq3_dataset/`）と `vuljit/data/<project>/<project>_daily_aggregated_metrics.csv` を入力に、時系列ごとのビルド番号と推定 fuzzing 実行回数を生成。
- 使用例: `python3 vuljit/RQ3/build_timeline.py --output-dir <out_dir> --fuzzing-multiplier 1.0`。出力はプロジェクトごとの `*_build_timeline.csv` と総括 `projects_summary.csv`。
- 出力カラム:
  - `build_index_start` / `build_index_end`: 当日ビルドの連番範囲。
  - `cumulative_builds`: 初日からの累積ビルド数。
  - `fuzzing_runs_daily` / `fuzzing_runs_cumulative`: 1ビルドあたり `--fuzzing-multiplier` 回の fuzz 実行を仮定した推定値。
  - `daily_commit_count`: RQ1/RQ2 元データのコミット数。存在しない場合は空。
- 想定: OSS-Fuzz の `builds_per_day` が暦日ベースで一定。実測のビルド欠損を扱う場合は `project_build_counts.csv` 側で補正する。

## 2. 指標定義（発見までの早さ / 無駄ビルド数）
- 記号:
  - `d_i`: 脆弱性 *i* の導入コミット日（`detection_time_results.csv: commit_date`）。
  - `r_i`: 脆弱性 *i* の検出日（`detection_time_results.csv: reported_date`）。
  - `B(t)`: `build_timeline.py` が出力する累積ビルド関数。日 `t` 時点の `cumulative_builds`。
  - `\tau`: 追加ビルドを発火させる判定閾値（後述）。
- **発見までの早さ（暦日）**
  - `T^{calendar}_i = (r_i - d_i)` 日。欠損のある行は除外。
- **発見までのビルド数**
  - `T^{build}_i = B(r_i) - B(d_i) + 1`。`B(r_i)`／`B(d_i)` はそれぞれ検出日・導入日の累積ビルド数。
- **無駄ビルド数**
  - 追加ビルド戦略 *s* で採用した決定時刻 `t_{s,i}` に対し、
    `W_{s,i} = B(t_{s,i}) - B(r_i)` （`B(t_{s,i}) >= B(r_i)`）。
  - 集計指標: `\text{mean}(W_{s,*})`, `\text{median}(W_{s,*})`, 任意分位点。
- **期待追加ビルド数**
  - `E_{s} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} B_{s,p}^{\text{extra}} / B_{p}^{\text{baseline}}`。
  - 基準 `B_{p}^{\text{baseline}}` は `project_build_counts.csv` の `builds_per_day` と観測期間から算出（フェーズ5で活用）。

## 3. Precision ベースの閾値調整手順
1. **予測確率の取得**: `vuljit/outputs/results/<model>/<project>/<project>_daily_aggregated_metrics_with_predictions.csv` から対象実験（例: `predicted_risk_VCCFinder_Coverage`）の確率列を抽出。欠損は除去。
2. **ラベル整合**: 同じ CSV の `predicted_label_*` 列では暫定閾値 0.5 を用いている。`is_vcc` と併せて、PR カーブ計算用の `(y_true, y_score)` ペアを作成。
3. **PR カーブ算出**: `precision_recall_curve(y_true, y_score)` で全候補閾値をスキャンし、Precision が目標値 `P^*` を上回る最小閾値 `\tau(P^*)` を選択。
4. **安定化**: 複数繰り返し `exp*_per_fold_metrics.csv` がある場合は fold ごとに `\tau(P^*)` を算出して中央値を採用。
5. **閾値の適用**: 選んだ `\tau(P^*)` を `build_timeline.py` によって得た時系列へ結合し、`predicted_risk \ge \tau(P^*)` の日を追加ビルド候補とする。
6. **監査**: Precision が所望水準より下がる場合は、`P^*` を調整するか、`predicted_risk` の平滑化（移動平均など）を検討。

## 4. 今後の実装ノート
- フェーズ3で実装する各戦略は、本ドキュメントの `T^{build}_i` / `W_{s,i}` を直接参照できるよう、ビルドタイムライン CSV と脆弱性検出ラベルの join を前提に進める。
- Precision 目標は 3～4 パターン（例: 0.6, 0.7, 0.8, 0.9）を想定し、フェーズ4で比較可能なよう `\tau(P^*)` と追加ビルド数の対応表を保存する。
