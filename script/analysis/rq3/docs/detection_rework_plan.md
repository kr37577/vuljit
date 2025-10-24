# 追加ビルドシミュレーション検出ロジック再設計計画

目的: 追加ビルドが検出ビルド番号に到達した瞬間のみを TP と判断し、それ以降の追加ビルドは FP．

---
## 1. 前処理・基礎データ整備
- ✅ `_baseline_detection_metrics` で推定した `baseline_detection_build_number` を整数型で保持できるよう確認し、欠損プロジェクトはフォールバック値（`inf`）を返す。
- ✅ `baseline_df` から `builds_per_day` を取得し、シミュレーション用に `project -> (threshold, build_rate)` のマップを生成する。
- ✅ 検出ウィンドウ (`detection_window_days`) 設定がある場合、ベースラインの進行にも適用できるようスライディングウィンドウの仕組みを事前に検討。

## 2. タイムライン生成
- ✅ 各プロジェクトについて、追加ビルドイベントを日付順に整列。
- ✅ `last_processed_date` と `baseline_progress`(float) を保持し、イベント間の日数差 × `builds_per_day` で通常ビルドの累積を更新。
- ✅ 検出ウィンドウが有効な場合に備え、通常ビルドの寄与を `deque[(date, builds)]` として管理し、ウィンドウから外れた分を差し引ける仕組みを盛り込む。
- ✅ イベントが存在しないプロジェクト（追加ビルドゼロ）は、ベースラインのみで検出されたものとして `detections_baseline_only` に計上し、イベントログにはレコードを追加しない方針をコメントで明示。

## 3. 検出判定ステートマシン
- ✅ プロジェクトごとに以下の状態を保持:
  - `detection_threshold` (検出ビルド番号)
  - `baseline_progress` (通常ビルドの累積)
  - `baseline_history` (ウィンドウ減算用の deque)
  - `detected` フラグ (`None`=未検出, `'baseline'`, `'additional'`)
  - `triggered_event_index` (追加ビルドで検出したイベント ID)

- ✅ 各イベント処理フロー:
  1. **ベースライン進行**: 現イベント日までの経過日数で `baseline_progress` を更新。
  2. **ウィンドウ処理**（オプション）: 検出ウィンドウを導入する場合、`baseline_history` と追加ビルドのキューからウィンドウ外を削除し、`baseline_progress` も減算する。
  3. **通常ビルド判定**: `baseline_progress >= threshold` かつ `detected is None` の場合、`detected = 'baseline'` とし、以降の追加ビルドは `baseline_only` で扱う。
  4. **追加ビルド検証**: `detected is None` のとき、イベントの `scheduled_additional_builds` を一時的に累積に足し込んで閾値到達をチェック。
     - 到達した場合: イベントを `classification='tp'` とし、`detected='additional'`、`detections_with_additional += 1`。イベント内で消費したビルド数を `consumed_builds` に記録し、同イベント内の未消費分を `wasted_within_event` カラムへ書き出す。累積値はイベント前の `baseline_progress` に戻す（解析目的の仮想累積値である旨をコメントに残す）。
     - 到達しない場合: イベントは `classification='fp_pending'`（後で FP に確定）。累積値は元に戻す。
  5. **検出済み後の扱い**: `detected is not None` の状態で発生する以降のイベントは即座に `classification='fp_post_detection'` に設定し（イベントレコードは残す）、`wasted_triggers` に加算。

- ✅ イベント処理終了後、`detected` が `'baseline'` の場合は、全イベントを `classification='baseline_only'` として記録し、`baseline_only_triggers`／`detections_baseline_only` に集計する（成功扱いには含めないがログには残す）。
- ✅ `detected` が `None`（最後まで閾値に届かない）の場合、全イベントを `classification='fp'` に確定させ、`detections_with_additional` は 0 のまま、`detections_missing` などを増やす。
- ✅ 戦略別サマリでは
  - `success_triggers = count(classification == 'tp')`
  - `wasted_triggers = total_events - success_triggers`
  - `builds_success = sum(consumed_builds for tp)` (tp からのみ)
  - `builds_wasted = total_additional_builds - builds_success`
  - `detections_baseline_only = count(projects where detected == 'baseline')`
  - `detections_with_additional = count(projects where detected == 'additional')`

## 5. イベントログスキーマ更新
- ✅ `classification` の値を新ルールに合わせて定義（例: `tp`, `fp`, `fp_post_detection`, `baseline_only`, `expired`）。
- ✅ 追加カラム `evaluation_baseline` / `evaluation_trial`（判定時の基準累積値）を記録し、イベント内の浪費量を `wasted_within_event` で表現。
- ✅ 既存カラムで不要になった `buffer_after` 等は新出力から除外し、ドキュメントで置き換えを明記。

## 6. テスト計画
- ✅ **TP 判定**: ベースラインが 4/5、追加ビルド 2 のイベントで検出成立 → 当該イベントが `tp`、以降 `fp_post_detection`。
- ✅ **FP 判定**: すべてのイベントが閾値に届かず → 全イベント `fp`、`detections_with_additional = 0`。
- ✅ **Baseline Only**: イベント前のベースライン進行で閾値到達 → イベントは `baseline_only`、`detections_baseline_only = 1`。
- ✅ **Post Detection FP**: 1 つ目で検出、2 つ目以降は自動 FP。
- ✅ **ウィンドウ挙動**: ウィンドウ設定で古いイベントが `expired` になり、検出判定から除外される。
- ✅ 戦略サマリの新カラムが期待どおりに計算されることを確認。

## 7. ドキュメント更新
- ✅ `additional_builds_sim_spec.md` に新しい判定ロジックを詳細記述。
- ✅ `phase6_results.md` や CLI ヘルプに、検出が追加ビルド単位で判断されること、ベースラインのみ到達時は成功と数えないことを明記。

## 8. 実装手順のまとめ
1. 既存 `_summarize_wasted_builds` を分割し、ステートマシン処理・後処理を導入。
2. イベント記録およびサマリ生成を新しい分類に合わせて書き換え。
3. テストケースをリプレイス／追加。
4. ドキュメントを更新。
5. ローカルでシミュレーションを実行し、旧出力との比較で差分を検証。

---
この計画に基づいて実装を進める前に、方針のフィードバックをもらえると助かります。
