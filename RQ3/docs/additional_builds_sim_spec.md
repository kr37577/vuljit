# additional_builds_cli.py 関数仕様メモ

この文書は `vuljit/RQ3/cli/additional_builds_cli.py` 内の主要関数について、目的・入出力・内部ロジック・エッジケースを整理したものです。追加ビルドシミュレーションを拡張／改修する際のリファレンスとして利用できます。

## `_ensure_directory(path: str) -> str`
- **目的**: 出力先のディレクトリが存在することを保証し、作成後のパスを返す。
- **入出力**:
  - 入力: 任意の文字列パス。
  - 出力: `os.makedirs(..., exist_ok=True)` 実行後の同じパスを返す。
- **ロジック**: `os.makedirs` でディレクトリを作成（既存でもOK）して、そのまま `path` を戻す。加工は行わない。
- **エッジケース**: パスが空文字や無効な場合は `OSError` が上がる可能性がある。呼び出し元で例外処理すること。

## `_load_detection_table(path: str) -> pd.DataFrame`
- **目的**: `detection_time_results.csv` を読み込み、プロジェクト名を正規化しつつ検出日数を数値化する。
- **入出力**:
  - 入力: CSV パス。
  - 出力: `project`, `detection_time_days` を含む DataFrame。
- **ロジック**:
  1. `pd.read_csv` で読み込み。
  2. `project` 列が存在すればそれを、なければ `package_name` を流用し、文字列トリム。
  3. プロジェクト名が空の行を除去。
  4. `detection_time_days` を `pd.to_numeric(errors="coerce")` で数値化し、欠損を落とす。
- **エッジケース**: `project` も `package_name` も無い場合は `KeyError`。`detection_time_days` 欠損行は自動で除外される。

## `_load_build_counts(path: str) -> pd.DataFrame`
- **目的**: 通常ビルド頻度（1 日あたりビルド数）テーブルを読み込む。
- **入出力**:
  - 入力: CSV パス。
  - 出力: `project`, `builds_per_day` を含む DataFrame。
- **ロジック**:
  1. CSV 読み込み。
  2. `project` 列の空白をトリムし、空の行を除去。
  3. `builds_per_day` を `pd.to_numeric(errors="coerce")` で数値化し、NaN 行を除去。
- **エッジケース**: ビルド頻度が 0 の行はそのまま残るので、しきい値計算で `inf` 扱いになる可能性がある。

## `_normalize_to_date(series: pd.Series) -> pd.Series`
- **目的**: タイムスタンプ列を UTC → 日付フロア → タイムゾーンなしの datetime64 へ変換する。
- **ロジック**:
  1. `pd.to_datetime(..., utc=True)` でタイムゾーン付きの Timestamp に変換。
  2. `.dt.normalize()` で日付に丸める（時刻切り捨て）。
  3. `.dt.tz_convert(None)` でタイムゾーン情報を剥がす。
- **エッジケース**: 変換不能な値は NaT になる。呼び出し元の `dropna` に依存。

## `_prepare_schedule_for_waste_analysis(df: pd.DataFrame) -> pd.DataFrame`
- **目的**: 戦略のスケジュールを「プロジェクト×日付×追加ビルド数」の形式に正規化する。
- **ロジック**:
  1. 空 DataFrame はそのまま返却。
  2. `merge_date_ts` > `merge_date` の優先度で日付列を選ぶ。
  3. `_normalize_to_date` で `schedule_date` を生成。
  4. `scheduled_additional_builds` を数値化し NaN を 0 で埋める。
  5. `project` をトリムし、`project`/`schedule_date` 欠損行を落とす。
- **出力**: 追加ビルド数が `float` 化された DataFrame。
- **エッジケース**: 日付列が存在しない場合は `schedule_date` に NaT を入れたまま返す。後段で既存ロジックが `dropna` を行う。

## `_safe_ratio(numerator: float, denominator: float) -> float`
- **目的**: 0 除算を回避しつつ比率を計算するユーティリティ。
- **ロジック**: 分母が真偽値 False (0.0) の場合は `nan` を返す。

## `_clip_detection_days(value: float) -> float`
- **目的**: 検出日数を 0 以上に丸め、非数値は NaN にする。
- **ロジック**: `math.isfinite` で有限かを確認し、負値は 0 に置換。

## `_estimate_detection_build_number(days: float, builds_per_day: float) -> float`
- **目的**: 検出に要するビルド番号を `ceil(max(days,0) * builds_per_day)` で推定する。
- **ロジック詳細**:
  - `days` も `builds_per_day` も有限でなければ NaN。
  - `builds_per_day <= 0` の場合は NaN（頻度が無いプロジェクトは推定不能）。
  - 積が 0 以下（0 日や 0 ビルド/日）なら最小値として 1.0 を返す。
  - それ以外は `math.ceil` で整数ビルド番号を返却。

## `_baseline_detection_metrics(detection_df, build_counts_df) -> pd.DataFrame`
- **目的**: 検出テーブルとビルド頻度を結合し、ベースラインの検出日数・ビルド数・推定ビルド番号を構築する。
- **ロジック詳細**:
  1. `detection_df` をコピーし、`detection_time_days` を `_clip_detection_days` で 0 以上に補正。
  2. プロジェクト単位の中央値で `baseline_detection_days` を算出。
  3. `build_counts_df` と左結合。
  4. `builds_per_day` 欠損は 0 埋め。
  5. `baseline_detection_builds = baseline_detection_days * builds_per_day` を計算。
  6. `_estimate_detection_build_number` を使って `baseline_detection_build_number` を追加。
- **戻り値**: 以上の列を含む DataFrame。ビルド頻度が無い場合はビルド番号が NaN になる。

## `_build_threshold_map(baseline_df: pd.DataFrame) -> Dict[str, float]`
- **目的**: プロジェクトごとの検出閾値（累積ビルド数）を辞書形式で取得する。
- **ロジック**:
  1. 各行について `baseline_detection_build_number` を最優先。
  2. 無効なら `baseline_detection_builds` を、さらに無効なら `days * builds_per_day` や `days` をフォールバック。
  3. 有効な値が得られない場合は `inf` に落とす。
- **エッジケース**: `project` 空白行はスキップ。`builds_per_day` などが NaN でも `inf` を返せる。

## `_prepare_project_metrics(schedules, baseline_df) -> pd.DataFrame`
- **目的**: 戦略×プロジェクト単位の追加ビルド指標を算出し、ベースライン情報と結合する。
- **ロジック**:
  1. 各戦略で `summarize_schedule_by_project` を呼び、結果をベースラインとマージ。
  2. 列名を `trigger_count` などへ正規化。
  3. トリガ平均 / 推定残余日数 / 残余ビルドを計算。
  4. 結果を縦方向に結合。
- **備考**: `summarize_schedule_by_project` は 追加ビルドミニマルシミュレーターから提供されるヘルパ。

## `_aggregate_strategy_metrics(project_df) -> pd.DataFrame`
- **目的**: プロジェクトメトリクスを戦略レベルで集約し、総ビルド数や平均・中央値を整理する。
- **ロジック**:
  - 戦略ごとに `scheduled_builds`、`estimated_detection_days/builds` の平均・中央値、節約ビルド数などを算出し、リストへ追加。
  - DataFrame 化して返す。

## `_prepare_daily_totals(schedules) -> pd.DataFrame`
- **目的**: 戦略×プロジェクト×日単位の追加ビルド総数を作成する。
- **ロジック**:
  - 各戦略で `merge_date_ts` → datetime 化 → `groupby([project, date]).sum()`。
  - 空のときはレコード無しのテンプレート DataFrame を返す。

## `_summarize_wasted_builds(schedules, baseline_df, detection_window_days)`
- **目的**: 推定した検出ビルド番号に対し、追加ビルドイベント単位で TP/FP を判定し、戦略別に浪費量を集計する。
- **ロジック詳細**:
  1. `_build_threshold_map` で閾値を、ベースライン表から `builds_per_day` を読み出し、`project -> (threshold, rate)` の辞書を用意する。
  2. `_prepare_schedule_for_waste_analysis` で戦略のスケジュールを正規化し、プロジェクトごとに日付順へソート。
  3. プロジェクト内のイベントを走査し、前回イベントからの経過日数に応じて `baseline_progress` を加算する。検出ウィンドウが指定されている場合は、`deque[(date, builds)]` で保持している寄与からウィンドウ外の分を減算する。
  4. ベースライン寄与だけで閾値に達した時点で `detected_state='baseline'` とし、以降のイベントは `baseline_only` として記録する。
  5. 未検出状態では、当該イベントの追加ビルド量を一時的に累積して閾値到達を判定し、
     - 到達した場合は `classification='tp'`、消費量を `consumed_builds` に、未使用分を `wasted_within_event` に記録。検出済み状態へ移行し、以後のイベントは `fp_post_detection` とする。
     - 到達しない場合は `classification='fp'` とし、追加ビルド全量を `wasted_within_event` と見なす。
  6. イベントレコードは (`strategy`, `project`, `schedule_date`, `threshold`, `scheduled_builds`, `consumed_builds`, `wasted_within_event`, `success`, `expired`, `detection_id`, `classification`, `evaluation_baseline`, `evaluation_trial`) を持つ。
  7. 戦略単位では `baseline_only_triggers` を含むサマリを構築し、成功率や浪費率、`detections_with_additional` / `detections_baseline_only` を併せて出力する。
- **出力**: `(summary_df, events_df)`。イベント表は `tp` / `fp` / `fp_post_detection` / `baseline_only` 等の分類で 1 行ずつ記録される。
- **注意事項**: 追加ビルドの寄与は判定後に累積へ残さないため、検出ウィンドウは通常ビルド寄与のみに影響する。

## `_plot_additional_builds_boxplot(project_df, output_dir)`
- **目的**: 戦略ごとのプロジェクト追加ビルド分布を箱ひげ図で可視化する。
- **ロジック**:
  - 戦略ごとに `scheduled_builds` 列を抽出し、`plt.boxplot(showfliers=False)` で描画。
  - `additional_builds_boxplot.png` を保存してパスを返す。空データなら None。

## `parse_args() -> argparse.Namespace`
- **目的**: CLI 用の引数パーサを構築。主要パラメータは `--predictions-root`, `--risk-column`, `--label-column`, `--risk-threshold`, `--detection-table`, `--build-counts`, `--output-dir`, `--detection-window-days`, `--silent`。

## `main() -> None`
- **目的**: 追加ビルドシミュレーションの CLI エントリーポイント。
- **処理手順**:
  1. 引数解釈・出力ディレクトリ作成。
  2. 検出テーブル・ビルド頻度を読み込み、ベースラインメトリクスを構築。
  3. Phase 3 戦略を `run_minimal_simulation` で実行し、サマリ/スケジュールを取得。
  4. プロジェクトメトリクス・戦略集約・日別集計を作成。
  5. `_summarize_wasted_builds` で TP/FP 分析を実施。
  6. すべての結果を CSV / PNG に出力し、`--silent` でなければ標準出力に要約を表示。
- **留意事項**: `load_detection_baseline` からの追加統計をサマリに埋め込む。検出ウィンドウ指定は0未満だと切り上げ。

