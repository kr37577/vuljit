# RQ3 モジュール再編計画

## 目的
フェーズ 3〜5 のシミュレーション関連ロジックを責務ごとに分割し、再利用性・テスト容易性・CLI の一貫性を高める。現状の `simulate_additional_builds.py` / `minimal_simulation_wrapper.py` に加え、`threshold_precision_analysis.py` や `timeline_cli_wrapper.py` でもデータ入出力や戦略呼び出しが重複しているため、共通の `core` 層を整備し CLI/分析用スクリプトから再利用する構成へ移行する。

```
RQ3/
  core/
    __init__.py
    io.py             # ファイル入出力・パス解決・デフォルト設定
    baseline.py       # 検出ベースライン算出・閾値テーブル構築
    predictions.py    # 予測ファイル列挙・読み込みヘルパー（フェーズ横断で共有）
    scheduling.py     # 戦略呼び出しラッパー・threshold 変換
    simulation.py     # run_minimal_simulation・浪費判定
    metrics.py        # 集計ヘルパー・安全な比率計算
    plotting.py       # 可視化ユーティリティ
    timeline.py       # build_timeline ロジック
  cli/
    additional_builds_cli.py    # 旧 phase5_simulation CLI
    minimal_simulation_cli.py   # 旧 phase5_minimal_simulation CLI
    phase4_main.py              # threshold_precision_analysis の CLI ラッパー
    build_timeline_cli.py       # build_timeline CLI
  data/            # 既定で参照する CSV/モデル
  docs/            # ドキュメント類（既存を維持）
  tests/           # 既存テストを維持、必要に応じて追加
```

## 前提確認
- `core/` ディレクトリは既に存在し空。`__init__.py` で段階的にエクスポートを整理する。
- `additional_build_strategies.py` は戦略実装のソースとして存続させ、呼び出しは `core.scheduling` 経由に切り替える。
- CLI スクリプトは最終的に `python -m RQ3.cli.additional_builds_cli` のようにモジュール呼び出しできる構成へ統一する。

## 作業ステップ

### フェーズ 0: 足回りの整備（完了）
- [✅] `RQ3/core/__init__.py` を作成し、主要ファサード関数を公開する準備をする。
- [✅] `RQ3/cli/` ディレクトリを新設。
- [✅] 将来的に `RQ3/data/` を利用する場合に備え、ディレクトリのみ作成（必要であれば `.gitkeep` を配置）。

### フェーズ 1: 共通入出力レイヤーの抽出
- [✅] `simulate_additional_builds.py` / `minimal_simulation_wrapper.py` / `threshold_precision_analysis.py` / `timeline_cli_wrapper.py` から重複する `_ensure_directory`, `_load_detection_table`, `_load_build_counts`, `_parse_build_counts`, `_iter_prediction_files` などを `core/io.py` と `core/predictions.py` に分離する。
- [✅] `DEFAULT_OUTPUT_DIR` など CLI 依存の定数は `core.io.DEFAULTS`（辞書）として集中管理し、各 CLI では `core.io.resolve_default_path` 経由で取得する。
- [✅] 例外時の扱いや `os.path` ベースの相対パス連結も `core.io` に集約し、テスト可能な関数にする。
- [✅] `core/__init__.py` で `from .io import DEFAULTS, ensure_directory, load_detection_table` のように公開する。

### フェーズ 2: ベースライン・タイムライン基盤の再配置
- [✅] `_baseline_detection_metrics`, `_build_threshold_map` と関連補助関数を `core/baseline.py` へ移し、`threshold_precision_analysis` と追加ビルド系 CLI から統一利用する。
- [✅] `timeline_cli_wrapper.py` の `_parse_build_counts` / `_scan_daily_records` / `_compute_timeline` を `core/timeline.py` に移し、CLI 側は I/O のみを担当する。
- [✅] `timeline` ロジックでは dataclass 依存を排除し、辞書ベース or `NamedTuple` で返却することで pandas 変換を容易にする。
- [✅] `core/__init__.py` で `baseline` / `timeline` の主要 API を再エクスポートする。

### フェーズ 3: 戦略呼び出しとシミュレーション土台の整理
- [✅] `core/scheduling.py` を新設し、`additional_build_strategies` の関数を薄くラップして例外処理・引数デフォルトを統一する（例: `get_strategy("median", **kwargs)`）。
- [✅] `core/predictions.py` で `threshold_precision_analysis` が使用している予測ファイル列挙・読み込み処理を提供し、`additional_build_strategies` 内でも可能であれば再利用する。
- [✅] `minimal_simulation_wrapper` の `run_minimal_simulation`, `summarize_schedule`, `summarize_schedule_by_project` を `core/simulation.py` / `core/metrics.py` へ移し、CLI／他フェーズから直接利用できるようにする。
- [✅] `core/simulation` では `SimulationResult` のようなデータクラス（summary, schedules, metadata）を導入し、`simulate_additional_builds` と `threshold_precision_analysis` の双方で同じ結果構造を共有する。

### フェーズ 4: 集計・可視化の細分化
- [✅] `simulate_additional_builds` から `_prepare_project_metrics`, `_aggregate_strategy_metrics`, `_prepare_daily_totals`, `_safe_ratio` を `core/metrics.py` へ移管し、総計関数を API 化する。
- [✅] `_summarize_wasted_builds`, `_prepare_schedule_for_waste_analysis` も `core/simulation` に集約し、内部で `core.metrics` を利用する構造へ整理する。
- [✅] `_plot_additional_builds_boxplot` 等の matplotlib 依存コードを `core/plotting.py` として抽出し、戻り値で生成ファイルパスを返すよう共通化する。
- [✅] Matplotlib 依存を避けたいテスト向けに `core.plotting` で `if not plt:` のようなインポートガードを整備する。

### フェーズ 5: CLI エントリポイントの再構築
- [✅] `simulate_additional_builds.py` を `cli/additional_builds_cli.py` へ移動し、`core` の API に依存する薄い CLI として書き換える。
- [✅] `minimal_simulation_wrapper.py` を `cli/minimal_simulation_cli.py` に移す。同時にルート直下には互換ラッパーを残す。
- [✅] `threshold_precision_analysis.py` は CLI とロジックを分割し、`cli/phase4_main.py`（argparse 担当）から `core` API を呼び出す形に揃える。
- [✅] `timeline_cli_wrapper.py` も `cli/build_timeline_cli.py` に移し、既存スクリプトは互換ラッパーとして提供する。

### フェーズ 6: 既存コードの移行調整
- [✅] `additional_build_strategies.py` 内で `os.path` やファイル読み込み処理が残っている場合は `core.io` / `core.predictions` の関数を参照するように置換し、重複ロジックを削減する。
- [✅] 各 CLI からのインポートを相対パス（`from RQ3.core import simulation` 等）に統一し、`try/except ImportError` ブロックは極力削除する。
- [✅] `DEFAULT_*` 定数参照箇所をすべて `core.io.DEFAULTS[...]` へのアクセスに切り替える。

### フェーズ 7: テストと検証
- **テスト基盤整備**
  - [✅] `tests/fixtures/` に最小構成の CSV / JSON / ディレクトリ構造を用意し、`conftest.py` で `tmp_path` をラップするフィクスチャ（例: `phase5_dataset`）を提供する。
  - [✅] 既存テスト `tests/test_core_simulation_run.py` を新しいフィクスチャ利用に書き換え、期待値算出の計算根拠をコメントで明示する。
- **モジュール単体テスト**
  - [✅] `core.io`, `core.baseline`, `core.metrics`, `core.simulation`, `core.timeline`, `core.plotting`, `core.predictions`, `core.scheduling` それぞれについて、成功パス・異常系（ファイル欠損、値の欠落、Matplotlib 非導入など）を検証するテストを作成し、関数名毎にカバレッジを確保する。
  - [✅] `SimulationResult` の直列化（`dict()` 変換や空スケジュール）など境界ケースを `tests/test_core_simulation_run.py` に追加する。
- **CLI スモークテスト**
  - [✅] `python -m RQ3.cli.additional_builds_cli` / `minimal_simulation_cli` / `phase4_main` / `build_timeline_cli` を PyTest から `subprocess` で起動し、`tmp_path` に成果物（CSV / PNG）が生成されること、異常パラメータで適切に失敗することを確認する。
  - [✅] CLI 実行時に実データへアクセスしないよう、全入出力をフィクスチャで差し替えるモック層を整備する（例: `monkeypatch` で `run_minimal_simulation` や `collect_predictions` を置換）。
- **統合テスト / 回帰検証**
  - [✅] フェーズ 5 完全フロー用の縮小データセット（数件のプロジェクトと日付）を用意し、`python -m RQ3.cli.additional_builds_cli` を実行して生成された CSV の主要列を Snapshot テストで検証する。
  - [✅] フェーズ 4 精度曲線出力についても同様の縮小データで回帰テストを追加し、主な統計値（閾値・precision/recall）が既知の値と一致することを確認する。


### フェーズ 8: ドキュメントと設定反映
- [ ] `docs/` 内の仕様書（例: `additional_builds_sim_spec.md`）や README の import 例を新パッケージ構成に合わせて更新する。
- [ ] `setup.cfg` / `pyproject.toml` が存在する場合は `RQ3.core` / `RQ3.cli` をパッケージとして認識させる設定（`packages = find:` など）を確認・修正する。
- [ ] 不要になったバックアップスクリプトを整理し、`legacy_snapshots/` の取り扱い方針を決定する。

## リスクと検討事項
- `additional_build_strategies` はファイル I/O/乱択ロジックが複雑なため、`core.scheduling` での薄いラップに留めるか、戦略自体をモジュール分割するかをフェーズ 6 のタイミングで再評価する。
- Matplotlib 非依存環境（CI）でグラフ生成テストをどう扱うか（モック化 or スキップ条件の導入）。
- 既存 CLI のユーザが直接 `python simulate_additional_builds.py` を実行している可能性があるため、互換ラッパーをどこまで残すか決定する必要がある。

## 次のアクション
- フェーズ 6 のタスク着手：`additional_build_strategies.py` の I/O ロジックの共通化や、CLI からのインポート統一、`DEFAULTS` 参照の置き換えを進める。
