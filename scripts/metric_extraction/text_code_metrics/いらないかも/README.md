# scripts/metric_extraction/text_code_metrics

このディレクトリはコミットメトリクス抽出とテキスト系特徴量生成の補助スクリプト群を含みます。
ここにある主要スクリプトと役割は以下の通りです。

## 各ファイルの説明
- `ex2.sh` — SLURM ジョブ用のラッパーシェルスクリプト。プロジェクト一覧 (`project_dir.txt`) を読み、プロジェクトごとに `vccfinder_commit_message_metrics.py` を呼び出します（`PYENV` 経由で Python 実行）。ログ／エラー出力先のディレクトリを作成し、個別プロジェクトの失敗を無視して続行するオプションを持ちます。
- `vccfinder_commit_message_metrics.py` — 指定プロジェクトのコミットメトリクス CSV を読み込み、コミットメッセージに対する TF-IDF（上位10語など）を計算して CSV に追加します。
- `text_raw_convert_metrics.py` — `files_term_freq` / `patch_term_diff` といった JSON 文字列カラムを展開して、列ごとに整数化したCSVを作るユーティリティ。
- `get_feature_commit_func.py`, `vccfinder_metrics_calculator.py`, `text_metrics_calculator.py` など — コミットの特徴量抽出ロジック群（未実装/部分実装の関数があるため、実行前に内容の補完が必要です）。
- `project_dir.txt` / `project_dir_test.txt` — 処理対象プロジェクト名の一覧ファイル。
- `label.py` — 脆弱性情報（OSS-Fuzz 由来の CSV）を参照してコミットに `is_vcc` ラベルを付与するスクリプト。
- `repo_commit_processor_test.py` 等 — リポジトリ処理のための並列処理ラッパー（改善案が含まれています）。

## 使い方（簡単な流れ）
1. プロジェクトごとのコミットメトリクスが `output/<project>/<project>_commit_metrics_with_vulnerability_label.csv` のように用意されていることを確認します（`label.py` を先に使うケース）。
2. SLURM 環境でまとめて回す場合は `ex2.sh` を sbatch に渡します。あるいはローカルで単一プロジェクトを処理する場合は `vccfinder_commit_message_metrics.py -p <project>` を実行します。

## csh（ユーザーのシェル）での実行時メモ
スレッド数やハッシュ種を固定して再現性を高める場合は、ジョブ開始前に以下の環境変数を設定してください（csh/tcsh 用）：

```csh
# 乱数・ハッシュ・スレッド固定
setenv PYTHONHASHSEED 0
setenv OMP_NUM_THREADS 1
setenv OPENBLAS_NUM_THREADS 1
setenv MKL_NUM_THREADS 1
setenv NUMEXPR_NUM_THREADS 1
```

`ex2.sh` 内では `PYENV` 経由で Python を指定するようになっています。適切な Python 仮想環境が構築されていることを確認してください。

## 依存関係
少なくとも以下が必要です（バージョンを固定することを推奨します）:
- Python 3.8+
- pandas, scikit-learn, numpy
- scikit-learn, gitpython（get_feature_commit_func の実行時）
- imbalanced-learn, xgboost（使用する場合）

依存をピン固定するには `pip freeze > requirements.txt` を作成して共有してください。

## 再現性（同じ結果を得るためのポイント）
- スレッド数を1にする（上の env 指示）。
- 全ての乱数生成に同一の `random_state`（または `PYTHONHASHSEED`）を与える。
- 分割やサンプリング（KFold, RandomUnderSampler, RandomizedSearchCV 等）には `random_state` を明示的に渡す。
- ファイル読み込み後は `sort`（例: `merge_date`, `commit_hash`）で安定ソートし、glob 等は `sorted()` で順序を固定する。
- 可能であれば実行環境（OS、Python とライブラリのバージョン、BLAS 実装）を Docker で固定する。

## 推奨コマンド例
ローカル1プロジェクト実行（csh）:

```csh
setenv PYTHONHASHSEED 0
setenv OMP_NUM_THREADS 1
# 仮想環境の python を使う場合
/work/riku-ka/.pyenv/versions/py3/bin/python /work/riku-ka/vuljit/scripts/metric_extraction/text_code_metrics/vccfinder_commit_message_metrics.py -p arrow
```

SLURMでバッチ実行:

```csh
sbatch /work/riku-ka/vuljit/scripts/metric_extraction/text_code_metrics/ex2.sh
```

## 補足・注意
- リポジトリ内の一部モジュールには未実装／部分実装の関数があるため、実行前に中身を完成させる必要があります。エラー発生時はログ（`logs/`、`errors/`）を確認してください。
- 出力先パス、入力ファイル名（特に `output/<project>/...` の構成）はスクリプトによって期待値が異なります。運用する前に1プロジェクトで動作確認することを推奨します。

必要なら、README にさらに「実行例」「入出力フォーマット仕様」「テスト用小データの作り方」を追記します。要望を教えてください。
