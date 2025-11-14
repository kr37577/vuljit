# Patch-Coverage 入力整合性 改修計画（詳細版）

## ゴール
2024-XX-XX 時点で観測されている「OSS-Fuzz プロジェクト名 (`project_name`) と `repo_name` が一致しないために `*_patch_coverage.csv` が生成されない」問題を根本解消し、以下を満たす。

1. すべての `revisions_with_commit_date_<project>.csv` が `project` 列を持つ。
2. 各パイプラインステージ（diff 生成～パッチカバレッジ算出）が `project` 列で対象をフィルタし、`repo_name` はリポジトリ解決専用に使う。
3. 旧フォーマットの CSV（`project` 列なし）が混在しても、フォールバックで破綻しない。
4. `datasets/derived_artifacts/patch_coverage_metrics/<project>/<project>_patch_coverage.csv` が全対象プロジェクトで非空生成され、`scripts/modeling/aggregate_metrics_pipeline.py` が参照できる。

---

## 1. 調査フェーズ

### 1.1 現状把握
- コマンド:
  - `python3 scripts/metric_extraction/patch_coverage_pipeline/tools/list_repo_names.py`（後述で作成）で各 CSV に含まれる `repo_name` 種類と件数を一覧化。
  - `ls data/intermediate/cloned_repos` でクローン済みリポジトリが揃っているかチェック。
- 成果物:
  - `notes/patch_coverage_repo_inventory.md`（新規）に、`project_name`→`repo_name` 一覧と欠損クローンの有無を記載。

### 1.2 影響範囲の洗い出し
- 以下のスクリプトで `repo_name == project_name` の前提がある箇所を `rg` で確認し、ファイルと行番号を一覧化。
  - `scripts/metric_extraction/patch_coverage_pipeline/create_daily_diff.py`
  - `scripts/metric_extraction/patch_coverage_pipeline/process_diffs.py`
  - `scripts/metric_extraction/patch_coverage_pipeline/run_culculate_patch_coverage_pipeline.py`
  - `patch_coverage_culculater/calculate_patch_coverage_per_project.py`（外部利用分）
- 結果を `notes/patch_coverage_repo_inventory.md` に追記しておく。

---

## 2. 実装フェーズ

### 2.1 `create_project_csvs_from_srcmap.py`
1. `generate_revisions()` の戻りレコードへ `project` 列を追加。
2. CSV ヘッダーを `['project', 'date', 'repo_name', 'url', 'revision']` に変更。
3. 既存 `DEFAULT_OUTPUT_PREFIX` の説明に「project 列を含む」と追記。
4. 単体テストスクリプト `tests/patch_coverage/test_create_project_csvs.py` を追加し、仮の srcmap JSON から期待通りの CSV が出るか検証。

### 2.2 `revision_with_date.py`
1. `pd.read_csv()` 直後に以下を実施:
   ```python
   if 'project' not in df.columns:
       df['project'] = project_name  # CSV ファイル名から推定
   ```
2. `to_csv()` 前に列順を `['project', 'date', 'repo_name', 'url', 'revision', 'commit_date']` に統一。
3. ログに「project 列が無い場合はフォールバックした」旨を表示。
4. CLI ヘルプへ `project` 列追加の注意書きを記載。

### 2.3 `prepare_patch_coverage_inputs.py`
1. `generate_revisions()` の戻り値が `project` 列を含む前提でドキュメント更新。
2. `--skip-commit` 時にテンポラリディレクトリへ書き捨てる既知問題を修復（`cleanup` フラグを `not args.skip_commit` 条件に変更）。
3. 実行完了時に「生成された CSV に project 列が存在するか」をチェックし、欠落していれば警告。

### 2.4 差分生成・パッチ計算系
#### 2.4.1 `create_daily_diff.py`
- `required_cols` に `project` を追加。
- `df = df[df['project'] == project_name]` へ修正し、フィルタ結果をログ。
- `repo_name` ごとに `git diff` するため、日別ループ内で `current_row['repo_name']` からローカルパスを構築。

#### 2.4.2 `process_diffs.py`
- CSV 解析時に `project` 列を強制。
- フィルタとログも `project` ベースに合わせる。
- 既存差分ファイルの命名が `repo_name` を含む場合の検証を追加。

#### 2.4.3 `calculate_patch_coverage_per_project.py`
- 入力ファイルに `project` 列が無い際のフォールバック実装。
- `repo_local_path = repos_dir / row['repo_name']` とし、`url` ベースの推測は警告用途に限定。
- 日付スキップ機構を `(date, repo_name)` 単位に拡張する TODO をコードコメントに記載。

#### 2.4.4 `run_culculate_patch_coverage_pipeline.py`
- 上記と同様のフィルタ・ローカルパス修正。
- 並列ワーカーへ `repo_name` を引き渡し、ログにも表示。
- `skipped_dates` を `(date, repo_name)` セットに変更（再実行安全性向上）。

### 2.5 付随スクリプト
- `submit_patch_coverage_jobs.sh` や `run_shell_for_patch_projects.sh` に `project` 列必須化の注意を追記。
- `patch_coverage_pipeline/README.md`（新規）を作り、実行順序と環境変数を整理。

---

## 3. データ更新フェーズ

### 3.1 再生成
1. `create_project_csvs_from_srcmap.py --root datasets/raw/srcmap_json --out /tmp/revisions_projected`.
2. `revision_with_date.py --src /tmp/revisions_projected --out datasets/derived_artifacts/patch_coverage_inputs`.
3. 既存ファイルをバックアップ (`datasets/derived_artifacts/patch_coverage_inputs_backup_YYYYMMDD` など)。

### 3.2 暫定マイグレーション
- 再生成が困難な場合は次のスクリプトで `project` 列を追加。
  ```bash
  python3 scripts/metric_extraction/patch_coverage_pipeline/tools/add_project_column.py \
    --inputs datasets/derived_artifacts/patch_coverage_inputs
  ```
- 上記スクリプトは CSV ファイル名から `<project>` を推定して全行に埋める。

### 3.3 バリデーション
- `python3 scripts/metric_extraction/patch_coverage_pipeline/tools/validate_project_column.py` で以下をチェック:
  - `project` 列が全行に存在。
  - `repo_name` と `project` がすべて空でない。
  - `project` ごとの `repo_name` 数が調査時の想定と一致 or 逸脱を警告。

---

## 4. 動作確認フェーズ

### 4.1 単独スクリプトテスト
- `pytest tests/patch_coverage/test_create_project_csvs.py`
- `pytest tests/patch_coverage/test_revision_with_date.py`

### 4.2 E2E テスト
1. プロジェクト例: `apache-commons`, `php`, `skia`, `libyal`.
2. コマンド:
   ```bash
   python3 scripts/metric_extraction/patch_coverage_pipeline/run_culculate_patch_coverage_pipeline.py \
     --project apache-commons --workers 1
   ```
3. 成功条件:
   - `datasets/derived_artifacts/patch_coverage_metrics/apache-commons/apache-commons_patch_coverage.csv` が生成され、`repo_name` が 1 つでも `project` と異なる行を持つ。
   - ログに `repo_name` を含む進捗表示が出る。

### 4.3 集約パイプライン確認
- `scripts/modeling/aggregate_metrics_pipeline.py --project apache-commons ...` を実行し、`patch_coverage_file` が読み込めることを確認。
- `patch_sum_by_date` に `NaN` が残らないか手動でチェック。

---

## 5. ドキュメント / 周知

1. `scripts/data_storage_locations.md` とリポジトリ `README.md` に `project` 列追加と再生成手順を追記。
2. `scripts/metric_extraction/patch_coverage_pipeline/README.md`（新規）で実行順序と要件を整理。
3. Slack / ドキュメントに「再生成が必要」「古い CSV は非対応」などのリリースノートを共有。

---

## 6. フォローアップ / 改善案

- `(date, file_path)` 単位でスキップ管理する仕組みの設計（部分的に失敗した日の再処理を容易にする）。
- GCS 取得失敗時のリトライやエラーカテゴリ別のリカバリ機構を検討。
- `repo_name` ↔ `project_name` のマッピング CSV を `datasets/reference_mappings/ossfuzz_repo_mapping.csv` として定期更新し、将来的な自動検証に備える。

---

## 付録: 補助スクリプト案

### list_repo_names.py
```python
from pathlib import Path
import csv

base = Path('datasets/derived_artifacts/patch_coverage_inputs')
for csv_path in sorted(base.glob('revisions_with_commit_date_*.csv')):
    with csv_path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        repo_names = {row.get('repo_name', '').strip() for row in reader}
    print(f\"{csv_path.name}: repos={len(repo_names)} sample={list(repo_names)[:5]}\")
```

### add_project_column.py
```python
import argparse, csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', required=True)
args = parser.parse_args()

for path in Path(args.inputs).glob('revisions_with_commit_date_*.csv'):
    project = path.stem.replace('revisions_with_commit_date_', '')
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if 'project' in fieldnames:
            continue
        rows = list(reader)
    if not rows:
        continue
    fieldnames = ['project'] + fieldnames
    for row in rows:
        row['project'] = project
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    print(f\"added project column: {path}\")
```

---

この計画に沿って実装→データ更新→検証→周知を完了させることで、`repo_name` と `project_name` の不一致によるパッチカバレッジ欠損問題を解消できる。
