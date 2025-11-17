# Patch Coverage Pipeline Enhancement Plan

## 背景とゴール
- 一部プロジェクト（php / ghostscript / mruby など）では `revisions_with_commit_date_<project>.csv` に複数の依存リポジトリが混在し、`run_culculate_patch_coverage_pipeline.py` が `repo_name` に該当するローカルクローンを見つけられずパッチカバレッジ CSV を生成できていない。
- `oss_fuzz_project_info.py` により生成される `datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv` には OSS-Fuzz プロジェクトと canonical な `main_repo` URL がまとまっている。
- このメタデータを利用して「各プロジェクトが本来差分を取りたいリポジトリ」を特定し、パイプラインで不要な依存リポジトリ行をスキップ／適切なディレクトリにマッピングできるようにする。

## 現状調査メモ
- `c_cpp_vulnerability_summary.csv` では ghostscript/mruby/php はいずれも 1 行のみで、それぞれ `git://git.ghostscript.com/ghostpdl.git`（行2）、`https://github.com/mruby/mruby`（行8）、`https://github.com/php/php-src.git`（行15）が canonical repo として記録されている。
- `vuljit/datasets/raw/cloned_c_cpp_projects` には `ghostscript/`, `mruby/`, `php/` の本体クローンは存在するが、CSV に現れる `afl`, `libfuzzer`, `mruby_seeds`, `php-src/sapi/fuzzer` などは存在しない。
- 入力 CSV の `repo_name` 内訳は以下の通りで、依存リポジトリ行が多数を占める。  
  • ghostscript: `cups`, `freetype`, `ghostpdl`, `aflplusplus`, `afl`, `centipede`, `fuzz-introspector`（最初の3つで1753行ずつ）。  
  • mruby: `mruby`, `mruby_seeds`, `libprotobuf-mutator`, `LPM/external.protobufexternal.protobuf`, `aflplusplus`, `afl`, `libfuzzer`, `centipede`, `fuzz-introspector`。  
  • php: `php-src`, `aflplusplus`, `php-src/oniguruma`, `afl`, `libfuzzer`, `php-src/sapi/fuzzer`, `fuzz-introspector`, `centipede`。  
  これらを canonical repo のみ（ghostpdl/mruby/php-src）に絞り込む処理が不可欠。
- 上流スクリプト `prepare_patch_coverage_inputs.py` は `generate_revisions` と `append_commit_dates` をラップして CSV を作成する。ここへ `--filter-to-main-repo` のようなオプションを追加する場合、プロジェクト名と `repo_name` の組み合わせが揃っているためフィルタは可能。
- テスト環境には Python3 があり（pandas 非インストール）、`pytest` 等を別途追加する場合は依存を意識する。乾式検証には小型 CSV（該当3プロジェクトの一部抜粋）を使うのが現実的。

## 実装ステップ案
1. **メタデータ読み込み関数の内製**
   - 共有ヘルパーは `prepare_patch_coverage_inputs.py` に集約し、`normalize_repo_url(url)` や `load_canonical_repo_map(csv_path)` を提供する。`run_culculate_patch_coverage_pipeline.py` からこれらをインポートし、`c_cpp_vulnerability_summary.csv` から `Dict[str, RepoInfo]`（project -> {main_repo_url, repo_dir_name}）を取得できるようにする。
   - `repo_dir_name` は URL の末尾（`git://git.ghostscript.com/ghostpdl.git` → `ghostpdl`）をベースにしつつ、ローカルクローン名が異なる場合へ対応できるよう JSON/CSV 形式の簡易 override 設定を読み込むヘルパー関数も同ファイル内に追加する。
   - URL 正規化処理は `oss_fuzz_project_info.py` の `normalize_repo` と同等のロジックを転記して再利用し、外部モジュールは増やさない。

2. **パイプライン CLI の拡張**
   - `run_culculate_patch_coverage_pipeline.py` に以下の CLI オプションを追加する。  
     `--main-repo-map PATH`: 既定で `datasets/derived_artifacts/oss_fuzz_metadata/c_cpp_vulnerability_summary.csv` を指し、`none` を指定すると無効化できる。
     `--repo-name-overrides PATH`: 任意。ローカルディレクトリ名を上書きする JSON/CSV を渡せるようにする。
   - 起動時にメタデータを読み込み、`project -> canonical_repo_dir` の辞書を `process_project` に渡す。

3. **CSV フィルタリングとディレクトリ解決の修正**
   - 下流 (`run_culculate_patch_coverage_pipeline.py`) の `process_project` 冒頭で canonical repo が設定されている場合、`df = df[df['repo_name'] == canonical_repo_name]` で不要な依存リポジトリ行を除外し、行数が 2 未満になった場合は「差分を計算できない」旨の警告を出して次のプロジェクトへ進む（差分対象は最低2行必要）。
   - `repo_local_path` の決定ロジックを更新し、`repo_name` が canonical と一致しない場合でも canonical repo のディレクトリへフォールバックできるようにする（`repos_dir / canonical_repo_dir_name` を優先）。複数の canonical repo 情報を持つプロジェクトが将来的に出た場合は、優先順位（例: メタデータの並び順・CLI で指定した名前）を決めてログに明示する。
   - もしローカルに該当ディレクトリが存在しない場合は、現在の挙動と同様に警告を出しつつ日付をスキップする。上流（`prepare_patch_coverage_inputs.py`）側でも同じ canonical map を読み込めるようにし、`--filter-to-main-repo` のようなオプションで “差分対象リポジトリのみを含む CSV” を生成できるよう機能追加する（既存の CSV を必要とするワークフローには影響しないようデフォルトは従来通り）。フィルタを適用した結果データが 2 行未満になった場合は CSV 出力時に警告を出す。

4. **追加の運用補助**
   - 追加スクリプトを増やさず、既存の `run_shell_for_patch_projects.sh` から呼び出せるように、`run_culculate_patch_coverage_pipeline.py` に `--dry-run-missing-repos`（仮）オプションを実装し、メタデータが示す canonical repo ディレクトリが存在しない場合に `git clone` コマンド例をログ出力する。上流で `--filter-to-main-repo` を使って生成した CSV に対してもこのチェックを共有する。
   - README もしくは同ディレクトリの `USAGE.md` に、メタデータオプションの使い方と期待する入力ファイル構造を追記する（既存ドキュメントに追記するのみで新規ファイルを作らない）。

5. **テストと検証**
   - ユニットテスト: repo-map ローダーの URL 正規化と override 適用を `pytest` で検証する。
   - ドライラン: 問題の 3 プロジェクトで `--workers 1 --project <name>` を指定して実行し、`patch_coverage_metrics/<project>/<project>_patch_coverage.csv` が生成されること・不要リポジトリが処理されないことを確認する。
   - 既存プロジェクトへの影響確認として、`--main-repo-map none` を指定した場合の挙動が従来と同じであることを一度確認しておく。

## 期待される効果
- 依存リポジトリがクローンされていなくても、メタデータが示す本体リポジトリのみを対象に差分を取得できるため、これまで空だったプロジェクトのパッチカバレッジ CSV を出力できる。
- `c_cpp_vulnerability_summary.csv` を再利用することで、プロジェクトごとの canonical repo 情報を一元管理でき、今後 OSS-Fuzz のプロジェクトが増えた場合でも同じ仕組みで対応可能になる。

## 追加タスク: repo_name を canonical repo に揃える
上流で `repo_name` を canonical repo 名に統一すると、後段のフィルタやフォールバックがさらに簡潔になる。以下のステップで進める。

1. **設計**
   - `create_project_csvs_from_srcmap.py` で `url` を `normalize_repo_url` 相当のロジックで正規化し、`c_cpp_vulnerability_summary.csv` から読み込んだ canonical map と照合。マッチした場合は **repo_name をプロジェクト名 (`project` 列) に置き換える**（override を指定した場合は override 後のローカルディレクトリ名）。これにより `repo_name = project` となり、クローン済みディレクトリ（`clone_and_run_projects.sh` が作成する `${clone_dir}/${project}`）と一致する。
   - canonical map に存在しない依存リポジトリ（afl など）は従来の `/src/<dir>` 名を維持する。ただし後段のフィルタで自動的に除外されるよう、`is_canonical` フラグや補助情報を付ける案も検討。

2. **実装**
   - `create_project_csvs_from_srcmap.py` に canonical map ローディング機能と `--canonical-map/--repo-name-overrides` オプションを追加済みだが、`repo_name` の置き換えロジックを「URL が canonical `main_repo` と一致したら `project` 名を採用する」形に修正する。
   - `generate_revisions` が返す CSV に `repo_name = project` を書き込み、必要なら元の `/src/<dir>` 名を `source_repo_name` として残すオプションも検討。
   - `revision_with_date.py` 側で URL から `repo_name` を補完する fallback も canonical map を参照するよう更新し、URL との整合性を確保。

3. **検証**
   - 代表的なプロジェクト（ghostscript/mruby/php）で `revisions_<project>.csv`／`revisions_with_commit_date_<project>.csv` を再生成し、`repo_name` がプロジェクト名（ghostscript/mruby/php）になっていることを確認。
   - `clone_and_run_projects.sh` でクローンしたディレクトリ（project 名ベース）と一致するため、`--filter-to-main-repo` を外しても動作するか確認し、必要に応じてフィルタ処理を「保険」として残す。
   - ドキュメントを更新し、「repo_name は canonical repo にマッピングされる」旨を明記する。
