import os
import json
import csv
from pathlib import Path
from collections import defaultdict
import argparse
from typing import Dict, List, Optional

from prepare_patch_coverage_inputs import (
    DEFAULT_CANONICAL_MAP_PATH,
    load_canonical_repo_map,
    load_repo_name_overrides,
    normalize_repo_url,
)

# --- 設定 (ENV/引数で上書き可能) ---
DEFAULT_OUTPUT_PREFIX = 'revisions'


def generate_revisions(root: Path | str,
                       out: Path | str,
                       prefix: str = DEFAULT_OUTPUT_PREFIX,
                       canonical_repo_map: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, int]:
    """
    srcmap JSON を探索してプロジェクトごとの revisions_<project>.csv を生成する。
    出力は ['project', 'date', 'repo_name', 'url', 'revision'] の5列構成。
    生成件数をプロジェクト名→レコード数で返す。
    """
    project_data: Dict[str, List[List[str]]] = defaultdict(list)
    root_path = Path(root)
    output_path = Path(out)

    if not root_path.is_dir():
        raise FileNotFoundError(f"探索対象のディレクトリが見つかりません: {root_path}")

    output_path.mkdir(exist_ok=True)
    print(f"結果は '{output_path}' ディレクトリに保存されます。")
    print(f"'{root_path}' ディレクトリを探索しています...")

    canonical_repo_map = canonical_repo_map or {}
    for json_path in root_path.glob('*/json/*.json'):
        try:
            project_name = json_path.parent.parent.name
            date = json_path.stem
            canonical_entry = canonical_repo_map.get(project_name)
            canonical_url = ""
            canonical_repo_name = project_name
            if canonical_entry:
                canonical_url = canonical_entry.get("normalized_repo", "") or normalize_repo_url(
                    canonical_entry.get("main_repo")
                )
                canonical_repo_name = canonical_entry.get("repo_name", project_name)

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for src_path, info in data.items():
                if isinstance(info, dict) and 'rev' in info and 'url' in info:
                    repo_name = src_path.replace('/src/', '')
                    url = info['url']
                    revision = info['rev']
                    if canonical_url and normalize_repo_url(url) == canonical_url:
                        repo_name = canonical_repo_name
                    project_data[project_name].append([project_name, date, repo_name, url, revision])
                else:
                    print(f"警告: ファイル '{json_path}' のキー '{src_path}' に 'rev' または 'url' が見つかりません。")

        except json.JSONDecodeError:
            print(f"エラー: ファイル '{json_path}' は不正なJSON形式です。")
        except Exception as e:
            print(f"予期せぬエラーが発生しました ({json_path}): {e}")

    if not project_data:
        print("処理対象のデータが見つかりませんでした。")
        return {}

    print("\nCSVファイルを生成しています...")

    stats: Dict[str, int] = {}
    for project_name, records in project_data.items():
        file_name = f"{prefix}_{project_name}.csv"
        output_csv_path = output_path / file_name

        try:
            records.sort(key=lambda x: (x[1], x[2]))

            with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['project', 'date', 'repo_name', 'url', 'revision'])
                writer.writerows(records)

            stats[project_name] = len(records)
            print(f"  - '{output_csv_path}' を生成しました ({len(records)}件)。")

        except IOError as e:
            print(f"エラー: ファイル '{output_csv_path}' の書き込みに失敗しました: {e}")

    print(f"\n処理が完了しました。{len(project_data)}個のプロジェクトについてCSVファイルを生成しました。")
    return stats


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create per-project revision CSVs from srcmap JSONs (with project column).'
    )
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parent.parent.parent  # vuljit
    default_src_root = os.environ.get('VULJIT_SRCDOWN_DIR', str(repo_root / 'data' / 'srcmap_json'))
    default_out = os.environ.get('VULJIT_INTERMEDIATE_DIR', str(repo_root / 'data' / 'intermediate'))
    default_out = os.path.join(default_out, 'patch_coverage', 'csv_results')

    parser.add_argument('--root', default=default_src_root, help='Root directory containing <project>/json/<date>.json')
    parser.add_argument('--out', default=default_out, help='Output directory for per-project revisions_*.csv')
    parser.add_argument('--prefix', default=DEFAULT_OUTPUT_PREFIX, help='Output filename prefix (default: revisions)')
    parser.add_argument(
        '--canonical-map',
        default=str(DEFAULT_CANONICAL_MAP_PATH),
        help="CSV containing project -> main_repo mappings. Use 'none' to disable canonical mapping.",
    )
    parser.add_argument(
        '--repo-name-overrides',
        help='Optional JSON/CSV mapping canonical repo name -> local directory.',
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        canonical_map: Dict[str, Dict[str, str]] = {}
        overrides: Dict[str, str] = {}
        if args.canonical_map and args.canonical_map.lower() != "none":
            overrides = load_repo_name_overrides(args.repo_name_overrides)
            canonical_map = load_canonical_repo_map(args.canonical_map, overrides)
        generate_revisions(
            args.root,
            args.out,
            args.prefix,
            canonical_repo_map=canonical_map,
        )
    except FileNotFoundError as e:
        print(f"エラー: {e}")


if __name__ == '__main__':
    main()
