import argparse
import csv
import json
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Dict, Optional, Tuple

from revision_with_date import append_commit_dates, parse_args as parse_revision_args

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_CANONICAL_MAP_PATH = REPO_ROOT / "datasets" / "derived_artifacts" / "oss_fuzz_metadata" / "c_cpp_vulnerability_summary.csv"
DEFAULT_PREFIX = "revisions"

_CREATE_MODULE = None


def _get_create_module():
    global _CREATE_MODULE
    if _CREATE_MODULE is None:
        try:
            _CREATE_MODULE = import_module("create_project_csvs_from_srcmap")
        except ModuleNotFoundError:
            _CREATE_MODULE = import_module(
                "vuljit.scripts.metric_extraction.patch_coverage_pipeline.create_project_csvs_from_srcmap"
            )
    return _CREATE_MODULE


def normalize_repo_url(url: Optional[str]) -> str:
    """Normalize repository URLs for consistent comparison."""
    if not url:
        return ""
    norm = url.strip()
    if not norm:
        return ""
    norm = norm.rstrip("/")
    if norm.lower().endswith(".git"):
        norm = norm[:-4]
    return norm


def derive_repo_dir_name(url: Optional[str]) -> str:
    """Extract the repository directory name from a canonical URL."""
    normalized = normalize_repo_url(url)
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def load_repo_name_overrides(path: Optional[str]) -> Dict[str, str]:
    """Load repo directory overrides mapping canonical repo name -> local directory."""
    if not path:
        return {}
    override_path = Path(path)
    if not override_path.is_file():
        print(f"⚠️  repo overrideファイルが見つかりません: {override_path}")
        return {}

    try:
        if override_path.suffix.lower() == ".json":
            data = json.loads(override_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).strip(): str(v).strip() for k, v in data.items()}
            print(f"⚠️  repo override JSON の形式が不正です: {override_path}")
            return {}

        overrides: Dict[str, str] = {}
        with override_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                key = row[0].strip()
                value = row[1].strip()
                if key and value:
                    overrides[key] = value
        return overrides
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"⚠️  repo override の読み込みに失敗しました ({override_path}): {exc}")
        return {}


def load_canonical_repo_map(
    csv_path: Optional[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, str]]:
    """Load canonical repo information keyed by project name."""
    if not csv_path:
        return {}
    metadata_path = Path(csv_path)
    if not metadata_path.is_file():
        print(f"⚠️  canonical repo CSV が見つかりません: {metadata_path}")
        return {}

    override_map = overrides or {}
    repo_map: Dict[str, Dict[str, str]] = {}
    try:
        with metadata_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                project = (row.get("project") or "").strip()
                main_repo = (row.get("main_repo") or "").strip()
                if not project or not main_repo:
                    continue
                repo_dir_name = derive_repo_dir_name(main_repo)
                if not repo_dir_name:
                    continue
                canonical_name = project
                repo_dir_name = override_map.get(project, repo_dir_name)
                repo_map[project] = {
                    "project": project,
                    "main_repo": main_repo,
                    "normalized_repo": normalize_repo_url(main_repo),
                    "repo_name": canonical_name,
                    "repo_dir_name": repo_dir_name,
                }
    except Exception as exc:  # pragma: no cover - logging only
        print(f"⚠️  canonical repo CSV の読み込みに失敗しました: {exc}")
        return {}

    return repo_map


def filter_commit_csvs_to_canonical(
    commit_out: str | Path,
    canonical_map: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, int]]:
    """Rewrite revisions_with_commit_date CSV files to keep only canonical repo rows."""
    out_dir = Path(commit_out)
    summary: Dict[str, Dict[str, int]] = {}
    if not canonical_map:
        return summary

    for project, entry in canonical_map.items():
        csv_path = out_dir / f"revisions_with_commit_date_{project}.csv"
        if not csv_path.is_file():
            continue

        kept_repo = entry["repo_name"]
        try:
            with csv_path.open("r", newline="", encoding="utf-8-sig") as src:
                reader = csv.DictReader(src)
                fieldnames = reader.fieldnames
                if fieldnames is None:
                    print(f"⚠️  '{csv_path}' のヘッダーを解釈できません。スキップします。")
                    continue
                rows = list(reader)
        except Exception as exc:
            print(f"⚠️  '{csv_path}' の読み込みに失敗しました: {exc}")
            continue

        total = len(rows)
        filtered_rows = [row for row in rows if (row.get("repo_name") or "").strip() == kept_repo]
        if not filtered_rows:
            print(f"⚠️  '{csv_path}' に canonical repo '{kept_repo}' の行が存在しませんでした。")
            continue

        try:
            with csv_path.open("w", newline="", encoding="utf-8-sig") as dst:
                writer = csv.DictWriter(dst, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(filtered_rows)
        except Exception as exc:
            print(f"⚠️  '{csv_path}' の書き込みに失敗しました: {exc}")
            continue

        summary[project] = {"original_rows": total, "kept_rows": len(filtered_rows)}
        removed = total - len(filtered_rows)
        print(f"  - フィルタ: '{project}' で {removed} 行を除外し {len(filtered_rows)} 行を保持しました。")
        if len(filtered_rows) < 2:
            print(f"  - 注意: '{project}' の canonical repo 行が2件未満のため、差分計算ができない可能性があります。")

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="srcmap JSON 解析からコミット日時付き CSV 生成までをまとめて実行するパイプライン。"
    )
    parser.add_argument(
        "--srcmap-root",
        help="srcmap JSON (<project>/json/<date>.json) が格納されたルートディレクトリ",
    )
    parser.add_argument(
        "--csv-out",
        help="revisions_<project>.csv を出力するディレクトリ。未指定時は一時ディレクトリを利用します。",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="revisions CSV のファイル名接頭辞 (既定: revisions)",
    )
    parser.add_argument(
        "--repos",
        help="Git リポジトリ clone 済みディレクトリのルート (コミット日時取得に使用)",
    )
    parser.add_argument(
        "--commit-out",
        help="revisions_with_commit_date_<project>.csv を出力するディレクトリ",
    )
    parser.add_argument(
        "--skip-revisions",
        action="store_true",
        help="revisions_<project>.csv 生成ステージをスキップする",
    )
    parser.add_argument(
        "--skip-commit",
        action="store_true",
        help="コミット日時付与ステージをスキップする",
    )
    parser.add_argument(
        "--canonical-map",
        default=str(DEFAULT_CANONICAL_MAP_PATH),
        help="project -> main_repo を記した CSV。'none' を指定すると無効化します。",
    )
    parser.add_argument(
        "--repo-name-overrides",
        help="canonical repo 名 -> ローカルディレクトリ名のマッピング (JSON またはCSV)",
    )
    parser.add_argument(
        "--filter-to-main-repo",
        action="store_true",
        help="revisions_with_commit_date CSV を canonical repo の行のみ残すようにフィルタします。",
    )
    return parser


def resolve_defaults(args: argparse.Namespace) -> Tuple[str, Optional[str], str, str, bool]:
    """
    既存スクリプトが持つデフォルト解決ロジックを利用するため、
    元関数の引数に合わせてフォールバックを計算する。
    """
    create_defaults = _get_create_module().parse_args([])
    revision_defaults = parse_revision_args([])

    srcmap_root = args.srcmap_root or create_defaults.root
    repos = args.repos or revision_defaults.repos
    commit_out = args.commit_out or revision_defaults.out

    cleanup = False
    csv_out = args.csv_out
    if args.skip_revisions:
        csv_out = csv_out or create_defaults.out
    else:
        if csv_out is None:
            if args.skip_commit:
                csv_out = create_defaults.out
                print("注意: --skip-commit 指定のため、revisions CSV を既定ディレクトリへ保存します。")
            else:
                cleanup = True

    return srcmap_root, csv_out, repos, commit_out, cleanup


def run_pipeline(
    srcmap_root: str,
    csv_out: str,
    prefix: str,
    repos: str,
    commit_out: str,
    skip_revisions: bool = False,
    skip_commit: bool = False,
    canonical_map_path: Optional[str] = None,
    repo_name_overrides_path: Optional[str] = None,
    filter_to_main_repo: bool = False,
) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}
    overrides = load_repo_name_overrides(repo_name_overrides_path) if repo_name_overrides_path else {}
    canonical_map = {}
    if canonical_map_path:
            canonical_map = load_canonical_repo_map(canonical_map_path, overrides)

    if skip_revisions:
        print("ステージ1 (revisions CSV 生成) をスキップします。")
    else:
        stats = _get_create_module().generate_revisions(
            srcmap_root,
            csv_out,
            prefix,
            canonical_repo_map=canonical_map,
        )
        results["revisions"] = stats

    if skip_commit:
        print("ステージ2 (コミット日時付与) をスキップします。")
    else:
        stats = append_commit_dates(csv_out, repos, commit_out)
        results["commit_dates"] = stats

        if filter_to_main_repo:
            if not canonical_map:
                print("⚠️  canonical repo 情報が取得できなかったためフィルタをスキップします。")
            else:
                filtered_stats = filter_commit_csvs_to_canonical(commit_out, canonical_map)
                results["filtered"] = filtered_stats

    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        srcmap_root, csv_out, repos, commit_out, cleanup = resolve_defaults(args)

        if args.skip_revisions and args.skip_commit:
            print("警告: 両ステージがスキップ指定されています。処理は行われません。")
            return

        canonical_map_path = args.canonical_map
        if canonical_map_path and canonical_map_path.lower() == "none":
            canonical_map_path = None

        if not args.skip_revisions and cleanup:
            with tempfile.TemporaryDirectory() as temp_dir:
                run_pipeline(
                    srcmap_root=srcmap_root,
                    csv_out=temp_dir,
                    prefix=args.prefix,
                    repos=repos,
                    commit_out=commit_out,
                    skip_revisions=args.skip_revisions,
                    skip_commit=args.skip_commit,
                    canonical_map_path=canonical_map_path,
                    repo_name_overrides_path=args.repo_name_overrides,
                    filter_to_main_repo=args.filter_to_main_repo,
                )
        else:
            if csv_out is None:
                raise ValueError("csv_out が未指定です。")
            run_pipeline(
                srcmap_root=srcmap_root,
                csv_out=csv_out,
                prefix=args.prefix,
                repos=repos,
                commit_out=commit_out,
                skip_revisions=args.skip_revisions,
                skip_commit=args.skip_commit,
                canonical_map_path=canonical_map_path,
                repo_name_overrides_path=args.repo_name_overrides,
                filter_to_main_repo=args.filter_to_main_repo,
            )
    except FileNotFoundError as e:
        print(f"エラー: {e}")
    except ValueError as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()
