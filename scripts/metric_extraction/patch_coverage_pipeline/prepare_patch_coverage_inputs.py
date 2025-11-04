import argparse
import tempfile
from typing import Dict, Optional, Tuple

from create_project_csvs_from_srcmap import (
    generate_revisions,
    DEFAULT_OUTPUT_PREFIX as DEFAULT_PREFIX,
    parse_args as parse_create_args,
)
from revision_with_date import append_commit_dates, parse_args as parse_revision_args


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
    return parser


def resolve_defaults(args: argparse.Namespace) -> Tuple[str, Optional[str], str, str, bool]:
    """
    既存スクリプトが持つデフォルト解決ロジックを利用するため、
    元関数の引数に合わせてフォールバックを計算する。
    """
    create_defaults = parse_create_args([])
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
) -> Dict[str, Dict[str, int]]:
    results: Dict[str, Dict[str, int]] = {}

    if skip_revisions:
        print("ステージ1 (revisions CSV 生成) をスキップします。")
    else:
        stats = generate_revisions(srcmap_root, csv_out, prefix)
        results["revisions"] = stats

    if skip_commit:
        print("ステージ2 (コミット日時付与) をスキップします。")
    else:
        stats = append_commit_dates(csv_out, repos, commit_out)
        results["commit_dates"] = stats

    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        srcmap_root, csv_out, repos, commit_out, cleanup = resolve_defaults(args)

        if args.skip_revisions and args.skip_commit:
            print("警告: 両ステージがスキップ指定されています。処理は行われません。")
            return

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
            )
    except FileNotFoundError as e:
        print(f"エラー: {e}")
    except ValueError as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    main()
