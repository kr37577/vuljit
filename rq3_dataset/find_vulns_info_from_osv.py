"""
OSS-Fuzz脆弱性データ抽出ツール (複数introduced/fixedコミット対応版)

このスクリプトは、Google OSS-Fuzz脆弱性リポジトリ(google/oss-fuzz-vulns)から
YAMLフォーマットの脆弱性情報を抽出し、分析しやすいCSV形式に変換します。

OSVフォーマット(https://ossf.github.io/osv-schema/)に準拠したYAMLファイルから
脆弱性ID、影響を受けるパッケージ、複数のコミット情報（introducedおよびfixed）、
重要度などのデータを抽出します。
"""

import os
import argparse
import yaml
import pandas as pd

# Directory containing the YAML files (cloned or vendored google/oss-fuzz-vulns)
here = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(here, '..'))
default_vuln_dir = os.environ.get('VULJIT_OSV_VULNS_DIR', os.path.join(repo_root, 'oss-fuzz-vulns', 'vulns'))
default_out_csv = os.environ.get('VULJIT_VUL_CSV', os.path.join(here, 'oss_fuzz_vulns_2025802.csv'))

parser = argparse.ArgumentParser(description='Extract OSV vulnerabilities into CSV (introduced/fixed commits supported)')
parser.add_argument('--vulns-dir', default=default_vuln_dir, help='Path to oss-fuzz-vulns/vulns directory')
parser.add_argument('--out', default=default_out_csv, help='Output CSV path')
args = parser.parse_args()
vuln_dir = args.vulns_dir

records = []  # list to collect vulnerability info dicts

for dirpath, _, filenames in os.walk(vuln_dir):
    for fname in filenames:
        if not fname.endswith((".yaml", ".yml")):
            continue  # skip non-YAML files
        file_path = os.path.join(dirpath, fname)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # parse YAML safely
                vuln = yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Log and skip file
            print(f"YAML parse error in {fname}: {e}")
            continue
        except FileNotFoundError:
            print(f"Error: Directory not found: {vuln_dir}")
            print("Please ensure you have cloned the 'google/oss-fuzz-vulns' repository")
            print(
                "and the 'vuln_dir' variable points to the correct 'vulns' subdirectory.")
            exit()  # スクリプトを終了

        if not vuln:  # skip if file is empty or couldn't be parsed into a dict
            continue

        # Top-level fields
        vuln_id = vuln.get('id')
        summary = vuln.get('summary')
        details = vuln.get('details')
        modified = vuln.get('modified')
        published = vuln.get('published')
        monorail_id = None  # 初期化

        # Report URL from references and Monorail ID
        report_url = None
        for ref in vuln.get('references', []):
            if str(ref.get('type')).upper() == "REPORT":
                report_url = ref.get('url')
                # Extract Monorail ID if URL pattern matches
                if report_url and "bugs.chromium.org/p/oss-fuzz/issues/detail?id=" in report_url:
                    try:
                        monorail_id = int(report_url.split("id=")[-1])
                    except ValueError:
                        # Keep as string if not integer
                        monorail_id = report_url.split("id=")[-1]
                break

        # Package and ecosystem (assuming first entry in 'affected' list)
        package_name = ecosystem = None
        # 複数のintroduced/fixedコミットを格納する文字列 (セミコロン区切り)
        introduced_commits_str = ""
        fixed_commits_str = ""
        versions = None
        severity = None
        repo = None

        affected_list = vuln.get('affected', [])
        if affected_list:
            # 最初の affected エントリを取得
            affected_entry = affected_list[0]
            pkg_info = affected_entry.get('package', {})
            package_name = pkg_info.get('name')
            ecosystem = pkg_info.get('ecosystem')

            # Versions (list -> join into string for CSV)
            versions_list = affected_entry.get('versions', [])
            if versions_list:
                versions = ";".join(versions_list)  # セミコロン区切り

            # Ranges (commits introduced/fixed)
            for r in affected_entry.get('ranges', []):
                if r.get('type') == 'GIT':
                    # repo は通常 range ごとに同じはずだが、異なる場合も考慮し上書き
                    repo = r.get('repo')
                    # 各range内で見つかった introduced/fixed コミットを一時的に格納するリスト
                    current_range_introduced_commits = []
                    current_range_fixed_commits = []
                    for event in r.get('events', []):
                        # introduced が見つかったらリストに追加
                        if 'introduced' in event:
                            current_range_introduced_commits.append(
                                event['introduced'])
                        # fixed が見つかったらリストに追加
                        if 'fixed' in event:
                            current_range_fixed_commits.append(event['fixed'])

                    # このrangeで見つかったintroducedコミットをセミコロン区切りで結合し追記
                    if current_range_introduced_commits:
                        if introduced_commits_str:  # 既に他のrangeのintroducedがあれば区切り文字を追加
                            introduced_commits_str += ";"
                        introduced_commits_str += ";".join(
                            current_range_introduced_commits)

                    # このrangeで見つかったfixedコミットをセミコロン区切りで結合し追記
                    if current_range_fixed_commits:
                        if fixed_commits_str:  # 既に他のrangeのfixedがあれば区切り文字を追加
                            fixed_commits_str += ";"
                        fixed_commits_str += ";".join(
                            current_range_fixed_commits)

            # Severity can be top-level or within affected
            # Check package-level severity in ecosystem_specific or severity fields
            ecospec = affected_entry.get('ecosystem_specific', {})
            # Check database_specific as well
            db_specific = affected_entry.get('database_specific', {})

            # Prioritize ecosystem_specific.severity
            if ecospec and ecospec.get('severity'):
                severity = ecospec['severity']
            # Fallback to database_specific.severity (some OSS-Fuzz entries use this)
            elif db_specific and db_specific.get('severity'):
                severity = db_specific['severity']
            # Fallback to severity directly under affected entry (less common in OSS-Fuzz)
            elif affected_entry.get('severity'):
                # OSV schema allows severity obj list here, extract score if so
                sev_list_pkg = affected_entry.get('severity')
                if isinstance(sev_list_pkg, list) and sev_list_pkg:
                    severity = sev_list_pkg[0].get('score')
                elif isinstance(sev_list_pkg, str):  # Or if it's just a string
                    severity = sev_list_pkg

        # If no package-level severity found, check top-level severity field
        if severity is None:
            # OSV top-level severity is a list of objects with 'score'
            sev_list_top = vuln.get('severity')
            if sev_list_top and isinstance(sev_list_top, list):
                # Take first severity score (could be CVSS string or a rating)
                severity = sev_list_top[0].get('score')

        # Append the collected info as a record (dict)
        records.append({
            "monorail_id": monorail_id,
            "OSV_id": vuln_id,
            "repo": repo,
            "summary": summary,
            "details": details,
            "modified": modified,
            "published": published,
            "report_url": report_url,
            "package_name": package_name,
            "ecosystem": ecosystem,
            "introduced_commits": introduced_commits_str,  # 修正：キー名を変更
            "fixed_commits": fixed_commits_str,
            "versions": versions,
            "severity": severity
        })

# Create DataFrame and output to CSV
if records:  # レコードが少なくとも1つある場合のみDataFrameを作成
    df = pd.DataFrame(records)

    # monorail_id でソート (数値としてソートするために数値に変換、変換できない場合は後方に配置)
    df['monorail_id_num'] = pd.to_numeric(df['monorail_id'], errors='coerce')
    df = df.sort_values(by='monorail_id_num', na_position='last').drop(
        columns=['monorail_id_num'])

    # CSVファイルに保存
    output_filename = args.out
    # 項目（列）の順序を指定
    columns_order = [
        "monorail_id", "OSV_id", "package_name", "ecosystem", "repo",
        "severity", "summary", "details", "introduced_commits", "fixed_commits",
        "versions", "published", "modified", "report_url"
    ]
    # DataFrameの列を再構成
    df = df[columns_order]

    df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"Saved {len(df)} vulnerability records to {output_filename}")
else:
    print(f"No vulnerability records found or processed in {vuln_dir}.")
