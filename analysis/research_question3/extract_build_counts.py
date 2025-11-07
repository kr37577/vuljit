import os
import csv
import yaml
import argparse
from pathlib import Path

def create_build_counts_csv(projects_base_dir: str, output_csv_file: str):
    """
    OSS-Fuzzのプロジェクトディレクトリをスキャンし、各プロジェクトの
    一日あたりのビルド回数をCSVファイルに書き出します。
    """
    # プロジェクトフォルダが格納されているベースディレクトリと出力CSVは引数で受け取る

    # builds_per_dayが指定されていない場合のデフォルト値
    # OSS-Fuzzのドキュメントによると、デフォルトは1日1回です
    default_builds_per_day = 1
    
    # 抽出したデータを格納するリスト
    all_projects_data = []

    print(f"'{projects_base_dir}' 内のプロジェクトをスキャンしています...")

    # ベースディレクトリが存在するか確認
    if not os.path.isdir(projects_base_dir):
        print(f"エラー: ディレクトリが見つかりません: {projects_base_dir}")
        return

    # ベースディレクトリ内の各エントリをループ処理
    for project_name in os.listdir(projects_base_dir):
        project_path = os.path.join(projects_base_dir, project_name)
        
        # ディレクトリであることだけを確認
        if os.path.isdir(project_path):
            project_yaml_path = os.path.join(project_path, 'project.yaml')
            
            builds_per_day = default_builds_per_day # デフォルト値を設定

            # project.yamlファイルが存在するか確認
            if os.path.isfile(project_yaml_path):
                try:
                    with open(project_yaml_path, 'r', encoding='utf-8') as f:
                        yaml_data = yaml.safe_load(f)
                        # builds_per_dayキーが存在すればその値を取得
                        if yaml_data and 'builds_per_day' in yaml_data:
                            builds_per_day = yaml_data['builds_per_day']
                except yaml.YAMLError as e:
                    print(f"警告: {project_name} のYAMLファイルを解析できませんでした: {e}")
                except Exception as e:
                    print(f"警告: {project_name} のファイルを読み込めませんでした: {e}")

            all_projects_data.append({
                'project': project_name,
                'builds_per_day': builds_per_day
            })

    # データをCSVファイルに書き出す
    if not all_projects_data:
        print("処理対象のプロジェクトが見つかりませんでした。")
        return

    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['project', 'builds_per_day']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(all_projects_data)
            
        print(f"\n完了しました。{len(all_projects_data)} 件のプロジェクトデータを '{output_csv_file}' に保存しました。")

    except IOError as e:
        print(f"\nエラー: CSVファイルへの書き込みに失敗しました: {e}")


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    repo_root = here.parent.parent
    derived_root = repo_root / "datasets" / "derived_artifacts"
    default_output_dir = derived_root / "oss_fuzz_build_counts"
    default_output_dir.mkdir(parents=True, exist_ok=True)

    default_projects = os.environ.get(
        'VULJIT_OSS_FUZZ_PROJECTS_DIR',
        os.path.join(str(repo_root), 'oss-fuzz', 'projects'),
    )
    default_out = os.environ.get(
        'VULJIT_BUILD_COUNTS_CSV',
        str(default_output_dir / 'project_build_counts.csv'),
    )

    parser = argparse.ArgumentParser(description='Scan oss-fuzz/projects and compute builds_per_day per project')
    parser.add_argument('--projects-dir', default=default_projects, help='Path to oss-fuzz/projects directory')
    parser.add_argument('--out', default=default_out, help='Output CSV path')
    args = parser.parse_args()

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_build_counts_csv(args.projects_dir, str(output_path))
