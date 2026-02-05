#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: run_shell_for_patch_projects.sh -s SCRIPT [options] [-- extra-args...]

Enumerate revisions_with_commit_date_<project>.csv files and execute the given shell
script once per project. The project name is passed as the first argument to SCRIPT.

Options:
  -s FILE  Shell script to execute (required)
  -i DIR   Directory containing revisions_with_commit_date_*.csv
           (default: /work/riku-ka/vuljit/datasets/patch_coverage_inputs)
  -S CMD   Shell/interpreter to run SCRIPT with (default: bash)
  -a FILE  Shell snippet to source before executing SCRIPT (e.g. environment setup)
  -h       Show this help and exit

Extra arguments after -- are forwarded to SCRIPT after the project name.
USAGE
}
vuljit_dir=""
DEFAULT_INPUT_DIR="${vuljit_dir}/datasets/derived_artifacts/patch_coverage_inputs"

export VULJIT_CLONED_REPOS_DIR="${vuljit_dir}/datasets/raw/cloned_c_cpp_projects"

input_dir=""
shell_script=""
shell_cmd="sbatch"
activate_script=""

while getopts ":s:i:S:a:h" opt; do
  case "${opt}" in
    s) shell_script="$OPTARG" ;;
    i) input_dir="$OPTARG" ;;
    S) shell_cmd="$OPTARG" ;;
    a) activate_script="$OPTARG" ;;
    h) usage; exit 0 ;;
    :) echo "Missing argument for -${OPTARG}" >&2; usage >&2; exit 1 ;;
    \?) echo "Unknown option: -${OPTARG}" >&2; usage >&2; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

extra_args=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  extra_args=("$@")
fi

if [[ -z "$shell_script" ]]; then
  echo "Error: -s SCRIPT is required." >&2
  usage >&2
  exit 1
fi

input_dir="${input_dir:-$DEFAULT_INPUT_DIR}"

if [[ ! -d "$input_dir" ]]; then
  echo "Input directory not found: $input_dir" >&2
  exit 1
fi
if [[ ! -f "$shell_script" ]]; then
  echo "Shell script not found: $shell_script" >&2
  exit 1
fi
if [[ -n "$activate_script" && ! -f "$activate_script" ]]; then
  echo "Activation script not found: $activate_script" >&2
  exit 1
fi

if [[ -n "$activate_script" ]]; then
  # shellcheck disable=SC1090
  source "$activate_script"
fi

mapfile -t project_csvs < <(find "$input_dir" -maxdepth 1 -type f -name 'revisions_with_commit_date_*.csv' -print | sort)
if [[ ${#project_csvs[@]} -eq 0 ]]; then
  echo "No revisions_with_commit_date_*.csv files found in $input_dir" >&2
  exit 1
fi

echo "Found ${#project_csvs[@]} projects. Executing ${shell_script} for each..."

for csv_path in "${project_csvs[@]}"; do
  filename="$(basename "$csv_path")"
  project="${filename#revisions_with_commit_date_}"
  project="${project%.csv}"
  if [[ -z "$project" ]]; then
    echo "Skipping malformed filename: $filename" >&2
    continue
  fi

  echo "======================================================================"
  echo "Project: ${project}"

  if [[ -n "$shell_cmd" ]]; then
    printf 'Running: %q %q' "$shell_cmd" "$shell_script"
    printf ' %q' "$project" "${extra_args[@]}"
    printf '\n'
    "$shell_cmd" "$shell_script" "$project" "${extra_args[@]}"
  else
    printf 'Running: %q' "$shell_script"
    printf ' %q' "$project" "${extra_args[@]}"
    printf '\n'
    "$shell_script" "$project" "${extra_args[@]}"
  fi

  echo "Project ${project} completed."
done

echo "All projects processed."
