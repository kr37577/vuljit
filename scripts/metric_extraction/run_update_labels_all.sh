#!/usr/bin/env bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BASE_DIR="${REPO_ROOT}/datasets/derived_artifacts/commit_metrics"
DEFAULT_UPDATE_SCRIPT="${SCRIPT_DIR}/update_labels.sh"

BASE_DIR="${DEFAULT_BASE_DIR}"
UPDATE_SCRIPT="${DEFAULT_UPDATE_SCRIPT}"
DRY_RUN=false
UPDATE_ARGS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: run_update_labels_all.sh [options] [-- update_labels_args...]

Runs update_labels.sh for every *_commit_metrics_with_tfidf.csv found under
the commit_metrics directory tree.

Options:
  --base-dir PATH       Root directory holding per-project commit_metrics
                        subdirectories (default: datasets/.../commit_metrics)
  --update-script PATH  Path to update_labels.sh (default: alongside this script)
  -n, --dry-run         Only print the commands that would be executed
  -h, --help            Show this help
  (update_labels.sh にはスクリプト内の UPDATE_ARGS がそのまま渡されます)
EOF
}

while (($#)); do
  case "$1" in
    --base-dir)
      [[ $# -ge 2 ]] || die "Missing value for --base-dir"
      BASE_DIR="$2"
      shift 2
      ;;
    --update-script)
      [[ $# -ge 2 ]] || die "Missing value for --update-script"
      UPDATE_SCRIPT="$2"
      shift 2
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "Unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

[[ -d "${BASE_DIR}" ]] || die "Base directory not found: ${BASE_DIR}"
[[ -x "${UPDATE_SCRIPT}" || -f "${UPDATE_SCRIPT}" ]] || die "update_labels.sh not found/executable: ${UPDATE_SCRIPT}"

mapfile -d '' CSV_FILES < <(find "${BASE_DIR}" -mindepth 2 -maxdepth 2 -type f -name '*_commit_metrics_with_tfidf.csv' -print0 | sort -z)
(( ${#CSV_FILES[@]} > 0 )) || die "No *_commit_metrics_with_tfidf.csv files found under ${BASE_DIR}"

echo "Found $(( ${#CSV_FILES[@]} )) CSV files to process under ${BASE_DIR}"

failures=()
processed=0

for csv_path in "${CSV_FILES[@]}"; do
  csv_path="${csv_path%$'\n'}"
  project="$(basename "$(dirname "${csv_path}")")"
  ((++processed))
  echo ""
  echo "[$processed/${#CSV_FILES[@]}] ==> ${project}"
  echo "     CSV: ${csv_path}"

  if [[ "${DRY_RUN}" == true ]]; then
    echo "     DRY RUN: sbatch ${UPDATE_SCRIPT} ${UPDATE_ARGS[*]} ${project} ${csv_path}"
    continue
  fi

  if sbatch "${UPDATE_SCRIPT}" "${UPDATE_ARGS[@]}" "${project}" "${csv_path}"; then
    echo "     ✔ Completed ${project}"
  else
    echo "     ✖ Failed ${project}"
    failures+=("${project}")
  fi
done

if [[ "${DRY_RUN}" == true ]]; then
  echo ""
  echo "Dry run complete. ${processed} CSV files would be processed."
  exit 0
fi

if (( ${#failures[@]} > 0 )); then
  echo ""
  echo "Completed with failures (${#failures[@]} project(s)): ${failures[*]}"
  exit 1
fi

echo ""
echo "All ${processed} projects processed successfully."
