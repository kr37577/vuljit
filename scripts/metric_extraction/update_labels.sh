#!/usr/bin/env bash
#SBATCH --job-name=update_labels
#SBATCH --output=logs/update_labels_%A_%a.out
#SBATCH --error=errors/update_labels_%A_%a.err
#SBATCH --array=1
#SBATCH --time=04:00:00
#SBATCH --partition=cluster_short
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=13


set -euo pipefail

echo "Python環境の準備 (pyenv)..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
PYENV_ENV_NAME="py3"
PYTHON_EXEC="${PYENV_ROOT}/versions/${PYENV_ENV_NAME}/bin/python"


SCRIPT_DIR="/work/riku-ka/vuljit/scripts/metric_extraction"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LABEL_SCRIPT="${SCRIPT_DIR}/text_code_metrics/label.py"
DEFAULT_VULN_FILE="${REPO_ROOT}/datasets/derived_artifacts/vulnerability_reports/oss_fuzz_vulnerabilities.csv"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_EXEC}}"
VULN_FILE="${VULN_FILE:-${DEFAULT_VULN_FILE}}"
KEEP_BACKUP=true
RECOMPUTE_TFIDF=true
TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES:-10}"
TFIDF_STOP_WORDS="${TFIDF_STOP_WORDS:-english}"

usage() {
  cat <<'EOF'
Usage: update_labels.sh [options] PROJECT_NAME CSV_PATH [PROJECT_NAME CSV_PATH ...]

Options:
  -p, --python PATH      Python interpreter to use (default: $PYTHON_BIN or python3)
  -v, --vuln-file PATH   Vulnerability CSV to pass to label.py
      --tfidf-max-features N  Pass max_features to the TF-IDF calculator (default: ${TFIDF_MAX_FEATURES})
      --tfidf-stop-words VAL  Pass stop_words to the TF-IDF calculator (default: ${TFIDF_STOP_WORDS})
  -h, --help             Show this help

For each PROJECT_NAME/CSV_PATH pair, label.py is run against the CSV, and the
resulting *_with_vulnerability_label.csv file is moved back to CSV_PATH so that
the final artifact keeps its original name.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "${LABEL_SCRIPT}" ]] || die "label.py not found at ${LABEL_SCRIPT}"

while (($#)); do
  case "$1" in
    -p|--python)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -v|--vuln-file)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      VULN_FILE="$2"
      shift 2
      ;;
    --tfidf-max-features)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      TFIDF_MAX_FEATURES="$2"
      shift 2
      ;;
    --tfidf-stop-words)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      TFIDF_STOP_WORDS="$2"
      shift 2
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

(( $# > 0 )) || die "At least one PROJECT_NAME CSV_PATH pair is required"
(( $# % 2 == 0 )) || die "Arguments must be supplied as PROJECT_NAME CSV_PATH pairs"

timestamp() {
  date +"%Y%m%d%H%M%S"
}

process_pair() {
  local project="$1"
  local csv_path="$2"

  [[ -f "${csv_path}" ]] || die "CSV not found: ${csv_path}"

  local backup_path=""
  if [[ "${KEEP_BACKUP}" == true ]]; then
    backup_path="${csv_path}.bak.$(timestamp)"
    cp "${csv_path}" "${backup_path}"
    echo "    Backup created: ${backup_path}"
  fi

  local output_dir
  output_dir="$(cd "$(dirname "${csv_path}")" && pwd)"
  local label_output="${output_dir}/${project}_commit_metrics_with_vulnerability_label.csv"

  echo "==> Re-labeling '${project}' (${csv_path})"
  "${PYTHON_BIN}" "${LABEL_SCRIPT}" \
    "${project}" \
    "${csv_path}" \
    -o "${output_dir}" \
    --vuln_file "${VULN_FILE}"

  if [[ "${label_output}" == "${csv_path}" ]]; then
    echo "    Output already matches target file. No rename needed."
  elif [[ -f "${label_output}" ]]; then
    mv "${label_output}" "${csv_path}"
    echo "    Updated labels written back to ${csv_path}"
  else
    die "Expected output file not found after labeling: ${label_output}"
  fi

  if [[ "${RECOMPUTE_TFIDF}" == true ]]; then
    recompute_tfidf "${csv_path}"
  fi
}

recompute_tfidf() {
  local csv_path="$1"
  local tfidf_tmp="${csv_path}.tfidf.tmp"
  local pythonpath="${SCRIPT_DIR}"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="${SCRIPT_DIR}:${PYTHONPATH}"
  fi

  echo "    Recomputing TF-IDF features for ${csv_path}"
  PYTHONPATH="${pythonpath}" \
  CSV_INPUT="${csv_path}" \
  CSV_OUTPUT="${tfidf_tmp}" \
  TFIDF_MAX_FEATURES="${TFIDF_MAX_FEATURES}" \
  TFIDF_STOP_WORDS="${TFIDF_STOP_WORDS}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd
from text_code_metrics.vccfinder_commit_message_metrics import add_commit_tfidf

csv_input = os.environ["CSV_INPUT"]
csv_output = os.environ["CSV_OUTPUT"]
max_features = int(os.environ["TFIDF_MAX_FEATURES"])
stop_words_raw = os.environ["TFIDF_STOP_WORDS"]
stop_words = None if stop_words_raw.lower() in ("", "none", "null") else stop_words_raw

df = pd.read_csv(csv_input)
base_df = df.drop(columns=[c for c in df.columns if c.startswith("VCC_w")], errors="ignore")
final_df, _ = add_commit_tfidf(base_df, max_features=max_features, stop_words=stop_words)
final_df.to_csv(csv_output, index=False)
PY

  mv "${tfidf_tmp}" "${csv_path}"
  echo "    TF-IDF columns refreshed."
}

while (($#)); do
  project="$1"
  csv="$2"
  shift 2
  process_pair "${project}" "${csv}"
done
