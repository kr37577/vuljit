#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    cat <<'EOF' >&2
Usage: run_all.sh <issues_csv> [additional arguments...]

arguments after <issues_csv> are forwarded to measure_detection_time.py.
EOF
    exit 1
fi

issues_csv="$1"
shift

python3 "$SCRIPT_DIR/measure_detection_time.py" --issues-csv "$issues_csv" "$@"
python3 "$SCRIPT_DIR/extract_build_counts.py"
bash "$SCRIPT_DIR/rq3.sh"
