BASE=vuljit/datasets/raw/coverage_report

while read -r P; do
  [ -z "$P" ] && continue
  D="$BASE/$P"
  if [ ! -d "$D" ]; then
    echo "$P MISSING_PROJECT_DIR"
    continue
  fi

  find "$D" -maxdepth 1 -type d -name '20??????' -printf '%f\n' | sort > /tmp/cov_dates.txt

  if [ ! -s /tmp/cov_dates.txt ]; then
    echo "$P NO_DATES"
    continue
  fi

  miss=0
  while read -r d; do
    if [ ! -f "$D/$d/linux/summary.json" ]; then
      echo "$P $d MISSING_SUMMARY"
      miss=$((miss+1))
    fi
  done < /tmp/cov_dates.txt

  echo "$P missing_summary_count=$miss"
done < projects15.txt
