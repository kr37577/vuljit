#!/usr/bin/env bash
set -euo pipefail

PY=python

# ==== 入力パス（あなたの環境に合わせて要確認） ====
RQ3_PY="/work/riku-ka/vuljit/prediction/rq3_test_now.py"
VULN="/work/riku-ka/vuljit/rq3_dataset/detection_time_results.csv"
DAILY="/work/riku-ka/vuljit/outputs/results/random_forest"
BUILDS="/work/riku-ka/vuljit/rq3_dataset/project_build_counts.csv"

# ==== 主要ハイパラ ====
N=3
OUT="/work/riku-ka/vuljit/outputs/rq3_det_outputs/predicted_risk_VCCFinder_Coverage"
R1=0.5
R2=0.05
RISK_COL="predicted_risk_VCCFinder_Coverage"

# ==== 共通オプション ====
COMMON_OPTS=(
  --vuln_info_file "$VULN"
  --daily_base_dir "$DAILY"
  --build_counts_file "$BUILDS"
  --r1 "$R1" --r2 "$R2"
  --risk_column "$RISK_COL"
  --risk_norm none
  --size_transform log
  --aggregate_target_panel_all
  --aggregate_target_panel_reached
  --plot_per_project_curves
  --report_kpis
  --common_only_panels
  --make_builds_plots
  --emit_waste_csv
  --assume_daily_builds
  --cv_mode walk_forward 
  --min_test_days 0

)

# ========== 0) 実行内容の明示 ==========
echo "[RUN] main run (paper-replication):"
echo "  b0_method = monden"
echo "  baseline  = calendar"
echo "  effort_scale = 10.0"
echo "  risk_norm=rank, size_transform=log"
echo "  outputs  -> $OUT"

# ========== 1) あなたの現在設定を忠実に再現 ==========
$PY "$RQ3_PY" \
  "${COMMON_OPTS[@]}" \
  --target_found "$N" \
  --b0_method monden \
  --effort_scale 10.0 \
  --baseline_mode calendar \
  --out_dir "$OUT"

# ---- 後処理（N=3 の可読な統計・図）----
# 1) 2群比較（例：A1 vs B2）
# $PY post_pair_test.py \
#   --csv "$OUT/agg_required_N${N}_eligible.csv" \
#   --out_dir "$OUT/post_N${N}" \
#   --strategy_a "A1_Uniform" \
#   --strategy_b "B1_Risk_Proportional" \
#   --n_boot 20000 --seed 2025

# # 2) サバイバル曲線（右打ち切り：未到達）
# $PY post_survival.py \
#   --csv "$OUT/agg_required_N${N}_eligible.csv" \
#   --out_dir "$OUT/post_N${N}" \
#   --compare "A1_Uniform" "B1_Risk_Proportional"

# # 3) 校正図（基準努力における H 期待 vs 実測）
# $PY post_calibration.py \
#   --kpi_csv "$OUT/agg_kpis.csv" \
#   --out_dir "$OUT/post_N${N}" \
#   --strategy "B1_Risk_Proportional"

# ========== 2) 感度分析 ==========
#   - baseline_mode ∈ {calendar, unique_days, vuln_lag_sum}
#   - b0_method     ∈ {exposure, monden}
#   - N ∈ {1..5}（必要なら調整）
# echo "[RUN] sensitivity analysis (baseline × b0_method × N)"
# for NN in 1 2 3 4 5; do
#   for B0 in monden; do
#     for BASE in calendar unique_days vuln_lag_sum; do
#       SUB="$OUT/sens_N${NN}_${B0}_${BASE}"
#       echo "  -> N=${NN}, b0=${B0}, baseline=${BASE} -> $SUB"
#       $PY "$RQ3_PY" \
#         "${COMMON_OPTS[@]}" \
#         --target_found "$NN" \
#         --b0_method "$B0" \
#         --baseline_mode "$BASE" \
#         --effort_scale 10.0 \
#         --out_dir "$SUB"

#       # 代表比較（A1 vs B2）と要約
#       if [[ -f "$SUB/agg_required_N${NN}_eligible.csv" ]]; then
#         $PY post_pair_test.py \
#           --csv "$SUB/agg_required_N${NN}_eligible.csv" \
#           --out_dir "$SUB" \
#           --strategy_a "A1_Uniform" \
#           --strategy_b "B2_Risk_x_SizeTrans" \
#           --n_boot 10000 --seed 2025 || true
#       fi
#     done
#   done
# done

echo "DONE. outputs -> $OUT"
