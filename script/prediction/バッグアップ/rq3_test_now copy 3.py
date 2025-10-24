#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ3 Deterministic (Monden strict) + Aggregates v2 + CSV dumps
- 3つの集約図オプションに加えて、再解析用の詳細CSVを自動保存
- 追加: 横軸=ビルド回数版の図、各PJのビルド軸カーブ、無駄(waste)CSV
 - 追加: 固定回数Fuzz（各陽性に一定回数）シナリオのコストと期待発見数のCSV出力（任意）

出力CSV（args.out_dir配下）:
  - agg_meta_projects.csv
      各プロジェクトのメタ（H_actual_test, baseline, b0, builds_per_day, 期間, など）
  - agg_required_N{N}_all.csv
      分母=全プロジェクト。各戦略・各PJの N件到達に要するビルド数/％（到達不可は inf）
  - agg_required_N{N}_eligible.csv
      分母=eligible（H_actual_test>=N）。同上
  - agg_required_pct_{P}.csv
      分母=H_actual_test>0。実観測のP%到達に要するビルド数/％（到達不可は inf）
  - agg_waste_N{N}.csv / _summary.csv（オプション）
      baseline比の節約率、最良戦略比の無駄率（eligible & reachedのみ）
  - agg_fixed_runs_per_flag.csv / _by_project.csv（オプション）
      フラグ陽性ごとに固定回数（例: 3回）を割り当てた場合の総ビルドコストと期待発見数

Core model (deterministic):
  b_i = b0 / S_i
  H_i(E_i) = a_i * (1 - exp(-(b0 / S_i) * E_i))
  E_i(B) = B * w_i / sum_j w_j
  H(B) = sum_i H_i(E_i(B))
  b0_hat = N / sum(E_hist_i / S_i)  (exposure-MLE, builds as effort)
"""

import os
import sys
import glob
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# 先頭付近
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DEFAULT_R1 = 0.5
DEFAULT_R2 = 0.05
TRAIN_PERIOD_RATIO = 0.5
DEFAULT_RISK_COL = "predicted_risk_VCCFinder_Coverage"

# ======== I/Oユーティリティ ========

def load_build_counts(build_counts_csv: str) -> pd.DataFrame:
    df = pd.read_csv(build_counts_csv)
    if "project" not in df.columns or "builds_per_day" not in df.columns:
        raise ValueError("build_counts_file must contain 'project' and 'builds_per_day'.")
    return df

def list_projects_from_daily_base(daily_base_dir: str) -> list:
    return [os.path.basename(d) for d in sorted(glob.glob(os.path.join(daily_base_dir, "*"))) if os.path.isdir(d)]

def resolve_daily_csv_path(daily_base_dir: str, project: str) -> str:
    return os.path.join(daily_base_dir, project, f"{project}_daily_aggregated_metrics_with_predictions.csv")

def coerce_date_only(series):
    return pd.to_datetime(series, errors="coerce").dt.date

def safe_change_size(df: pd.DataFrame) -> pd.Series:
    if "lines_added" in df.columns and "lines_deleted" in df.columns:
        return (df["lines_added"].fillna(0) + df["lines_deleted"].fillna(0)).astype(float)
    if "change_size" in df.columns:
        return df["change_size"].fillna(0).astype(float)
    raise ValueError("Need 'lines_added' and 'lines_deleted' or 'change_size'.")

# ======== リスク・サイズ重み ========

def normalize_risk(arr: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(arr, float)
    if mode == "rank":
        r = np.argsort(np.argsort(x)) + 1
        return r.astype(float)
    if mode == "minmax":
        mn, mx = np.nanmin(x), np.nanmax(x)
        return (x - mn) / (mx - mn + 1e-12)
    if mode == "softmax":
        z = x - np.nanmax(x)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12)
    return x

def size_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    s = np.asarray(arr, float)
    if mode == "log1p":
        return np.log1p(s)
    if mode == "log":
        return np.log(s)
    if mode == "sqrt":
        return np.sqrt(s)
    return s

# ======== baseline努力の定義 ========

def compute_baseline_effort(df_test: pd.DataFrame,
                            vtest: pd.DataFrame,
                            builds_per_day: int,
                            assume_daily_builds: bool,
                            mode: str) -> int:
    """
    baseline_effortの定義：
      - calendar: 期間日数×builds_per_day（暦日仮定）
      - unique_days: 実観測のテスト日数×builds_per_day
      - vuln_lag_sum: 報告日-導入日の遅延合計×builds_per_day
    """
    test_start = df_test["merge_date"].min()
    test_end   = df_test["merge_date"].max()
    if assume_daily_builds or mode == "calendar":
        days_span = (pd.to_datetime(test_end) - pd.to_datetime(test_start)).days + 1
        return int(days_span) * int(builds_per_day)
    if mode == "unique_days":
        return int(df_test["merge_date"].nunique()) * int(builds_per_day)
    # vuln_lag_sum
    if vtest is None or vtest.empty:
        return int(df_test["merge_date"].nunique()) * int(builds_per_day)
    days = (pd.to_datetime(vtest["reported_date_date"]) - pd.to_datetime(vtest["commit_date_date"])).dt.days
    return int(np.maximum(1, days.fillna(1).astype(int)).sum()) * int(builds_per_day)

# ======== a_i ========

def compute_a_values(df: pd.DataFrame, r1: float, r2: float) -> pd.DataFrame:
    df = df.copy()
    H_i = df.get("vcc_commit_count", df.get("is_vcc", 0)).fillna(0).astype(float)
    S_new_i = df["S_new"].fillna(0).astype(float) if "S_new" in df.columns else safe_change_size(df)
    S_reuse_i = df["S_reuse"].fillna(0).astype(float) if "S_reuse" in df.columns else pd.Series(0.0, index=df.index)
    df["a_i"] = H_i + (r1 * S_new_i / 1000.0) + (r2 * S_reuse_i / 1000.0)
    return df

# ======== b0 推定（門田） ========

def estimate_b0_monden(df_train: pd.DataFrame, builds_per_day: int,
                       vuln_info_df: pd.DataFrame, project: str) -> float:
    train_start = df_train["merge_date"].min()
    train_end   = df_train["merge_date"].max()

    vdf = vuln_info_df.copy()
    if "project" in vdf.columns:
        vdf = vdf[vdf["project"].astype(str).str.lower() == str(project).lower()]
    else:
        if "repo" in vdf.columns:
            vdf = vdf[vdf["repo"].astype(str).str.contains(project, case=False, na=False)]
        if vdf.empty and "package_name" in vuln_info_df.columns:
            vdf = vuln_info_df[vuln_info_df["package_name"].astype(str).str.lower() == str(project).lower()]
    if vdf.empty:
        return 0.01

    vdf["commit_date_date"]   = coerce_date_only(vdf.get("commit_date"))
    vdf["reported_date_date"] = coerce_date_only(vdf.get("reported_date"))
    vtrain = vdf[(vdf["commit_date_date"] >= train_start) & (vdf["commit_date_date"] <= train_end)].copy()
    if vtrain.empty:
        return 0.01

    dtrain = df_train[["merge_date", "change_size", "a_i"]].copy()
    dtrain.rename(columns={"merge_date": "date"}, inplace=True)
    merged = pd.merge(vtrain, dtrain, left_on="commit_date_date", right_on="date", how="left")

    merged["change_size"] = merged["change_size"].fillna(1.0).astype(float)
    merged["a_i"]         = merged["a_i"].fillna(1.0).astype(float)

    days = (pd.to_datetime(merged["reported_date_date"]) - pd.to_datetime(merged["commit_date_date"])).dt.days
    days = np.maximum(1, days.fillna(1).astype(int))
    t_i  = days * int(max(1, builds_per_day))

    H_tot = len(merged)
    a_tot = float(merged["a_i"].sum())
    S_tot = float(np.maximum(merged["change_size"].values, 1.0).sum())
    t_tot = float(t_i.sum())

    if t_tot <= 0 or S_tot <= 0 or a_tot <= 0 or H_tot <= 0 or H_tot >= a_tot:
        return 0.01

    try:
        b0_hat = - (S_tot / t_tot) * np.log(1.0 - (H_tot / a_tot))
        if not np.isfinite(b0_hat) or b0_hat <= 0:
            return 0.01
        return float(b0_hat)
    except Exception:
        return 0.01

def estimate_b0_from_history(df_train: pd.DataFrame, builds_per_day: int,
                             vuln_info_df: pd.DataFrame, project: str,
                             method: str = "exposure") -> float:
    if method == "monden":
        return estimate_b0_monden(df_train, builds_per_day, vuln_info_df, project)

    train_start = df_train["merge_date"].min()
    train_end   = df_train["merge_date"].max()

    vdf = vuln_info_df.copy()
    if "project" in vdf.columns:
        vdf_proj = vdf[vdf["project"].astype(str).str.lower() == str(project).lower()]
    else:
        vdf_proj = vdf.copy()
        if "repo" in vdf_proj.columns:
            vdf_proj = vdf_proj[vdf_proj["repo"].astype(str).str.contains(project, case=False, na=False)]
        if vdf_proj.empty and "package_name" in vuln_info_df.columns:
            vdf_proj = vuln_info_df[vuln_info_df["package_name"].astype(str).str.lower() == str(project).lower()]

    if vdf_proj.empty:
        return 0.01

    vdf_proj["commit_date_date"]   = coerce_date_only(vdf_proj.get("commit_date"))
    vdf_proj["reported_date_date"] = coerce_date_only(vdf_proj.get("reported_date"))
    vtrain = vdf_proj[(vdf_proj["commit_date_date"] >= train_start) & (vdf_proj["commit_date_date"] <= train_end)].copy()
    if vtrain.empty:
        return 0.01

    dtrain = df_train[["merge_date", "change_size"]].copy()
    dtrain.rename(columns={"merge_date": "date"}, inplace=True)
    merged = pd.merge(vtrain, dtrain, left_on="commit_date_date", right_on="date", how="left")
    merged["change_size"].fillna(1.0, inplace=True)
    S_i = np.maximum(merged["change_size"].astype(float).values, 1.0)
    days = (pd.to_datetime(merged["reported_date_date"]) - pd.to_datetime(merged["commit_date_date"])).dt.days
    days = np.maximum(1, days.fillna(1).astype(int))
    t_i  = days * int(max(1, builds_per_day))

    X = (t_i / S_i).sum()
    N = len(merged)
    if X <= 0:
        return 0.01
    return float(N) / float(X)

# ======== 期待発見カーブ ========

def expected_found_curve(df_test: pd.DataFrame, budgets: np.ndarray, b0: float, risk_col: str) -> dict:
    a_i = df_test["a_i"].values.astype(float)
    S_i = np.maximum(safe_change_size(df_test).values.astype(float), 1.0)
    Risk = df_test[risk_col].fillna(0).values.astype(float)
    n = len(df_test)
    base = np.full(n, 1.0 / n)

    risk_w   = normalize_risk(Risk, EXPECTED_FOUND_OPTS["risk_norm"])
    size_w   = size_transform(S_i, EXPECTED_FOUND_OPTS["size_transform"])
    weights = {
        "A1_Uniform": np.ones(n),
        "A2_Size_Proportional": S_i,
        "B1_Risk_Proportional": risk_w,
        "B2_Risk_x_SizeTrans": risk_w * size_w,
    }
    curves = {}
    for name, w in weights.items():
        w = base if (np.all(w <= 0) or np.sum(w) <= 0) else (w / np.sum(w))
        y = []
        for B in budgets:
            E_i = B * w
            lam = (b0 / S_i) * E_i
            y.append(np.sum(a_i * (1.0 - np.exp(-lam))))
        curves[name] = np.array(y, float)
    return curves

def budget_to_reach_target(curve_y: np.ndarray, budgets: np.ndarray, target: float) -> float:
    idxs = np.where(curve_y >= target)[0]
    return float(budgets[idxs[0]]) if len(idxs) > 0 else float("inf")

def ecdf_from_values(values: np.ndarray):
    vals = np.sort(values)
    y = np.arange(1, len(vals)+1) / len(vals) if len(vals) > 0 else np.array([])
    return vals, y

# ======== Fixed runs per flag (predicted/ground-truth) ========
def _derive_flag_column_from_risk(risk_col: str) -> str | None:
    prefix = "predicted_risk_"
    if isinstance(risk_col, str) and risk_col.startswith(prefix):
        return "predicted_label_" + risk_col[len(prefix):]
    return None

def compute_fixed_runs_per_flag_metrics(df_test: pd.DataFrame,
                                        runs_per_flag: int,
                                        b0: float,
                                        baseline_effort: float,
                                        risk_col: str,
                                        flag_source: str = "predicted",
                                        flag_column: str | None = None,
                                        flag_threshold: float = 0.5) -> dict:
    """Compute total cost (builds) and expected findings when allocating a fixed
    number of runs per flagged day.

    - Flag definition:
      * predicted: use `flag_column` if provided; otherwise derive from `risk_col` by
        replacing 'predicted_risk_' with 'predicted_label_'. If neither exists, fall
        back to risk >= flag_threshold.
      * ground_truth: flag days with ground-truth vuln (vcc_commit_count>0 or is_vcc==1).
    - Each flagged index i gets effort E_i = runs_per_flag; non-flagged get 0.
    - Cost (builds) = runs_per_flag * |F|; Expected findings = sum_i a_i*(1-exp(-(b0/S_i)*E_i)).
    """
    out = {
        "n_flagged": 0,
        "total_effort_builds": 0.0,
        "effort_pct_of_baseline": np.nan,
        "expected_found": 0.0,
        "H_actual_flagged": 0,
        "flag_source": flag_source,
        "flag_column": flag_column,
        "flag_threshold": float(flag_threshold),
    }
    if runs_per_flag is None or runs_per_flag <= 0 or df_test is None or df_test.empty:
        return out

    # Build flag mask
    flagged_mask = None
    if flag_source == "predicted":
        col = flag_column or _derive_flag_column_from_risk(risk_col)
        if col and col in df_test.columns:
            try:
                flagged_mask = pd.to_numeric(df_test[col], errors="coerce").fillna(-1).astype(int) == 1
            except Exception:
                flagged_mask = df_test[col].astype(bool)
        elif (risk_col in df_test.columns):
            # fallback to thresholding risk
            try:
                flagged_mask = pd.to_numeric(df_test[risk_col], errors="coerce").astype(float) >= float(flag_threshold)
            except Exception:
                flagged_mask = df_test[risk_col] >= float(flag_threshold)
        else:
            flagged_mask = pd.Series(False, index=df_test.index)
    else:  # ground_truth
        vcc = df_test.get("vcc_commit_count", df_test.get("is_vcc", 0))
        try:
            vcc = pd.to_numeric(vcc, errors="coerce").fillna(0).astype(int)
        except Exception:
            vcc = pd.Series(0, index=df_test.index)
        flagged_mask = vcc > 0

    if flagged_mask is None:
        return out

    n_flagged = int(flagged_mask.sum())
    out["n_flagged"] = n_flagged
    total_effort = float(max(0, runs_per_flag) * n_flagged)
    out["total_effort_builds"] = total_effort
    out["effort_pct_of_baseline"] = (total_effort / float(baseline_effort) * 100.0) if baseline_effort and baseline_effort > 0 else np.nan

    if n_flagged <= 0 or not (np.isfinite(b0) and b0 > 0):
        return out

    # Expected findings under SRGM with fixed E_i per flagged
    a_i = df_test["a_i"].astype(float).values
    S_i = np.maximum(safe_change_size(df_test).astype(float).values, 1.0)
    E = np.zeros(len(df_test), dtype=float)
    E[flagged_mask.values] = float(runs_per_flag)
    lam = (float(b0) / S_i) * E
    expected_found = float(np.sum(a_i * (1.0 - np.exp(-lam))))
    out["expected_found"] = expected_found

    # Actual vulns contained in flagged set (for reference)
    vcc = df_test.get("vcc_commit_count", df_test.get("is_vcc", 0))
    try:
        vcc = pd.to_numeric(vcc, errors="coerce").fillna(0).astype(int)
    except Exception:
        vcc = pd.Series(0, index=df_test.index)
    out["H_actual_flagged"] = int(vcc[flagged_mask].sum())
    return out

# ======== Project-level aggregation helpers ========
def _aggregate_project_level(rows_df: pd.DataFrame | list[dict], value_col: str) -> pd.DataFrame:
    """Aggregate fold-level rows to project-level by mean over folds (per strategy).

    Returns a wide DataFrame indexed by project with one column per strategy containing
    the mean of finite values in `value_col` across folds for that project/strategy.
    Rows with no finite values for a strategy become NaN for that strategy.
    """
    if rows_df is None:
        return pd.DataFrame()
    df = rows_df if isinstance(rows_df, pd.DataFrame) else pd.DataFrame(rows_df)
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    df = df[np.isfinite(df[value_col])].copy()
    if df.empty:
        return pd.DataFrame()
    gb = (df.groupby(["project", "strategy"], dropna=False)[value_col]
            .mean()
            .unstack("strategy"))
    return gb

# ======== 分布統計CSV ========

def save_distribution_stats(required_dict: dict, out_csv_path: str):
    rows, pooled = [], []
    for strat, vals in required_dict.items():
        arr = np.asarray(vals, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        pooled.append(arr)
        rows.append({
            "strategy": strat,
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size>1 else 0.0,
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        })
    if rows:
        os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv_path, index=False)
        if pooled:
            all_arr = np.concatenate(pooled)
            pd.DataFrame([{
                "strategy": "ALL_POOLED",
                "n": int(all_arr.size),
                "mean": float(np.mean(all_arr)),
                "std": float(np.std(all_arr, ddof=1)) if all_arr.size>1 else 0.0,
                "median": float(np.median(all_arr)),
                "p25": float(np.percentile(all_arr, 25)),
                "p75": float(np.percentile(all_arr, 75)),
                "p90": float(np.percentile(all_arr, 90)),
                "min": float(np.min(all_arr)),
                "max": float(np.max(all_arr)),
            }]).to_csv(out_csv_path.replace(".csv","_pooled.csv"), index=False)

# ======== 図描画（PJ別） ========
def plot_project_curves(project: str,
                        budgets: np.ndarray,
                        baseline_effort: float,
                        curves_abs: dict,
                        out_dir: str,
                        target_found: int | None = None,
                        target_pct: float | None = None,
                        h_actual_test: int | None = None):
    """各プロジェクトの SRGM 期待発見曲線を % と builds の両方で保存"""
    if baseline_effort <= 0:
        return
    os.makedirs(os.path.join(out_dir, "per_project"), exist_ok=True)

    # %軸
    x_pct = (budgets / baseline_effort) * 100.0
    fig = plt.figure(figsize=(8.5,5.5)); ax = fig.add_subplot(1,1,1)
    for name, y in curves_abs.items():
        ax.plot(x_pct, y, label=name, linewidth=1.7)
    ax.set_xlabel("Effort (% of Actual)"); ax.set_ylabel("Expected cumulative findings H(B)")
    ax.set_title(f"{project}: Expected discoveries vs. effort")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, linestyle="--", alpha=0.6); ax.legend(fontsize=8)
    if target_found is not None and target_found > 0:
        ax.axhline(target_found, linestyle=":", linewidth=1.2)
    if (target_pct is not None) and (h_actual_test is not None) and (h_actual_test > 0):
        y_target = (target_pct/100.0)*h_actual_test
        ax.axhline(y_target, linestyle="--", linewidth=1.2)
    fig.savefig(os.path.join(out_dir, "per_project", f"{project}_curves.png"),
                dpi=300, bbox_inches="tight"); plt.close(fig)

    # builds軸
    fig2 = plt.figure(figsize=(8.5,5.5)); ax2 = fig2.add_subplot(1,1,1)
    for name, y in curves_abs.items():
        ax2.plot(budgets, y, label=name, linewidth=1.7)
    ax2.set_xlabel("Effort (builds)"); ax2.set_ylabel("Expected cumulative findings H(B)")
    ax2.set_title(f"{project}: Expected discoveries vs. builds")
    ax2.grid(True, linestyle="--", alpha=0.6); ax2.legend(fontsize=8)
    if target_found is not None and target_found > 0:
        ax2.axhline(target_found, linestyle=":", linewidth=1.2)
    if (target_pct is not None) and (h_actual_test is not None) and (h_actual_test > 0):
        y_target = (target_pct/100.0)*h_actual_test
        ax2.axhline(y_target, linestyle="--", linewidth=1.2)
    fig2.savefig(os.path.join(out_dir, "per_project", f"{project}_curves_builds.png"),
                 dpi=300, bbox_inches="tight"); plt.close(fig2)

# ======== 図描画（集約：％） ========

def plot_panel_ecdf_all(required_dict, counts_dict, denom, title, out_png):
    fig = plt.figure(figsize=(11,7))
    gs = fig.add_gridspec(2,1, height_ratios=[3,1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0,0])
    for strat, vals in required_dict.items():
        arr = np.array(vals, float)
        if arr.size == 0:
            continue
        xs, ys = ecdf_from_values(arr)
        ax1.step(xs, ys*100.0, where='post', label=strat)
    ax1.set_title(f"{title} (n={denom} projects)")
    ax1.set_xlabel("Effort (% of Actual)")
    ax1.set_ylabel("Projects Reached (cumulative %)")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1,0])
    labels, bars = [], []
    for strat in required_dict.keys():
        reached = counts_dict.get(strat,0)
        pct = (reached/denom*100.0) if denom>0 else 0.0
        labels.append(strat); bars.append(pct)
    ax2.bar(labels, bars)
    ax2.set_ylabel("Reach Rate (%)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i,v in enumerate(bars):
        ax2.text(i, v+1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def _pad_ylim(ax, frac=0.05):
    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * frac
    ax.set_ylim(y0, y1 + pad)

def _text_box():
    return dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.0)

def _add_box_legend(ax):
    """
    ボックス/バイオリン図で用いた記号の凡例を追加。
    - 25%~75%: 箱（IQR）
    - Range within 1.5IQR: ひげ
    - Median Line: 箱内の中央値線
    - Mean: 平均（緑のダイヤ）
    - Outliers: 外れ値（ダイヤ）※showfliers=Falseなので凡例のみ表示
    """
    handles = [
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="25%–75%"),
        Line2D([0], [0], color="black", linewidth=1.2, label="Range within 1.5IQR"),
        Line2D([0], [0], color="orange", linewidth=1.2, label="Median Line"),
        Line2D([0], [0], marker='D', markersize=6, markerfacecolor='green',
               markeredgecolor='black', linestyle='None', label="Mean"),
        Line2D([0], [0], marker='o', markersize=6, markerfacecolor='none',
               markeredgecolor='black', linestyle='None', label="Outliers"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=5, frameon=True, framealpha=0.9, fontsize=8)

def plot_violin_box_reached_percent(required_dict, out_png, title):
    labels = [k for k,v in required_dict.items() if len(v)>0]
    data = [np.array(required_dict[k], float) for k in labels]
    out_csv = os.path.join(os.path.dirname(out_png),
                           os.path.basename(out_png).replace(".png","_stats.csv"))
    save_distribution_stats(required_dict, out_csv)

    if not labels:
        fig,ax = plt.subplots(figsize=(8,4))
        ax.text(0.5,0.5,"No eligible projects (H_actual_test ≥ N)",ha='center',va='center')
        ax.axis('off'); fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig); return

    fig, ax = plt.subplots(figsize=(10,5))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.25); pc.set_edgecolor('black'); pc.set_linewidth(0.5)

    bp = ax.boxplot(
        data, labels=labels, showfliers=True, widths=0.28, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markersize=6, markeredgecolor='black', markerfacecolor='green', zorder=5)
    )
    # 箱の色は控えめ（凡例はライトグレーで示す）
    for patch in bp['boxes']:
        patch.set_alpha(0.6)
        patch.set_facecolor('lightgray')
        patch.set_edgecolor('black')
    for line in bp['whiskers'] + bp['caps'] + bp['medians']:
        line.set_linewidth(1.2)

    # # ── 統一位置での注記（全戦略で同じ高さに配置） ──
    # y_min = min(np.min(arr) for arr in data)
    # y_max = max(np.max(arr) for arr in data)
    # span  = max(1e-9, y_max - y_min)
    # label_y_med  = y_min + span * 0.06   # 全戦略共通の中央値ラベル高さ
    # label_y_mean = y_min + span * 0.84   # 全戦略共通の平均ラベル高さ

    # medians = [np.median(arr) for arr in data]
    # means   = [np.mean(arr)   for arr in data]
    # for i, (m, mu) in enumerate(zip(medians, means), start=1):
    #     ax.text(i, label_y_med,  f"med {m:.1f}%", ha="center", va="center", bbox=_text_box(), zorder=6)
    #     ax.text(i, label_y_mean, f"mean {mu:.0f}%", ha="center", va="center", bbox=_text_box(), zorder=6)

    ax.set_title(title)
    ax.set_ylabel("Effort (% of Actual)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    _add_box_legend(ax)
    _pad_ylim(ax, 0.08)
    fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig)

# ======== 図描画（集約：builds） ========

def plot_panel_ecdf_all_builds(required_builds_dict, counts_dict, denom, title, out_png):
    fig = plt.figure(figsize=(11,7))
    gs = fig.add_gridspec(2,1, height_ratios=[3,1], hspace=0.35)
    ax1 = fig.add_subplot(gs[0,0])
    for strat, vals in required_builds_dict.items():
        arr = np.array(vals, float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        xs, ys = ecdf_from_values(arr)
        ax1.step(xs, ys*100.0, where='post', label=strat)
    ax1.set_title(f"{title} (n={denom} projects)")
    ax1.set_xlabel("Effort (builds)")
    ax1.set_ylabel("Projects Reached (cumulative %)")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1,0])
    labels, bars = [], []
    for strat in required_builds_dict.keys():
        reached = counts_dict.get(strat,0)
        pct = (reached/denom*100.0) if denom>0 else 0.0
        labels.append(strat); bars.append(pct)
    ax2.bar(labels, bars)
    ax2.set_ylabel("Reach Rate (%)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    for i,v in enumerate(bars):
        ax2.text(i, v+1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    fig.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close(fig)

def plot_violin_box_reached_builds(required_dict, out_png, title):
    labels = [k for k,v in required_dict.items() if len(v)>0]
    data = [np.array(required_dict[k], float) for k in labels]
    out_csv = os.path.join(os.path.dirname(out_png),
                           os.path.basename(out_png).replace(".png","_stats.csv"))
    save_distribution_stats(required_dict, out_csv)

    if not labels:
        fig,ax = plt.subplots(figsize=(8,4))
        ax.text(0.5,0.5,"No eligible projects (H_actual_test ≥ N)",ha='center',va='center')
        ax.axis('off'); fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig); return

    fig, ax = plt.subplots(figsize=(10,5))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.25); pc.set_edgecolor('black'); pc.set_linewidth(0.5)

    bp = ax.boxplot(
        data, labels=labels, showfliers=True, widths=0.28, patch_artist=True, showmeans=True,
        meanprops=dict(marker='D', markersize=6, markeredgecolor='black', markerfacecolor='green', zorder=5)
    )
    for patch in bp['boxes']:
        patch.set_alpha(0.6)
        patch.set_facecolor('lightgray')
        patch.set_edgecolor('black')
    for line in bp['whiskers'] + bp['caps'] + bp['medians']:
        line.set_linewidth(1.2)

    # # ── 統一位置での注記 ──
    # y_min = min(np.min(arr) for arr in data)
    # y_max = max(np.max(arr) for arr in data)
    # span  = max(1e-9, y_max - y_min)
    # label_y_med  = y_min + span * 0.06
    # label_y_mean = y_min + span * 0.84

    # meds  = [np.median(arr) for arr in data]
    # means = [np.mean(arr)   for arr in data]
    # for i, (m, mu) in enumerate(zip(meds, means), start=1):
    #     ax.text(i, label_y_med,  f"med {m:.0f}", ha="center", va="center", bbox=_text_box(), zorder=6)
    #     ax.text(i, label_y_mean, f"mean {mu:.0f}", ha="center", va="center", bbox=_text_box(), zorder=6)

    ax.set_title(title)
    ax.set_ylabel("Effort (builds)")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    _add_box_legend(ax)
    _pad_ylim(ax, 0.08)
    fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig)

# ======== main ========

def main():
    ap = argparse.ArgumentParser(description="RQ3 Deterministic (Monden strict) + Aggregates v2 + CSV dumps")
    ap.add_argument("--daily_base_dir", required=True)
    ap.add_argument("--build_counts_file", required=True)
    ap.add_argument("--vuln_info_file", required=True)
    ap.add_argument("--risk_column", default=DEFAULT_RISK_COL)
    ap.add_argument("--r1", type=float, default=DEFAULT_R1)
    ap.add_argument("--r2", type=float, default=DEFAULT_R2)
    ap.add_argument("--out_dir", default="./rq3_det_outputs")
    ap.add_argument("--projects", nargs="*", default=None)
    ap.add_argument("--min_test_days", type=int, default=0)
    ap.add_argument("--effort_scale", type=float, default=2.0)
    ap.add_argument("--effort_steps", type=int, default=200)
    ap.add_argument("--baseline_mode", default="vuln_lag_sum",
                    choices=["calendar","unique_days","vuln_lag_sum"],
                    help="baseline_effortの定義（Monden枠組みに整合）")
    # CV/evaluation mode
    ap.add_argument("--cv_mode", default="walk_forward", choices=["simple","walk_forward"],
                    help="simple=従来の50/50。walk_forward=時系列ウォークフォワード（OOF列と整合）")
    ap.add_argument("--n_splits", type=int, default=10, help="walk_forward時の分割数（ML側と一致させる）")
    ap.add_argument("--train_recent", action="store_true",
                    help="walk_forward時に直前チャンクのみでb0推定（デフォルトは累積）")
    # target N
    ap.add_argument("--target_found", type=int, default=None, help="Fixed N for target panels (e.g., 3)")
    ap.add_argument("--aggregate_target_panel_all", action="store_true")
    ap.add_argument("--aggregate_target_panel_reached", action="store_true")
    # percentage target
    ap.add_argument("--target_pct", type=float, default=None, help="Target percentage (0-100) of actual discoveries")
    ap.add_argument("--aggregate_pct_panel", action="store_true")
    # 暦日ビルド仮定
    ap.add_argument("--assume_daily_builds", action="store_true",
        help="テスト期間の最初の日〜最後の日の暦日すべてで builds_per_day 回のビルドがあったと仮定")
    # per-project curves
    ap.add_argument("--plot_per_project_curves", action="store_true",
                    help="各プロジェクトの期待発見カーブ（% と builds）を出力")
    ap.add_argument("--plot_per_project_curves_per_fold", action="store_true",
                    help="walk_forward時にFoldごとのカーブも個別出力")
    # b0 推定方式
    ap.add_argument("--b0_method", default="exposure",
                    choices=["exposure", "monden"],
                    help="b0推定方法: exposure(従来) / monden(門田式)")
    # リスク重みの正規化とサイズ変換
    ap.add_argument("--risk_norm", default="none",
                    choices=["none","rank","minmax","softmax"],
                    help="リスク列の正規化方法（順位/最小最大/softmax など）")
    ap.add_argument("--size_transform", default="log",
                    choices=["none","log","log1p","sqrt"],
                    help="サイズ重みへの変換（逓減表現）")
    # b0 フォールバックと下限
    ap.add_argument("--b0_fallback", default="median",
                    choices=["median","constant"], help="b0推定失敗時の代替")
    ap.add_argument("--b0_min", type=float, default=1e-4)
    # KPI出力
    ap.add_argument("--report_kpis", action="store_true",
                    help="各戦略の基準努力時のH(B)や50%到達努力%などをCSV出力")
    # 共通PJのみ
    ap.add_argument("--common_only_panels", action="store_true",
                help="eligible のうち全戦略が到達できた共通PJのみでパネルを描画")
    # 追加オプション
    ap.add_argument("--make_builds_plots", action="store_true",
                    help="横軸をビルド回数にした ECDF/バイオリン図も出力")
    ap.add_argument("--emit_waste_csv", action="store_true",
                    help="戦略ごとの無駄（baseline比・最良戦略比）をCSV出力")
    # 固定回数Fuzz（各陽性に runs 回）
    ap.add_argument("--fixed_runs_per_flag", type=int, default=None,
                    help="各フラグ陽性に割り当てる固定回数（例: 3）。指定時にコストと期待発見数CSVを出力")
    ap.add_argument("--flag_source", default="predicted", choices=["predicted","ground_truth"],
                    help="フラグの定義: predicted=予測陽性列 / ground_truth=実ラベル")
    ap.add_argument("--flag_column", default=None,
                    help="フラグ列名（未指定時は predicted_label_<suffix from risk_column> を自動推定）")
    ap.add_argument("--flag_threshold", type=float, default=0.5,
                    help="フラグ列が無い場合の予測確率に対する閾値（predicted時のフォールバック）")

    args = ap.parse_args()
    global EXPECTED_FOUND_OPTS
    EXPECTED_FOUND_OPTS = {"risk_norm": args.risk_norm, "size_transform": args.size_transform}

    os.makedirs(args.out_dir, exist_ok=True)
    build_df = load_build_counts(args.build_counts_file)
    vuln_df = pd.read_csv(args.vuln_info_file, low_memory=False)
    projects = args.projects or list_projects_from_daily_base(args.daily_base_dir)
    if not projects:
        print("No projects found.", file=sys.stderr); sys.exit(1)

    # 収集用バケツ
    required_pct_all = {k: [] for k in ["A1_Uniform","A2_Size_Proportional","B1_Risk_Proportional","B2_Risk_x_SizeTrans"]}
    required_pct_reached = {k: [] for k in ["A1_Uniform","A2_Size_Proportional","B1_Risk_Proportional","B2_Risk_x_SizeTrans"]}
    pct_target_required_pct = {k: [] for k in ["A1_Uniform","A2_Size_Proportional","B1_Risk_Proportional","B2_Risk_x_SizeTrans"]}
    counts_all = {k:0 for k in required_pct_all.keys()}
    counts_reached = {k:0 for k in required_pct_reached.keys()}
    counts_pct = {k:0 for k in pct_target_required_pct.keys()}

    # builds軸のための別バケツ
    required_builds_all = {k: [] for k in required_pct_all.keys()}
    required_builds_reached = {k: [] for k in required_pct_all.keys()}
    counts_all_builds = {k:0 for k in required_pct_all.keys()}
    counts_reached_builds = {k:0 for k in required_pct_all.keys()}

    denom_all = 0
    denom_reached = 0
    denom_pct = 0

    rows_target_all = []
    rows_target_eligible = []
    rows_pct = []
    meta_rows = []
    rows_fixed = []  # 固定回数Fuzzの行（fold単位）

    b0_pool = []  # 他PJで推定できたb0のプール（フォールバック用）
    kpi_rows = []

    def _make_chunks(n_items: int, k: int):
        if k <= 0:
            return [list(range(n_items))]
        chunk_size = int(math.ceil(n_items / float(k)))
        return [list(range(i, min(i + chunk_size, n_items))) for i in range(0, n_items, chunk_size)]

    for project in projects:
        daily_csv = resolve_daily_csv_path(args.daily_base_dir, project)
        if not os.path.exists(daily_csv):
            continue
        try:
            df = pd.read_csv(daily_csv, low_memory=False)
        except Exception:
            continue
        if args.risk_column not in df.columns:
            continue

        df["merge_date"] = coerce_date_only(df["merge_date"])
        df.sort_values("merge_date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        row = build_df[build_df["project"].astype(str).str.lower() == project.lower()]
        builds_per_day = int(row["builds_per_day"].iloc[0]) if not row.empty else 1

        # 分岐: simple(50/50) or walk_forward(チャンク単位)
        if args.cv_mode == "simple":
            split_idx = int(len(df) * TRAIN_PERIOD_RATIO)
            if split_idx == 0 or split_idx >= len(df):
                continue
            df_train = df.iloc[:split_idx].copy()
            df_test = df.iloc[split_idx:].copy()

            if df_test["merge_date"].nunique() < args.min_test_days:
                continue

            df_train["change_size"] = safe_change_size(df_train)
            df_test["change_size"]  = safe_change_size(df_test)
            df_train["vcc_commit_count"] = df_train.get("vcc_commit_count", df_train.get("is_vcc", 0)).fillna(0)
            df_test["vcc_commit_count"]  = df_test.get("vcc_commit_count", df_test.get("is_vcc", 0)).fillna(0)
            df_test[args.risk_column]    = df_test[args.risk_column].fillna(0)

            df_train = compute_a_values(df_train, args.r1, args.r2)
            df_test  = compute_a_values(df_test,  args.r1, args.r2)

            b0 = estimate_b0_from_history(df_train, builds_per_day, vuln_df, project, method=args.b0_method)
            if not np.isfinite(b0) or b0 <= 0:
                if args.b0_fallback == "median" and len(b0_pool) > 0:
                    b0 = float(np.median(b0_pool))
                else:
                    b0 = max(args.b0_min, 0.01)
            if np.isfinite(b0) and b0 > 0.0:
                b0_pool.append(b0)

            H_actual_test = int(df_test["vcc_commit_count"].sum())
            test_start = df_test["merge_date"].min()
            test_end   = df_test["merge_date"].max()

            vuln_proj = vuln_df.copy()
            if "project" in vuln_proj.columns:
                vuln_proj = vuln_proj[vuln_proj["project"].astype(str).str.lower() == project.lower()]
            else:
                if "repo" in vuln_proj.columns:
                    vuln_proj = vuln_proj[vuln_proj["repo"].astype(str).str.contains(project, case=False, na=False)]
                if vuln_proj.empty and "package_name" in vuln_df.columns:
                    vuln_proj = vuln_df[vuln_df["package_name"].astype(str).str.lower() == project.lower()]
            vuln_proj["commit_date_date"]   = coerce_date_only(vuln_proj.get("commit_date"))
            vuln_proj["reported_date_date"] = coerce_date_only(vuln_proj.get("reported_date"))
            vtest = vuln_proj[(vuln_proj["commit_date_date"] >= test_start) & (vuln_proj["commit_date_date"] <= test_end)].copy()

            baseline_effort = compute_baseline_effort(
                df_test=df_test, vtest=vtest, builds_per_day=builds_per_day,
                assume_daily_builds=args.assume_daily_builds, mode=args.baseline_mode
            )

            max_effort = max(1, baseline_effort) * args.effort_scale
            budgets = np.linspace(0, max_effort, max(2, args.effort_steps))

            curves_abs = expected_found_curve(df_test, budgets, b0, args.risk_column)

            meta_rows.append({
                "project": project,
                "fold": None,
                "H_actual_test": H_actual_test,
                "baseline_effort_builds": baseline_effort,
                "b0_estimated": b0,
                "b0_method": args.b0_method,
                "builds_per_day": builds_per_day,
                "test_start": test_start,
                "test_end": test_end,
                "effort_scale": args.effort_scale,
                "effort_steps": args.effort_steps,
                "assume_daily_builds": bool(args.assume_daily_builds),
            })

            if args.report_kpis and baseline_effort > 0:
                for strat, y_abs in curves_abs.items():
                    idx_base = np.searchsorted(budgets, baseline_effort, side="right") - 1
                    idx_base = max(0, min(idx_base, len(budgets)-1))
                    H_at_base = float(y_abs[idx_base])
                    target50 = 0.5 * max(1, H_actual_test)
                    reqB50 = budget_to_reach_target(y_abs, budgets, target50)
                    reqPct50 = (reqB50 / baseline_effort) * 100.0 if np.isfinite(reqB50) and baseline_effort > 0 else np.inf
                    kpi_rows.append({
                        "project": project, "fold": None, "strategy": strat,
                        "H_expected_at_baseline": H_at_base,
                        "required_pct_for_50pct_actual": reqPct50,
                        "H_actual_test": H_actual_test,
                        "baseline_effort_builds": float(baseline_effort),
                        "b0_estimated": float(b0),
                    })

            if args.plot_per_project_curves:
                plot_project_curves(
                    project=project,
                    budgets=budgets,
                    baseline_effort=baseline_effort,
                    curves_abs=curves_abs,
                    out_dir=args.out_dir,
                    target_found=args.target_found,
                    target_pct=args.target_pct,
                    h_actual_test=H_actual_test
                )

            # 固定回数Fuzz（各フラグ陽性に runs 回）
            if args.fixed_runs_per_flag is not None and args.fixed_runs_per_flag > 0:
                fixed = compute_fixed_runs_per_flag_metrics(
                    df_test=df_test,
                    runs_per_flag=int(args.fixed_runs_per_flag),
                    b0=b0,
                    baseline_effort=baseline_effort,
                    risk_col=args.risk_column,
                    flag_source=args.flag_source,
                    flag_column=args.flag_column,
                    flag_threshold=float(args.flag_threshold)
                )
                rows_fixed.append({
                    "project": project,
                    "fold": None,
                    "fixed_runs_per_flag": int(args.fixed_runs_per_flag),
                    "n_flagged": fixed.get("n_flagged", 0),
                    "total_effort_builds": fixed.get("total_effort_builds", 0.0),
                    "effort_pct_of_baseline": fixed.get("effort_pct_of_baseline", np.nan),
                    "expected_found": fixed.get("expected_found", 0.0),
                    "H_actual_flagged": fixed.get("H_actual_flagged", 0),
                    "H_actual_test": int(H_actual_test),
                    "baseline_effort_builds": float(baseline_effort),
                    "b0_estimated": float(b0),
                    "flag_source": fixed.get("flag_source"),
                    "flag_column": fixed.get("flag_column"),
                    "flag_threshold": float(args.flag_threshold),
                })

            if args.target_found is not None:
                denom_all += 1
                eligible = H_actual_test >= args.target_found
                if eligible:
                    denom_reached += 1

                for strat, y_abs in curves_abs.items():
                    reqB = budget_to_reach_target(y_abs, budgets, args.target_found)
                    reached = np.isfinite(reqB)
                    req_pct_val = (reqB / baseline_effort) * 100.0 if (reached and baseline_effort > 0) else np.inf

                    rows_target_all.append({
                        "project": project,
                        "fold": None,
                        "strategy": strat,
                        "target_N": int(args.target_found),
                        "required_builds": (float(reqB) if reached else np.inf),
                        "required_pct": (float(req_pct_val) if reached else np.inf),
                        "reached_N": bool(reached),
                        "eligible_N": bool(eligible),
                        "H_actual_test": int(H_actual_test),
                        "baseline_effort_builds": float(baseline_effort),
                        "b0_estimated": float(b0),
                        "builds_per_day": int(builds_per_day),
                        "test_start": test_start,
                        "test_end": test_end,
                        "assume_daily_builds": bool(args.assume_daily_builds),
                    })

                    if eligible:
                        rows_target_eligible.append({
                            "project": project,
                            "fold": None,
                            "strategy": strat,
                            "target_N": int(args.target_found),
                            "required_builds": (float(reqB) if reached else np.inf),
                            "required_pct": (float(req_pct_val) if reached else np.inf),
                            "reached_N": bool(reached),
                            "H_actual_test": int(H_actual_test),
                            "baseline_effort_builds": float(baseline_effort),
                            "b0_estimated": float(b0),
                            "builds_per_day": int(builds_per_day),
                            "test_start": test_start,
                            "test_end": test_end,
                            "assume_daily_builds": bool(args.assume_daily_builds),
                        })

                    if reached and baseline_effort > 0:
                        required_pct_all[strat].append(req_pct_val); counts_all[strat] += 1
                        required_builds_all[strat].append(reqB); counts_all_builds[strat] += 1
                        if eligible:
                            required_pct_reached[strat].append(req_pct_val); counts_reached[strat] += 1
                            required_builds_reached[strat].append(reqB); counts_reached_builds[strat] += 1

            if args.target_pct is not None and H_actual_test > 0:
                denom_pct += 1
                target_val = (args.target_pct / 100.0) * H_actual_test
                for strat, y_abs in curves_abs.items():
                    reqB = budget_to_reach_target(y_abs, budgets, target_val)
                    reached = np.isfinite(reqB)
                    req_pct_val = (reqB / baseline_effort) * 100.0 if (reached and baseline_effort > 0) else np.inf

                    rows_pct.append({
                        "project": project,
                        "fold": None,
                        "strategy": strat,
                        "target_pct": float(args.target_pct),
                        "required_builds": (float(reqB) if reached else np.inf),
                        "required_pct": (float(req_pct_val) if reached else np.inf),
                        "reached_pct": bool(reached),
                        "H_actual_test": int(H_actual_test),
                        "baseline_effort_builds": float(baseline_effort),
                        "b0_estimated": float(b0),
                        "builds_per_day": int(builds_per_day),
                        "test_start": test_start,
                        "test_end": test_end,
                        "assume_daily_builds": bool(args.assume_daily_builds),
                    })

                    if reached and baseline_effort > 0:
                        pct_target_required_pct[strat].append(req_pct_val); counts_pct[strat] += 1
        else:
            # walk_forward モード: MLのOOF分割（ceil）と一致させる
            n = len(df)
            chunks = _make_chunks(n, max(1, args.n_splits))
            if len(chunks) <= 1:
                continue

            df[args.risk_column] = df[args.risk_column].fillna(0)

            vuln_proj = vuln_df.copy()
            if "project" in vuln_proj.columns:
                vuln_proj = vuln_proj[vuln_proj["project"].astype(str).str.lower() == project.lower()]
            else:
                if "repo" in vuln_proj.columns:
                    vuln_proj = vuln_proj[vuln_proj["repo"].astype(str).str.contains(project, case=False, na=False)]
                if vuln_proj.empty and "package_name" in vuln_df.columns:
                    vuln_proj = vuln_df[vuln_df["package_name"].astype(str).str.lower() == project.lower()]
            vuln_proj["commit_date_date"]   = coerce_date_only(vuln_proj.get("commit_date"))
            vuln_proj["reported_date_date"] = coerce_date_only(vuln_proj.get("reported_date"))

            for i in range(1, len(chunks)):
                test_idx = chunks[i]
                train_idx = chunks[i-1] if args.train_recent else [idx for sub in chunks[:i] for idx in sub]

                df_train_i = df.iloc[train_idx].copy()
                df_test_i  = df.iloc[test_idx].copy()

                if df_test_i["merge_date"].nunique() < args.min_test_days:
                    continue

                df_train_i["change_size"] = safe_change_size(df_train_i)
                df_test_i["change_size"]  = safe_change_size(df_test_i)
                df_train_i["vcc_commit_count"] = df_train_i.get("vcc_commit_count", df_train_i.get("is_vcc", 0)).fillna(0)
                df_test_i["vcc_commit_count"]  = df_test_i.get("vcc_commit_count", df_test_i.get("is_vcc", 0)).fillna(0)
                df_test_i[args.risk_column]     = df_test_i[args.risk_column].fillna(0)

                df_train_i = compute_a_values(df_train_i, args.r1, args.r2)
                df_test_i  = compute_a_values(df_test_i,  args.r1, args.r2)

                b0 = estimate_b0_from_history(df_train_i, builds_per_day, vuln_df, project, method=args.b0_method)
                if not np.isfinite(b0) or b0 <= 0:
                    if args.b0_fallback == "median" and len(b0_pool) > 0:
                        b0 = float(np.median(b0_pool))
                    else:
                        b0 = max(args.b0_min, 0.01)
                if np.isfinite(b0) and b0 > 0.0:
                    b0_pool.append(b0)

                H_actual_test = int(df_test_i["vcc_commit_count"].sum())
                test_start = df_test_i["merge_date"].min()
                test_end   = df_test_i["merge_date"].max()

                vtest = vuln_proj[(vuln_proj["commit_date_date"] >= test_start) & (vuln_proj["commit_date_date"] <= test_end)].copy()

                baseline_effort = compute_baseline_effort(
                    df_test=df_test_i, vtest=vtest, builds_per_day=builds_per_day,
                    assume_daily_builds=args.assume_daily_builds, mode=args.baseline_mode
                )

                max_effort = max(1, baseline_effort) * args.effort_scale
                budgets = np.linspace(0, max_effort, max(2, args.effort_steps))

                curves_abs = expected_found_curve(df_test_i, budgets, b0, args.risk_column)

                meta_rows.append({
                    "project": project,
                    "fold": i,
                    "H_actual_test": H_actual_test,
                    "baseline_effort_builds": baseline_effort,
                    "b0_estimated": b0,
                    "b0_method": args.b0_method,
                    "builds_per_day": builds_per_day,
                    "test_start": test_start,
                    "test_end": test_end,
                    "effort_scale": args.effort_scale,
                    "effort_steps": args.effort_steps,
                    "assume_daily_builds": bool(args.assume_daily_builds),
                })

                if args.report_kpis and baseline_effort > 0:
                    for strat, y_abs in curves_abs.items():
                        idx_base = np.searchsorted(budgets, baseline_effort, side="right") - 1
                        idx_base = max(0, min(idx_base, len(budgets)-1))
                        H_at_base = float(y_abs[idx_base])
                        target50 = 0.5 * max(1, H_actual_test)
                        reqB50 = budget_to_reach_target(y_abs, budgets, target50)
                        reqPct50 = (reqB50 / baseline_effort) * 100.0 if np.isfinite(reqB50) and baseline_effort > 0 else np.inf
                        kpi_rows.append({
                            "project": project, "fold": i, "strategy": strat,
                            "H_expected_at_baseline": H_at_base,
                            "required_pct_for_50pct_actual": reqPct50,
                            "H_actual_test": H_actual_test,
                            "baseline_effort_builds": float(baseline_effort),
                            "b0_estimated": float(b0),
                        })

                if args.plot_per_project_curves and args.plot_per_project_curves_per_fold:
                    plot_project_curves(
                        project=f"{project}_fold{i}",
                        budgets=budgets,
                        baseline_effort=baseline_effort,
                        curves_abs=curves_abs,
                        out_dir=args.out_dir,
                        target_found=args.target_found,
                        target_pct=args.target_pct,
                        h_actual_test=H_actual_test
                    )

                # 固定回数Fuzz（各フラグ陽性に runs 回）
                if args.fixed_runs_per_flag is not None and args.fixed_runs_per_flag > 0:
                    fixed = compute_fixed_runs_per_flag_metrics(
                        df_test=df_test_i,
                        runs_per_flag=int(args.fixed_runs_per_flag),
                        b0=b0,
                        baseline_effort=baseline_effort,
                        risk_col=args.risk_column,
                        flag_source=args.flag_source,
                        flag_column=args.flag_column,
                        flag_threshold=float(args.flag_threshold)
                    )
                    rows_fixed.append({
                        "project": project,
                        "fold": i,
                        "fixed_runs_per_flag": int(args.fixed_runs_per_flag),
                        "n_flagged": fixed.get("n_flagged", 0),
                        "total_effort_builds": fixed.get("total_effort_builds", 0.0),
                        "effort_pct_of_baseline": fixed.get("effort_pct_of_baseline", np.nan),
                        "expected_found": fixed.get("expected_found", 0.0),
                        "H_actual_flagged": fixed.get("H_actual_flagged", 0),
                        "H_actual_test": int(H_actual_test),
                        "baseline_effort_builds": float(baseline_effort),
                        "b0_estimated": float(b0),
                        "flag_source": fixed.get("flag_source"),
                        "flag_column": fixed.get("flag_column"),
                        "flag_threshold": float(args.flag_threshold),
                    })

                if args.target_found is not None:
                    denom_all += 1
                    eligible = H_actual_test >= args.target_found
                    if eligible:
                        denom_reached += 1

                    for strat, y_abs in curves_abs.items():
                        reqB = budget_to_reach_target(y_abs, budgets, args.target_found)
                        reached = np.isfinite(reqB)
                        req_pct_val = (reqB / baseline_effort) * 100.0 if (reached and baseline_effort > 0) else np.inf

                        rows_target_all.append({
                            "project": project,
                            "fold": i,
                            "strategy": strat,
                            "target_N": int(args.target_found),
                            "required_builds": (float(reqB) if reached else np.inf),
                            "required_pct": (float(req_pct_val) if reached else np.inf),
                            "reached_N": bool(reached),
                            "eligible_N": bool(eligible),
                            "H_actual_test": int(H_actual_test),
                            "baseline_effort_builds": float(baseline_effort),
                            "b0_estimated": float(b0),
                            "builds_per_day": int(builds_per_day),
                            "test_start": test_start,
                            "test_end": test_end,
                            "assume_daily_builds": bool(args.assume_daily_builds),
                        })

                        if eligible:
                            rows_target_eligible.append({
                                "project": project,
                                "fold": i,
                                "strategy": strat,
                                "target_N": int(args.target_found),
                                "required_builds": (float(reqB) if reached else np.inf),
                                "required_pct": (float(req_pct_val) if reached else np.inf),
                                "reached_N": bool(reached),
                                "H_actual_test": int(H_actual_test),
                                "baseline_effort_builds": float(baseline_effort),
                                "b0_estimated": float(b0),
                                "builds_per_day": int(builds_per_day),
                                "test_start": test_start,
                                "test_end": test_end,
                                "assume_daily_builds": bool(args.assume_daily_builds),
                            })

                        if reached and baseline_effort > 0:
                            required_pct_all[strat].append(req_pct_val); counts_all[strat] += 1
                            required_builds_all[strat].append(reqB); counts_all_builds[strat] += 1
                            if eligible:
                                required_pct_reached[strat].append(req_pct_val); counts_reached[strat] += 1
                                required_builds_reached[strat].append(reqB); counts_reached_builds[strat] += 1

                if args.target_pct is not None and H_actual_test > 0:
                    denom_pct += 1
                    target_val = (args.target_pct / 100.0) * H_actual_test
                    for strat, y_abs in curves_abs.items():
                        reqB = budget_to_reach_target(y_abs, budgets, target_val)
                        reached = np.isfinite(reqB)
                        req_pct_val = (reqB / baseline_effort) * 100.0 if (reached and baseline_effort > 0) else np.inf

                        rows_pct.append({
                            "project": project,
                            "fold": i,
                            "strategy": strat,
                            "target_pct": float(args.target_pct),
                            "required_builds": (float(reqB) if reached else np.inf),
                            "required_pct": (float(req_pct_val) if reached else np.inf),
                            "reached_pct": bool(reached),
                            "H_actual_test": int(H_actual_test),
                            "baseline_effort_builds": float(baseline_effort),
                            "b0_estimated": float(b0),
                            "builds_per_day": int(builds_per_day),
                            "test_start": test_start,
                            "test_end": test_end,
                            "assume_daily_builds": bool(args.assume_daily_builds),
                        })

                        if reached and baseline_effort > 0:
                            pct_target_required_pct[strat].append(req_pct_val); counts_pct[strat] += 1

    # ========== 集約図の出力（％／プロジェクト単位） ==========
    if args.aggregate_target_panel_all and args.target_found is not None:
        df_all = pd.DataFrame(rows_target_all) if rows_target_all else pd.DataFrame()
        out_png = os.path.join(args.out_dir, f"agg_panel_all_N{args.target_found}.png")
        if not df_all.empty:
            proj_pct = _aggregate_project_level(df_all, "required_pct")
            required_pct_all_proj = {col: proj_pct[col].dropna().astype(float).tolist()
                                     for col in proj_pct.columns} if not proj_pct.empty else {}
            denom_all_proj = int(df_all["project"].nunique())
            counts_all_proj = {k: len(v) for k, v in required_pct_all_proj.items()}
            plot_panel_ecdf_all(required_pct_all_proj, counts_all_proj, max(denom_all_proj,1),
                f"ECDF of Required Effort % to Reach N={args.target_found} (denominator: projects)",
                out_png)
        else:
            # 空の図（プロジェクトなし）
            fig, ax = plt.subplots(figsize=(8,4))
            ax.text(0.5,0.5,"No projects",ha='center',va='center'); ax.axis('off')
            fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig)

        # builds版（全PJ分母, プロジェクト単位）
        if args.make_builds_plots:
            out_png_b = os.path.join(args.out_dir, f"agg_panel_all_N{args.target_found}_builds.png")
            if not df_all.empty:
                proj_b = _aggregate_project_level(df_all, "required_builds")
                required_builds_all_proj = {col: proj_b[col].dropna().astype(float).tolist()
                                            for col in proj_b.columns} if not proj_b.empty else {}
                counts_all_b_proj = {k: len(v) for k, v in required_builds_all_proj.items()}
                plot_panel_ecdf_all_builds(required_builds_all_proj, counts_all_b_proj, max(denom_all_proj,1),
                    f"ECDF of Effort (builds) to Reach N={args.target_found} (denominator: projects)",
                    out_png_b)
            else:
                fig, ax = plt.subplots(figsize=(8,4))
                ax.text(0.5,0.5,"No projects",ha='center',va='center'); ax.axis('off')
                fig.savefig(out_png_b, dpi=300, bbox_inches='tight'); plt.close(fig)

    if args.aggregate_target_panel_reached and args.target_found is not None:
        df_elig = pd.DataFrame(rows_target_eligible) if rows_target_eligible else pd.DataFrame()
        out_png = os.path.join(args.out_dir, f"agg_panel_reached_N{args.target_found}.png")
        if df_elig.empty:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.text(0.5,0.5,"No eligible projects (H_actual_test >= N)",ha='center',va='center'); ax.axis('off')
            fig.savefig(out_png, dpi=300, bbox_inches='tight'); plt.close(fig)
        else:
            proj_pct_e = _aggregate_project_level(df_elig, "required_pct")
            required_pct_reached_proj = {col: proj_pct_e[col].dropna().astype(float).tolist()
                                         for col in proj_pct_e.columns} if not proj_pct_e.empty else {}
            denom_reached_proj = int(df_elig["project"].nunique())
            counts_reached_proj = {k: len(v) for k, v in required_pct_reached_proj.items()}
            plot_panel_ecdf_all(required_pct_reached_proj, counts_reached_proj, max(denom_reached_proj,1),
                f"ECDF (eligible only) of Effort % to Reach N={args.target_found} (denominator: projects)",
                out_png)

            box_png = os.path.join(args.out_dir, f"agg_box_reached_N{args.target_found}.png")
            plot_violin_box_reached_percent(required_pct_reached_proj, box_png,
                f"Required Effort % (eligible only) to Reach N={args.target_found}")

            # builds版（eligible, プロジェクト単位）
            if args.make_builds_plots:
                out_png_b = os.path.join(args.out_dir, f"agg_panel_reached_N{args.target_found}_builds.png")
                proj_b_e = _aggregate_project_level(df_elig, "required_builds")
                required_builds_reached_proj = {col: proj_b_e[col].dropna().astype(float).tolist()
                                                for col in proj_b_e.columns} if not proj_b_e.empty else {}
                counts_reached_b_proj = {k: len(v) for k, v in required_builds_reached_proj.items()}
                plot_panel_ecdf_all_builds(required_builds_reached_proj, counts_reached_b_proj, max(denom_reached_proj,1),
                    f"ECDF (eligible only) of Effort (builds) to Reach N={args.target_found}",
                    out_png_b)
                box_png_b = os.path.join(args.out_dir, f"agg_box_reached_N{args.target_found}_builds.png")
                plot_violin_box_reached_builds(required_builds_reached_proj, box_png_b,
                    f"Required Effort (builds) (eligible only) to Reach N={args.target_found}")

    # ========== 共通PJサブセット（％／builds） ==========
    if (args.aggregate_target_panel_reached and args.target_found is not None
        and args.common_only_panels and rows_target_eligible):

        df_elig = pd.DataFrame(rows_target_eligible)
        # % 共通（プロジェクト単位の平均で集約）
        piv = _aggregate_project_level(df_elig, "required_pct")
        if not piv.empty:
            mask_common = np.isfinite(piv.values).all(axis=1)
            piv_common = piv[mask_common]
            required_pct_common = {col: piv_common[col].dropna().astype(float).tolist()
                                   for col in piv_common.columns}
            denom_common = len(piv_common)
            counts_common = {k: denom_common for k in required_pct_common.keys()}

            out_ecdf = os.path.join(args.out_dir, f"agg_panel_reached_N{args.target_found}_common.png")
            plot_panel_ecdf_all(required_pct_common, counts_common, max(denom_common,1),
                f"ECDF (eligible + common) of Effort % to Reach N={args.target_found} (n={denom_common} projects)",
                out_ecdf)

            out_box = os.path.join(args.out_dir, f"agg_box_reached_N{args.target_found}_common.png")
            plot_violin_box_reached_percent(required_pct_common, out_box,
                f"Required Effort % (eligible + common) to Reach N={args.target_found} (n={denom_common} projects)")

            # builds 共通（プロジェクト単位）
            if args.make_builds_plots:
                piv_b = _aggregate_project_level(df_elig, "required_builds")
                mask_common_b = np.isfinite(piv_b.values).all(axis=1)
                piv_common_b = piv_b[mask_common_b]
                required_builds_common = {col: piv_common_b[col].dropna().astype(float).tolist()
                                          for col in piv_common_b.columns}
                denom_common_b = len(piv_common_b)
                counts_common_b = {k: denom_common_b for k in required_builds_common.keys()}

                out_ecdf_b = os.path.join(args.out_dir, f"agg_panel_reached_N{args.target_found}_common_builds.png")
                plot_panel_ecdf_all_builds(required_builds_common, counts_common_b, max(denom_common_b,1),
                    f"ECDF (eligible + common) of Effort (builds) to Reach N={args.target_found} (n={denom_common_b} projects)",
                    out_ecdf_b)

                out_box_b = os.path.join(args.out_dir, f"agg_box_reached_N{args.target_found}_common_builds.png")
                plot_violin_box_reached_builds(required_builds_common, out_box_b,
                    f"Required Effort (builds) (eligible + common) to Reach N={args.target_found} (n={denom_common_b} projects)")

    # ========== P%ターゲット・ECDF(%) ==========
    if args.aggregate_pct_panel and args.target_pct is not None:
        save_distribution_stats(
            pct_target_required_pct,
            os.path.join(args.out_dir, f"agg_box_pct_{int(args.target_pct)}_stats.csv")
        )
        out_png = os.path.join(args.out_dir, f"agg_panel_pct_{int(args.target_pct)}.png")
        denom = sum(len(v) for v in pct_target_required_pct.values())
        denom = denom if denom>0 else 1
        plot_panel_ecdf_all(pct_target_required_pct, counts_pct, denom,
            f"ECDF of Effort % to Reach {args.target_pct:.0f}% of Actual (denominator: H_actual>0)",
            out_png)

    # ========== 無駄(waste) CSV（プロジェクト単位: 平均） ==========
    if args.emit_waste_csv and rows_target_eligible and args.target_found is not None:
        df_e = pd.DataFrame(rows_target_eligible).copy()
        df_e = df_e[np.isfinite(df_e["required_builds"])]

        # プロジェクト単位の平均で集約
        agg_req = (df_e.groupby(["project","strategy"], dropna=False)["required_builds"]
                      .mean().reset_index())
        agg_base = (df_e.groupby("project", dropna=False)["baseline_effort_builds"]
                      .mean().reset_index())
        df_w = agg_req.merge(agg_base, on="project", how="left")

        df_w.rename(columns={"required_builds":"required_builds_mean",
                             "baseline_effort_builds":"baseline_effort_builds_mean"}, inplace=True)
        df_w["builds_saved_vs_baseline"] = df_w["baseline_effort_builds_mean"] - df_w["required_builds_mean"]
        df_w["pct_saved_vs_baseline"] = np.where(
            df_w["baseline_effort_builds_mean"]>0,
            100.0*df_w["builds_saved_vs_baseline"]/df_w["baseline_effort_builds_mean"],
            np.nan
        )

        # best per project（集約後の最小値）
        best = (df_w.loc[df_w.groupby(["project"])['required_builds_mean'].idxmin(),
                        ["project","required_builds_mean"]]
                   .rename(columns={"required_builds_mean":"best_required_builds_mean"}))
        df_w = df_w.merge(best, on=["project"], how="left")
        df_w["waste_vs_best_builds"] = df_w["required_builds_mean"] - df_w["best_required_builds_mean"]
        df_w["pct_over_best"] = np.where(
            df_w["best_required_builds_mean"]>0,
            100.0*df_w["waste_vs_best_builds"]/df_w["best_required_builds_mean"],
            np.nan
        )

        out_waste = os.path.join(args.out_dir, f"agg_waste_N{int(args.target_found)}.csv")
        df_w.to_csv(out_waste, index=False)

        summ = (df_w.groupby("strategy")[["pct_saved_vs_baseline","pct_over_best"]]
                .mean().reset_index())
        summ.to_csv(os.path.join(args.out_dir, f"agg_waste_N{int(args.target_found)}_summary.csv"),
                    index=False)

    # ========== メタ/詳細CSV ==========
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(os.path.join(args.out_dir, "agg_meta_projects.csv"), index=False)

    if args.target_found is not None and rows_target_all:
        pd.DataFrame(rows_target_all).to_csv(
            os.path.join(args.out_dir, f"agg_required_N{args.target_found}_all.csv"), index=False)

    if args.target_found is not None and rows_target_eligible:
        pd.DataFrame(rows_target_eligible).to_csv(
            os.path.join(args.out_dir, f"agg_required_N{args.target_found}_eligible.csv"), index=False)

    if args.target_pct is not None and rows_pct:
        pd.DataFrame(rows_pct).to_csv(
            os.path.join(args.out_dir, f"agg_required_pct_{int(args.target_pct)}.csv"), index=False)

    if args.report_kpis and kpi_rows:
        pd.DataFrame(kpi_rows).to_csv(
            os.path.join(args.out_dir, "agg_kpis.csv"), index=False)

    # ========== 固定回数FuzzのCSV出力 ==========
    if rows_fixed:
        df_fixed = pd.DataFrame(rows_fixed)
        df_fixed.to_csv(os.path.join(args.out_dir, "agg_fixed_runs_per_flag.csv"), index=False)
        # プロジェクト平均（fold平均、有限値のみ）
        agg_cols = ["n_flagged","total_effort_builds","effort_pct_of_baseline","expected_found","H_actual_flagged",
                    "H_actual_test","baseline_effort_builds","b0_estimated"]
        df_fixed_by_proj = (df_fixed.groupby("project", dropna=False)[agg_cols]
                                   .mean(numeric_only=True).reset_index())
        df_fixed_by_proj.to_csv(os.path.join(args.out_dir, "agg_fixed_runs_per_flag_by_project.csv"), index=False)

if __name__ == "__main__":
    main()
