import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANEL_PATH = "data/panel_q1_q9.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

WS_RE = re.compile(r"\s+")
USE_COLS = ["date", "code", "name", "sector", "close", "volume", "mktcap"]

df = pd.read_csv(
    PANEL_PATH,
    usecols=lambda c: c in USE_COLS,
    dtype={"code": str, "name": str, "sector": str, "date": str},
)

for col in ("close", "volume", "mktcap"):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["sector_clean"] = (
    df["sector"]
    .fillna("Unknown_Sector")
    .map(lambda x: WS_RE.sub(" ", str(x)).strip() or "Unknown_Sector")
)
df["name_clean"] = df["name"].fillna("").map(lambda x: WS_RE.sub(" ", str(x)).strip())
df = df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)

total_raw_rows = int(len(df))
bad_close_mask = df["close"].isna() | (df["close"] <= 0)
bad_mktcap_mask = df["mktcap"].isna() | (df["mktcap"] <= 0)
bad_sector_mask = df["sector_clean"] == "Unknown_Sector"
bad_volume_mask = df["volume"].isna() | (df["volume"] < 0)

defect_summary_df = pd.DataFrame([
    {"issue_type": "missing_or_nonpos_close", "affected_rows": int(bad_close_mask.sum()), "share_pct": float(bad_close_mask.mean() * 100.0)},
    {"issue_type": "missing_or_nonpos_mktcap", "affected_rows": int(bad_mktcap_mask.sum()), "share_pct": float(bad_mktcap_mask.mean() * 100.0)},
    {"issue_type": "unknown_or_missing_sector", "affected_rows": int(bad_sector_mask.sum()), "share_pct": float(bad_sector_mask.mean() * 100.0)},
    {"issue_type": "missing_or_neg_volume", "affected_rows": int(bad_volume_mask.sum()), "share_pct": float(bad_volume_mask.mean() * 100.0)},
])
defect_audit_path = os.path.join(OUT_DIR, "q1_q4_defect_audit.csv")
defect_summary_df.to_csv(defect_audit_path, index=False, encoding="utf-8")


def ci95_mean(arr):
    s = pd.Series(arr, dtype="float64").dropna()
    n = int(len(s))
    if n == 0:
        return {"mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n": 0}
    if n == 1:
        v = float(s.iloc[0])
        return {"mean": v, "ci95_low": v, "ci95_high": v, "n": 1}
    mean = float(s.mean())
    margin = float(1.96 * s.std(ddof=1) / np.sqrt(n))
    return {"mean": mean, "ci95_low": mean - margin, "ci95_high": mean + margin, "n": n}


# ==============================================================================
# Q1: MARKET-CAP RANKING OVER 2.5 YEARS (TOP 30, NEWCOMERS, SECTOR MOVES)
# ==============================================================================
m_df = df[~bad_mktcap_mask].copy()
m_df["rank_day"] = m_df.groupby("date", sort=False)["mktcap"].rank(ascending=False, method="min")

dates_sorted = sorted(m_df["date"].unique())
start_date, end_date = dates_sorted[0], dates_sorted[-1]

top30_start = (
    m_df[m_df["date"] == start_date]
    .sort_values("rank_day", kind="mergesort")
    .head(30)[["code", "name_clean", "sector_clean", "rank_day", "mktcap"]]
    .rename(columns={"name_clean": "name", "sector_clean": "sector", "rank_day": "start_rank", "mktcap": "start_mktcap"})
    .reset_index(drop=True)
)

top30_end = (
    m_df[m_df["date"] == end_date]
    .sort_values("rank_day", kind="mergesort")
    .head(30)[["code", "name_clean", "sector_clean", "rank_day", "mktcap"]]
    .rename(columns={"name_clean": "name", "sector_clean": "sector", "rank_day": "end_rank", "mktcap": "end_mktcap"})
    .reset_index(drop=True)
)

start_top30_codes = set(top30_start["code"])
end_top30_codes = set(top30_end["code"])
retained_codes = sorted(start_top30_codes & end_top30_codes)
newcomer_codes = sorted(end_top30_codes - start_top30_codes)
exited_codes = sorted(start_top30_codes - end_top30_codes)

newcomers_df = top30_end[top30_end["code"].isin(newcomer_codes)].copy()
newcomer_sectors_count = newcomers_df["sector"].value_counts().to_dict()

s_map = m_df[m_df["date"] == start_date].set_index("code")[["rank_day", "sector_clean", "name_clean"]]
e_map = m_df[m_df["date"] == end_date].set_index("code")[["rank_day", "sector_clean", "name_clean"]]
common_firms = s_map.join(e_map, lsuffix="_start", rsuffix="_end", how="inner").dropna(
    subset=["rank_day_start", "rank_day_end"]
)
common_firms["abs_rank_move"] = (common_firms["rank_day_end"] - common_firms["rank_day_start"]).abs()
common_firms["signed_rank_move"] = common_firms["rank_day_start"] - common_firms["rank_day_end"]  # + means moved up

q1_sector_moves_rows = []
for sec, grp in common_firms.groupby("sector_clean_end", sort=False):
    st = ci95_mean(grp["abs_rank_move"])
    if st["n"] >= 2:
        q1_sector_moves_rows.append({
            "sector": sec,
            "n_firms": st["n"],
            "mean_abs_rank_move": st["mean"],
            "abs_ci95_low": st["ci95_low"],
            "abs_ci95_high": st["ci95_high"],
            "mean_signed_rank_gain": float(grp["signed_rank_move"].mean()),
        })

q1_sector_moves_df = (
    pd.DataFrame(q1_sector_moves_rows)
    .sort_values("mean_abs_rank_move", ascending=False, kind="mergesort")
    .reset_index(drop=True)
)

# ==============================================================================
# Q2: SECTOR RETURN GAP / DISPERSION OVER TIME & KEY SECTORS
# ==============================================================================
p_df = df[~bad_close_mask].copy()
p_df["ret"] = p_df.groupby("code", sort=False)["close"].pct_change()
extreme_ret_mask = p_df["ret"].notna() & ((p_df["ret"] < -0.90) | (p_df["ret"] > 1.00))
extreme_ret_count = int(extreme_ret_mask.sum())
p_df = p_df[~extreme_ret_mask]

sec_daily = (
    p_df.dropna(subset=["ret"])
    .groupby(["date", "sector_clean"], as_index=False, sort=False)
    .agg(sec_ret=("ret", "mean"), n_firms=("code", "nunique"))
)
sec_daily = sec_daily[sec_daily["n_firms"] >= 3]
sec_pivot = sec_daily.pivot(index="date", columns="sector_clean", values="sec_ret").sort_index()

sec_cum_ret = ((1.0 + sec_pivot.fillna(0.0)).prod(axis=0) - 1.0) * 100.0
q2_sector_cum_df = (
    sec_cum_ret.sort_values(ascending=False)
    .rename_axis("sector")
    .reset_index(name="cum_return_pct")
)

disp_df = pd.DataFrame({
    "daily_std": sec_pivot.std(axis=1, skipna=True),
    "daily_iqr": sec_pivot.quantile(0.75, axis=1) - sec_pivot.quantile(0.25, axis=1),
}).dropna(how="all")

half_idx = max(1, len(disp_df) // 2)
q2_early_disp = ci95_mean(disp_df.iloc[:half_idx]["daily_std"])
q2_late_disp = ci95_mean(disp_df.iloc[half_idx:]["daily_std"])
q2_disp_diff = q2_late_disp["mean"] - q2_early_disp["mean"]

# Leave-one-sector-out contribution to cross-sector return variance
all_vals = sec_pivot.stack().dropna()
base_var = float(all_vals.var(ddof=1)) if len(all_vals) > 1 else np.nan
driver_rows = []
for col in sec_pivot.columns:
    sub_vals = sec_pivot.drop(columns=[col]).stack().dropna()
    sub_var = float(sub_vals.var(ddof=1)) if len(sub_vals) > 1 else np.nan
    var_drop = base_var - sub_var if (pd.notna(base_var) and pd.notna(sub_var)) else np.nan
    share_pct = (var_drop / base_var * 100.0) if (pd.notna(var_drop) and base_var > 0) else np.nan
    driver_rows.append({
        "sector": col,
        "var_drop_when_excluded": var_drop,
        "share_of_dispersion_pct": share_pct,
    })

q2_drivers_df = (
    pd.DataFrame(driver_rows)
    .sort_values("var_drop_when_excluded", ascending=False, kind="mergesort")
    .reset_index(drop=True)
)

# ==============================================================================
# Q3: CROSS-SECTOR CORRELATION OVER TIME & SHARP MARKET MOVES VS ORDINARY DAYS
# ==============================================================================
min_obs_corr = max(5, min(20, len(sec_pivot) // 2))
valid_cols = [c for c in sec_pivot.columns if sec_pivot[c].notna().sum() >= min_obs_corr]
c_pivot = sec_pivot[valid_cols].copy()

full_corr = c_pivot.corr()
upper_mask = np.triu(np.ones(full_corr.shape, dtype=bool), k=1)
full_pairs = full_corr.where(upper_mask).stack()
q3_overall_corr = ci95_mean(full_pairs)

q3_top30_pairs_df = (
    full_pairs.sort_values(ascending=False)
    .head(30)
    .reset_index()
)
q3_top30_pairs_df.columns = ["sector_a", "sector_b", "correlation"]

# Rolling 30-day average pairwise correlation over time
roll_30 = c_pivot.rolling(30, min_periods=10).corr()
roll_series_rows = []
for dt, sub_mat in roll_30.groupby(level=0):
    m = sub_mat.droplevel(0)
    vals = m.where(upper_mask).stack()
    if len(vals) > 0:
        roll_series_rows.append({"date": dt, "roll30_mean_corr": float(vals.mean())})
roll_corr_df = pd.DataFrame(roll_series_rows)

# Sharp market move days vs ordinary days (using equal-weighted market daily return)
mkt_daily_ret = c_pivot.mean(axis=1, skipna=True).dropna()
sharp_thresh = float(mkt_daily_ret.abs().quantile(0.90))
sharp_dates = set(mkt_daily_ret[mkt_daily_ret.abs() >= sharp_thresh].index)
ordinary_dates = set(mkt_daily_ret[mkt_daily_ret.abs() < sharp_thresh].index)

sharp_corr_mat = c_pivot.loc[c_pivot.index.isin(sharp_dates)].corr()
ord_corr_mat = c_pivot.loc[c_pivot.index.isin(ordinary_dates)].corr()

sharp_pairs = sharp_corr_mat.where(upper_mask).stack()
ord_pairs = ord_corr_mat.where(upper_mask).stack()

q3_sharp_ci = ci95_mean(sharp_pairs)
q3_ordinary_ci = ci95_mean(ord_pairs)

# ==============================================================================
# Q4: VOLUME SPIKE — DOES VOLUME LEAD PRICE OR FOLLOW IT? (+5, +10, +20 DAYS)
# ==============================================================================
v_df = p_df[~bad_volume_mask & (p_df["volume"] > 0)].sort_values(["code", "date"], kind="mergesort").copy()

g_code = v_df.groupby("code", sort=False)
v_df["vol_med20"] = g_code["volume"].transform(lambda s: s.rolling(20, min_periods=5).median().shift(1))
v_df["vol_mean20"] = g_code["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
v_df["ratio_med20"] = v_df["volume"] / v_df["vol_med20"]
v_df["ratio_mean20"] = v_df["volume"] / v_df["vol_mean20"]

# Post-spike returns (+5, +10, +20 trading days)
g_close = v_df.groupby("code", sort=False)["close"]
for h in (5, 10, 20):
    v_df[f"post_{h}d_pct"] = (g_close.shift(-h) / v_df["close"] - 1.0) * 100.0
    # Pre-spike returns (-5, -10, -20 trading days) to test whether volume follows prior price moves
    v_df[f"pre_{h}d_pct"] = (v_df["close"] / g_close.shift(h) - 1.0) * 100.0

definitions = [
    ("base_med20_ge_2.0x", v_df["ratio_med20"] >= 2.0),
    ("alt_med20_ge_1.5x", v_df["ratio_med20"] >= 1.5),
    ("alt_med20_ge_3.0x", v_df["ratio_med20"] >= 3.0),
    ("alt_mean20_ge_2.0x", v_df["ratio_mean20"] >= 2.0),
]

q4_results_by_def = {}
for def_name, mask in definitions:
    sub = v_df[mask]
    q4_results_by_def[def_name] = {
        "post_5d_pct": ci95_mean(sub["post_5d_pct"]),
        "post_10d_pct": ci95_mean(sub["post_10d_pct"]),
        "post_20d_pct": ci95_mean(sub["post_20d_pct"]),
        "pre_5d_pct": ci95_mean(sub["pre_5d_pct"]),
        "pre_10d_pct": ci95_mean(sub["pre_10d_pct"]),
        "pre_20d_pct": ci95_mean(sub["pre_20d_pct"]),
    }

q4_top30_spikes_df = (
    v_df[v_df["ratio_med20"] >= 2.0]
    .dropna(subset=["post_20d_pct"])
    .sort_values("ratio_med20", ascending=False, kind="mergesort")
    .head(30)[[
        "date", "code", "name_clean", "sector_clean", "volume", "ratio_med20",
        "pre_5d_pct", "post_5d_pct", "post_10d_pct", "post_20d_pct",
    ]]
    .rename(columns={"name_clean": "name", "sector_clean": "sector"})
    .reset_index(drop=True)
)
q4_top30_spikes_df.insert(0, "rank", q4_top30_spikes_df.index + 1)

# ==============================================================================
# SAVE UNIFIED REPORT (TXT + JSON) & CHART
# ==============================================================================
unified_payload = {
    "period": {"start_date": start_date, "end_date": end_date, "total_raw_rows": total_raw_rows},
    "defect_audit": {
        "extreme_daily_return_filtered_rows": extreme_ret_count,
        "by_issue": defect_summary_df.to_dict(orient="records"),
    },
    "q1_market_cap_ranking": {
        "top30_retained_count": len(retained_codes),
        "top30_newcomers_count": len(newcomer_codes),
        "top30_exited_count": len(exited_codes),
        "newcomer_sectors_count": newcomer_sectors_count,
        "top30_start": top30_start.to_dict(orient="records"),
        "top30_end": top30_end.to_dict(orient="records"),
        "sector_ranking_moves": q1_sector_moves_df.to_dict(orient="records"),
    },
    "q2_sector_return_gap": {
        "early_half_daily_std_ci95": q2_early_disp,
        "late_half_daily_std_ci95": q2_late_disp,
        "dispersion_change_late_minus_early": q2_disp_diff,
        "sector_cumulative_returns_pct": q2_sector_cum_df.to_dict(orient="records"),
        "dispersion_key_sectors_leave_one_out": q2_drivers_df.to_dict(orient="records"),
    },
    "q3_cross_sector_correlation": {
        "overall_pairwise_corr_ci95": q3_overall_corr,
        "sharp_market_move_days_corr_ci95": q3_sharp_ci,
        "ordinary_days_corr_ci95": q3_ordinary_ci,
        "sharp_move_abs_mkt_ret_threshold": sharp_thresh,
        "top30_correlated_sector_pairs": q3_top30_pairs_df.to_dict(orient="records"),
    },
    "q4_volume_lead_or_follow": {
        "definitions": q4_results_by_def,
        "top30_volume_spikes_base_med20_2x": q4_top30_spikes_df.to_dict(orient="records"),
    },
}

json_out = os.path.join(OUT_DIR, "q1_q4_unified_results.json")
with open(json_out, "w", encoding="utf-8") as jf:
    json.dump(unified_payload, jf, ensure_ascii=False, indent=2)

txt_out = os.path.join(OUT_DIR, "q1_q4_unified_report.txt")
with open(txt_out, "w", encoding="utf-8") as tf:
    tf.write("=== UNIFIED STATISTICAL REPORT: QUESTIONS 1-4 (WITH CI95 & DEFECT AUDIT) ===\n")
    tf.write(f"Date range: {start_date} -> {end_date} | Total raw rows: {total_raw_rows}\n")
    tf.write(f"Extreme daily return rows filtered (< -90% or > +100%): {extreme_ret_count}\n\n")
    tf.write("--- DEFECT AUDIT ---\n" + defect_summary_df.to_string(index=False) + "\n\n")

    tf.write("--- Q1: MARKET-CAP RANKING OVER 2.5 YEARS (TOP 30) ---\n")
    tf.write(
        f"Retained in Top 30: {len(retained_codes)} | Newcomers: {len(newcomer_codes)} | "
        f"Exited: {len(exited_codes)} | Newcomer sectors: {newcomer_sectors_count}\n\n"
    )
    tf.write("[Top 30 at Start]\n" + top30_start.to_string(index=False) + "\n\n")
    tf.write("[Top 30 at End]\n" + top30_end.to_string(index=False) + "\n\n")
    tf.write("[Sector Ranking Moves (Mean |Rank Change| & CI95)]\n" + q1_sector_moves_df.to_string(index=False) + "\n\n")

    tf.write("--- Q2: SECTOR RETURN GAP & DISPERSION ---\n")
    tf.write(
        f"Early-half daily std (CI95): {q2_early_disp['mean']:.6f} [{q2_early_disp['ci95_low']:.6f}, {q2_early_disp['ci95_high']:.6f}]\n"
        f"Late-half daily std (CI95):  {q2_late_disp['mean']:.6f} [{q2_late_disp['ci95_low']:.6f}, {q2_late_disp['ci95_high']:.6f}]\n"
        f"Change (Late - Early):       {q2_disp_diff:+.6f}\n\n"
    )
    tf.write("[Cumulative Return by Sector (%)]\n" + q2_sector_cum_df.to_string(index=False) + "\n\n")
    tf.write("[Key Sectors Driving Dispersion (Leave-One-Out)]\n" + q2_drivers_df.to_string(index=False) + "\n\n")

    tf.write("--- Q3: CROSS-SECTOR CORRELATION (SHARP MOVES VS ORDINARY DAYS) ---\n")
    tf.write(
        f"Overall pairwise corr (CI95):     {q3_overall_corr['mean']:.4f} [{q3_overall_corr['ci95_low']:.4f}, {q3_overall_corr['ci95_high']:.4f}]\n"
        f"Sharp market move days (CI95):    {q3_sharp_ci['mean']:.4f} [{q3_sharp_ci['ci95_low']:.4f}, {q3_sharp_ci['ci95_high']:.4f}] (n_pairs={q3_sharp_ci['n']})\n"
        f"Ordinary days corr (CI95):        {q3_ordinary_ci['mean']:.4f} [{q3_ordinary_ci['ci95_low']:.4f}, {q3_ordinary_ci['ci95_high']:.4f}] (n_pairs={q3_ordinary_ci['n']})\n\n"
    )
    tf.write("[Top 30 Correlated Sector Pairs]\n" + q3_top30_pairs_df.to_string(index=False) + "\n\n")

    tf.write("--- Q4: VOLUME SPIKE — LEAD VS FOLLOW (+5, +10, +20 DAYS & SENSITIVITY) ---\n")
    for d_key, d_val in q4_results_by_def.items():
        tf.write(f"[{d_key}]\n")
        for horizon in ("pre_5d_pct", "pre_10d_pct", "pre_20d_pct", "post_5d_pct", "post_10d_pct", "post_20d_pct"):
            h = d_val[horizon]
            tf.write(f"  {horizon:12s}: mean={h['mean']:+.3f}% [{h['ci95_low']:+.3f}, {h['ci95_high']:+.3f}] (n={h['n']})\n")
    tf.write("\n[Top 30 Volume Spike Events (base_med20_ge_2.0x)]\n" + q4_top30_spikes_df.to_string(index=False) + "\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("white")

p1 = q1_sector_moves_df.head(12).iloc[::-1]
axes[0, 0].barh(p1["sector"], p1["mean_abs_rank_move"])
axes[0, 0].set_title("Q1: Top 12 Sectors by Mean |Rank Move|")
axes[0, 0].set_xlabel("Mean |Rank Change|")

p2 = q2_sector_cum_df.head(12).iloc[::-1]
axes[0, 1].barh(p2["sector"], p2["cum_return_pct"])
axes[0, 1].set_title("Q2: Top 12 Sectors by Cumulative Return (%)")
axes[0, 1].set_xlabel("Cumulative Return (%)")

if len(roll_corr_df) > 0:
    axes[1, 0].plot(roll_corr_df["date"], roll_corr_df["roll30_mean_corr"])
    axes[1, 0].set_title("Q3: Rolling 30-Day Mean Cross-Sector Correlation")
    axes[1, 0].set_xlabel("Trading Date")
    axes[1, 0].tick_params(axis="x", rotation=25)

regimes = ["Sharp Move Days", "Ordinary Days"]
corr_means = [q3_sharp_ci["mean"], q3_ordinary_ci["mean"]]
axes[1, 1].bar(regimes, corr_means)
axes[1, 1].set_title("Q3: Mean Cross-Sector Correlation (Sharp vs Ordinary)")
axes[1, 1].set_ylabel("Mean Pairwise Correlation")

for ax in axes.flat:
    ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
chart_out = os.path.join(OUT_DIR, "q1_q4_overview.png")
plt.savefig(chart_out, dpi=200)
plt.close(fig)

print("=== Q1-Q4 UNIFIED ANALYSIS COMPLETE ===")
print(f"Unified report TXT: {txt_out}")
print(f"Unified results JSON: {json_out}")
print(f"Defect audit CSV: {defect_audit_path}")
print(f"Chart PNG: {chart_out}")
