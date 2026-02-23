#!/usr/bin/env python3
"""
Расчёт МСП-AVG (MEAN(BCD)) и экономической ценности Δ для раздела 4.4.

Вычисляет:
1. Per-strategy returns & volatility по подходам (A, B, C, D, AVG(BCD))
2. META-A, META-AVG(BCD), META-BEST at 3x daily + ФДР и 5x hourly + ФДР
3. Δ = ΔE(r) + (γ/2) * [σ²_A − σ²_PORT] при γ = 1..15
4. Таблица 4.29 (A vs BEST, A vs AVG) при γ = 3, 5, 10
5. Минимальный капитал = 3 млн / Δ
"""
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
STRAT_NAMES = {
    "S1_MeanRev": "Скольз. средние MR",
    "S2_Bollinger": "Bollinger Bands",
    "S3_Donchian": "Donchian Channels",
    "S4_Supertrend": "Supertrend",
    "S5_PivotPoints": "Pivot Points",
    "S6_VWAP": "VWAP",
}
APP_NAMES = {"B": "Адапт. стопы", "C": "Режим. фильтрация", "D": "Vol-targeting"}
BCD_YEARS = [2022, 2023, 2024, 2025]
BPY_D = 252
BPY_H = 2268
COMM_D = 0.0005
COMM_H = 0.0004
COST_ANNUAL = 3_000_000  # 3 млн руб. управленческие расходы

BEST_MAP_DAILY = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}
BEST_MAP_HOURLY = {
    "S1_MeanRev": "D", "S2_Bollinger": "D", "S3_Donchian": "B",
    "S4_Supertrend": "B", "S5_PivotPoints": "D", "S6_VWAP": "D",
}


# ── Helpers ──────────────────────────────────────────────────────
def net_returns(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm


def strategy_ew_returns(df_sub, comm):
    """EW portfolio of all tickers for one (strategy, approach)."""
    ticker_rets = {}
    for tkr, g in df_sub.groupby("ticker"):
        g = g.sort_values("date")
        nr = net_returns(g["position"].values, g["daily_gross_return"].values, comm)
        ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
    if not ticker_rets:
        return pd.Series(dtype=float)
    return pd.DataFrame(ticker_rets).sort_index().fillna(0.0).mean(axis=1)


def exposure_series(df_sub):
    """Fraction of (ticker) positions with |pos| > 0.001, per date."""
    return df_sub.groupby("date").apply(
        lambda g: (g["position"].abs() > 0.001).mean(), include_groups=False)


def agg_h2d(series):
    """Aggregate hourly returns to daily by compounding."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return (1 + s).groupby(s.index.normalize()).prod() - 1


def ann_metrics(daily_rets, bpy=252):
    """Annualized return (%), vol (%), Sharpe from daily returns."""
    r = np.asarray(daily_rets, dtype=float)
    r = r[np.isfinite(r)]
    mu = np.mean(r) * bpy
    sigma = np.std(r, ddof=1) * np.sqrt(bpy)
    sharpe = mu / sigma if sigma > 1e-12 else 0
    return mu, sigma, sharpe


def delta_u(mu_port, sigma_port, mu_a, sigma_a, gamma):
    """
    Δ = [μ_port − μ_A] + (γ/2) * [σ²_A − σ²_port]

    All in decimals (e.g., 0.20 for 20%).
    Returns Δ in percentage points.
    """
    delta_r = mu_port - mu_a  # decimal
    delta_risk = (gamma / 2) * (sigma_a**2 - sigma_port**2)  # decimal
    delta_total = delta_r + delta_risk  # decimal
    return delta_r * 100, delta_risk * 100, delta_total * 100


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("Loading positions data...")
pos_all = pd.read_parquet("results/final/strategies/walkforward_v4/daily_positions.parquet")
pos_all["date"] = pd.to_datetime(pos_all["date"])
pos_all = pos_all[pos_all["test_year"].isin(BCD_YEARS)]

pos_daily = pos_all[pos_all["tf"] == "daily"].copy()
pos_hourly = pos_all[pos_all["tf"] == "hourly"].copy()
print(f"  Daily: {len(pos_daily):,} rows, Hourly: {len(pos_hourly):,} rows")

# Load capital efficiency details for MMF rates
det_d = pd.read_csv("results/capital_efficiency_details.csv", index_col=0, parse_dates=True)
free_cap_a = np.maximum(0, 1 - det_d["META-A_exposure"].values)
mask = free_cap_a > 0.01
mmf_rate_daily = np.where(mask, det_d["META-A_mmf"].values / free_cap_a, 0)

det_h = pd.read_csv("results/hourly_capital_efficiency_details.csv", index_col=0, parse_dates=True)
free_cap_ha = np.maximum(0, 1 - det_h["META-A_exposure"].values)
mask_h = free_cap_ha > 0.01
mmf_rate_hourly = np.where(mask_h, det_h["META-A_mmf"].values / free_cap_ha, 0)


# ══════════════════════════════════════════════════════════════════
# BUILD PER-STRATEGY PORTFOLIOS
# ══════════════════════════════════════════════════════════════════
print("\nBuilding per-strategy portfolios...")

# Daily
daily_port = {}  # (strat, app) → Series of daily returns
daily_exp = {}   # (strat, app) → Series of exposure
for (strat, app), g in pos_daily.groupby(["strategy", "approach"]):
    daily_port[(strat, app)] = strategy_ew_returns(g, COMM_D)
    daily_exp[(strat, app)] = exposure_series(g)

# Hourly (raw hourly returns)
hourly_port_raw = {}
hourly_exp_raw = {}
for (strat, app), g in pos_hourly.groupby(["strategy", "approach"]):
    hourly_port_raw[(strat, app)] = strategy_ew_returns(g, COMM_H)
    hourly_exp_raw[(strat, app)] = exposure_series(g)

# Aggregate hourly → daily
hourly_port = {k: agg_h2d(v) for k, v in hourly_port_raw.items()}

# Hourly exposure → daily average
def hourly_exp_daily(exp_series):
    s = exp_series.copy()
    s.index = pd.to_datetime(s.index)
    return s.groupby(s.index.normalize()).mean()

hourly_exp = {k: hourly_exp_daily(v) for k, v in hourly_exp_raw.items()}

print(f"  Daily: {len(daily_port)} (strat,app) combos")
print(f"  Hourly: {len(hourly_port)} (strat,app) combos")


# ══════════════════════════════════════════════════════════════════
# SECTION 1: PER-STRATEGY RETURNS & VOLATILITY
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 1: Per-strategy доходность и волатильность (daily, net 0.05%)")
print("=" * 80)

print(f"\n{'Стратегия':<22} | {'Подход':<18} | {'Return%':>8} | {'Vol%':>7} | {'Sharpe':>7} | {'Экспоз%':>8}")
print("-" * 85)

for s in STRATEGIES:
    for app in ["A", "B", "C", "D"]:
        key = (s, app)
        if key in daily_port:
            mu, sigma, sh = ann_metrics(daily_port[key].values, BPY_D)
            exp_mean = daily_exp[key].mean() * 100 if key in daily_exp else 0
            app_name = APP_NAMES.get(app, "Базовый")
            print(f"{STRAT_NAMES[s]:<22} | {app_name:<18} | {mu*100:>8.2f} | {sigma*100:>7.2f} | {sh:>7.2f} | {exp_mean:>8.1f}")
    # AVG(BCD) for this strategy
    bcd_rets = []
    for app in ["B", "C", "D"]:
        key = (s, app)
        if key in daily_port:
            bcd_rets.append(daily_port[key])
    if bcd_rets:
        avg_ret = pd.DataFrame({i: r for i, r in enumerate(bcd_rets)}).fillna(0).mean(axis=1)
        mu, sigma, sh = ann_metrics(avg_ret.values, BPY_D)
        print(f"{STRAT_NAMES[s]:<22} | {'AVG(BCD)':<18} | {mu*100:>8.2f} | {sigma*100:>7.2f} | {sh:>7.2f} |")
    print()


# ══════════════════════════════════════════════════════════════════
# SECTION 2: BUILD META-PORTFOLIOS
# ══════════════════════════════════════════════════════════════════
def build_meta_daily(approach_map, port_dict, exp_dict):
    """Build META portfolio from per-strategy returns.
    approach_map: str (single approach) or dict {strat: approach}
    Returns (daily_returns_series, daily_exposure_series)
    """
    strat_rets = {}
    strat_exps = {}
    for s in STRATEGIES:
        app = approach_map if isinstance(approach_map, str) else approach_map[s]
        key = (s, app)
        if key in port_dict:
            strat_rets[s] = port_dict[key]
        if key in exp_dict:
            strat_exps[s] = exp_dict[key]

    ret_df = pd.DataFrame(strat_rets).sort_index().fillna(0)
    meta_ret = ret_df.mean(axis=1)

    exp_df = pd.DataFrame(strat_exps).sort_index().fillna(0)
    meta_exp = exp_df.mean(axis=1)

    return meta_ret, meta_exp


def build_meta_avg_daily(port_dict, exp_dict):
    """META-AVG: for each strategy, average B/C/D returns, then EW across strategies."""
    strat_avg_rets = {}
    strat_avg_exps = {}
    for s in STRATEGIES:
        bcd_r = {}
        bcd_e = {}
        for app in ["B", "C", "D"]:
            key = (s, app)
            if key in port_dict:
                bcd_r[app] = port_dict[key]
            if key in exp_dict:
                bcd_e[app] = exp_dict[key]
        if bcd_r:
            strat_avg_rets[s] = pd.DataFrame(bcd_r).fillna(0).mean(axis=1)
        if bcd_e:
            strat_avg_exps[s] = pd.DataFrame(bcd_e).fillna(0).mean(axis=1)

    ret_df = pd.DataFrame(strat_avg_rets).sort_index().fillna(0)
    meta_ret = ret_df.mean(axis=1)

    exp_df = pd.DataFrame(strat_avg_exps).sort_index().fillna(0)
    meta_exp = exp_df.mean(axis=1)

    return meta_ret, meta_exp


# Build daily metas
meta_a_d_ret, meta_a_d_exp = build_meta_daily("A", daily_port, daily_exp)
meta_best_d_ret, meta_best_d_exp = build_meta_daily(BEST_MAP_DAILY, daily_port, daily_exp)
meta_avg_d_ret, meta_avg_d_exp = build_meta_avg_daily(daily_port, daily_exp)

# Build hourly metas (from aggregated hourly→daily returns)
meta_a_h_ret, meta_a_h_exp = build_meta_daily("A", hourly_port, hourly_exp)
meta_best_h_ret, meta_best_h_exp = build_meta_daily(BEST_MAP_HOURLY, hourly_port, hourly_exp)
meta_avg_h_ret, meta_avg_h_exp = build_meta_avg_daily(hourly_port, hourly_exp)


# ══════════════════════════════════════════════════════════════════
# SECTION 3: APPLY SCALING + ФДР
# ══════════════════════════════════════════════════════════════════
def apply_scaling_fdr(meta_ret, meta_exp, mmf_rate, idx, k):
    """Apply k× scaling + ФДР.
    Returns daily total returns aligned to idx.
    """
    ret_aligned = meta_ret.reindex(idx).fillna(0).values
    exp_aligned = meta_exp.reindex(idx).fillna(0).values
    mr = mmf_rate[:len(idx)]

    scaled_ret = k * ret_aligned
    eff_exp = k * exp_aligned
    free = np.maximum(0, 1 - eff_exp)
    total = scaled_ret + free * mr
    return total, eff_exp


print("\n" + "=" * 80)
print("SECTION 2: META портфели — характеристики")
print("=" * 80)

# Daily 3x + ФДР
K_D = 3
total_a_d, exp_a_d = apply_scaling_fdr(meta_a_d_ret, meta_a_d_exp, mmf_rate_daily, det_d.index, K_D)
total_best_d, exp_best_d = apply_scaling_fdr(meta_best_d_ret, meta_best_d_exp, mmf_rate_daily, det_d.index, K_D)
total_avg_d, exp_avg_d = apply_scaling_fdr(meta_avg_d_ret, meta_avg_d_exp, mmf_rate_daily, det_d.index, K_D)

# Hourly 5x + ФДР
K_H = 5
total_a_h, exp_a_h = apply_scaling_fdr(meta_a_h_ret, meta_a_h_exp, mmf_rate_hourly, det_h.index, K_H)
total_best_h, exp_best_h = apply_scaling_fdr(meta_best_h_ret, meta_best_h_exp, mmf_rate_hourly, det_h.index, K_H)
total_avg_h, exp_avg_h = apply_scaling_fdr(meta_avg_h_ret, meta_avg_h_exp, mmf_rate_hourly, det_h.index, K_H)

# Print characteristics
print(f"\n{'Портфель':<28} | {'E(r)%':>7} | {'σ%':>7} | {'Sharpe':>7} | {'Ср.эксп%':>8}")
print("-" * 72)

for label, total, exp in [
    (f"МСП-A daily {K_D}x+ФДР", total_a_d, exp_a_d),
    (f"МСП-AVG daily {K_D}x+ФДР", total_avg_d, exp_avg_d),
    (f"МСП-BEST daily {K_D}x+ФДР", total_best_d, exp_best_d),
    (f"МСП-A hourly {K_H}x+ФДР", total_a_h, exp_a_h),
    (f"МСП-AVG hourly {K_H}x+ФДР", total_avg_h, exp_avg_h),
    (f"МСП-BEST hourly {K_H}x+ФДР", total_best_h, exp_best_h),
]:
    mu, sigma, sh = ann_metrics(total, BPY_D)
    avg_exp = np.mean(exp) * 100
    print(f"{label:<28} | {mu*100:>7.2f} | {sigma*100:>7.2f} | {sh:>7.2f} | {avg_exp:>8.1f}")

# Also print 1x + ФДР for reference
print(f"\n--- 1x + ФДР (reference) ---")
for label, meta_ret, meta_exp, mmf, idx in [
    ("МСП-A daily 1x+ФДР", meta_a_d_ret, meta_a_d_exp, mmf_rate_daily, det_d.index),
    ("МСП-AVG daily 1x+ФДР", meta_avg_d_ret, meta_avg_d_exp, mmf_rate_daily, det_d.index),
    ("МСП-BEST daily 1x+ФДР", meta_best_d_ret, meta_best_d_exp, mmf_rate_daily, det_d.index),
    ("МСП-A hourly 1x+ФДР", meta_a_h_ret, meta_a_h_exp, mmf_rate_hourly, det_h.index),
    ("МСП-AVG hourly 1x+ФДР", meta_avg_h_ret, meta_avg_h_exp, mmf_rate_hourly, det_h.index),
    ("МСП-BEST hourly 1x+ФДР", meta_best_h_ret, meta_best_h_exp, mmf_rate_hourly, det_h.index),
]:
    total_1x, _ = apply_scaling_fdr(meta_ret, meta_exp, mmf, idx, 1)
    mu, sigma, sh = ann_metrics(total_1x, BPY_D)
    print(f"  {label:<28} | {mu*100:>7.2f} | {sigma*100:>7.2f} | {sh:>7.2f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 4: DELTA COMPUTATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 3: Экономическая ценность Δ = ΔE(r) + (γ/2)·Δσ²")
print("=" * 80)

# Get annualized metrics in DECIMAL form
mu_a_d, sigma_a_d, _ = ann_metrics(total_a_d, BPY_D)
mu_best_d, sigma_best_d, _ = ann_metrics(total_best_d, BPY_D)
mu_avg_d, sigma_avg_d, _ = ann_metrics(total_avg_d, BPY_D)

mu_a_h, sigma_a_h, _ = ann_metrics(total_a_h, BPY_D)
mu_best_h, sigma_best_h, _ = ann_metrics(total_best_h, BPY_D)
mu_avg_h, sigma_avg_h, _ = ann_metrics(total_avg_h, BPY_D)

# Print key inputs
print(f"\nВходные параметры (annualized):")
print(f"  Daily {K_D}x+ФДР:")
print(f"    МСП-A:    μ = {mu_a_d*100:.2f}%, σ = {sigma_a_d*100:.2f}%, σ² = {sigma_a_d**2:.6f}")
print(f"    МСП-AVG:  μ = {mu_avg_d*100:.2f}%, σ = {sigma_avg_d*100:.2f}%, σ² = {sigma_avg_d**2:.6f}")
print(f"    МСП-BEST: μ = {mu_best_d*100:.2f}%, σ = {sigma_best_d*100:.2f}%, σ² = {sigma_best_d**2:.6f}")
print(f"  Hourly {K_H}x+ФДР:")
print(f"    МСП-A:    μ = {mu_a_h*100:.2f}%, σ = {sigma_a_h*100:.2f}%, σ² = {sigma_a_h**2:.6f}")
print(f"    МСП-AVG:  μ = {mu_avg_h*100:.2f}%, σ = {sigma_avg_h*100:.2f}%, σ² = {sigma_avg_h**2:.6f}")
print(f"    МСП-BEST: μ = {mu_best_h*100:.2f}%, σ = {sigma_best_h*100:.2f}%, σ² = {sigma_best_h**2:.6f}")

# Δσ² components
dsig2_best_d = sigma_a_d**2 - sigma_best_d**2
dsig2_avg_d = sigma_a_d**2 - sigma_avg_d**2
dsig2_best_h = sigma_a_h**2 - sigma_best_h**2
dsig2_avg_h = sigma_a_h**2 - sigma_avg_h**2

print(f"\n  Δσ² (A − PORT), decimal:")
print(f"    Daily BEST: {dsig2_best_d:.6f}  (σ_A²−σ_BEST²)")
print(f"    Daily AVG:  {dsig2_avg_d:.6f}  (σ_A²−σ_AVG²)")
print(f"    Hourly BEST: {dsig2_best_h:.6f}")
print(f"    Hourly AVG:  {dsig2_avg_h:.6f}")

# Δ at each gamma
print(f"\n--- Δ(γ) for Daily {K_D}x+ФДР ---")
print(f"{'γ':>3} | {'A vs BEST':>35} | {'A vs AVG':>35}")
print(f"    | {'ΔE(r)':>8} {'Δrisk':>8} {'Δ':>8} {'Мин.кап':>8} | {'ΔE(r)':>8} {'Δrisk':>8} {'Δ':>8} {'Мин.кап':>8}")
print("-" * 85)

for gamma in range(1, 16):
    dr_b, drisk_b, dt_b = delta_u(mu_best_d, sigma_best_d, mu_a_d, sigma_a_d, gamma)
    dr_a, drisk_a, dt_a = delta_u(mu_avg_d, sigma_avg_d, mu_a_d, sigma_a_d, gamma)
    cap_b = COST_ANNUAL / (dt_b / 100) if dt_b > 0.01 else float('inf')
    cap_a = COST_ANNUAL / (dt_a / 100) if dt_a > 0.01 else float('inf')
    cap_b_str = f"{cap_b/1e6:>6.1f}M" if cap_b < 1e12 else "∞"
    cap_a_str = f"{cap_a/1e6:>6.1f}M" if cap_a < 1e12 else "∞"
    print(f"{gamma:>3} | {dr_b:>+8.2f} {drisk_b:>+8.2f} {dt_b:>8.2f} {cap_b_str:>8} | "
          f"{dr_a:>+8.2f} {drisk_a:>+8.2f} {dt_a:>8.2f} {cap_a_str:>8}")

print(f"\n--- Δ(γ) for Hourly {K_H}x+ФДР ---")
print(f"{'γ':>3} | {'A vs BEST':>35} | {'A vs AVG':>35}")
print(f"    | {'ΔE(r)':>8} {'Δrisk':>8} {'Δ':>8} {'Мин.кап':>8} | {'ΔE(r)':>8} {'Δrisk':>8} {'Δ':>8} {'Мин.кап':>8}")
print("-" * 85)

for gamma in range(1, 16):
    dr_b, drisk_b, dt_b = delta_u(mu_best_h, sigma_best_h, mu_a_h, sigma_a_h, gamma)
    dr_a, drisk_a, dt_a = delta_u(mu_avg_h, sigma_avg_h, mu_a_h, sigma_a_h, gamma)
    cap_b = COST_ANNUAL / (dt_b / 100) if dt_b > 0.01 else float('inf')
    cap_a = COST_ANNUAL / (dt_a / 100) if dt_a > 0.01 else float('inf')
    cap_b_str = f"{cap_b/1e6:>6.1f}M" if cap_b < 1e12 else "∞"
    cap_a_str = f"{cap_a/1e6:>6.1f}M" if cap_a < 1e12 else "∞"
    print(f"{gamma:>3} | {dr_b:>+8.2f} {drisk_b:>+8.2f} {dt_b:>8.2f} {cap_b_str:>8} | "
          f"{dr_a:>+8.2f} {drisk_a:>+8.2f} {dt_a:>8.2f} {cap_a_str:>8}")


# ══════════════════════════════════════════════════════════════════
# SECTION 5: TABLE 4.29 FORMAT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 4: Таблица 4.29 — формат для диссертации (γ = 3, 5, 10)")
print("=" * 80)

print(f"\n{'Таймфрейм':<20} | {'Сравнение':<12} | {'γ':>3} | {'ΔE(r)':>8} | {'Δ(γ/2)σ²':>9} | "
      f"{'Δ':>6} | {'Доля риска':>10} | {'Мин.кап':>10}")
print("-" * 100)

for gamma in [3, 5, 10]:
    # Daily
    for label, mu_p, sigma_p in [
        ("A vs BEST", mu_best_d, sigma_best_d),
        ("A vs AVG", mu_avg_d, sigma_avg_d),
    ]:
        dr, drisk, dt = delta_u(mu_p, sigma_p, mu_a_d, sigma_a_d, gamma)
        risk_share = abs(drisk / dt * 100) if abs(dt) > 0.01 else 0
        cap = COST_ANNUAL / (dt / 100) if dt > 0.01 else float('inf')
        cap_str = f"{cap/1e6:.1f} млн" if cap < 1e12 else "∞"
        risk_str = f"{risk_share:.0f}" if risk_share <= 100 else "100+"
        print(f"{'Дневной '+str(K_D)+'x+ФДР':<20} | {label:<12} | {gamma:>3} | {dr:>+8.2f} | {drisk:>+9.2f} | "
              f"{dt:>6.2f} | {risk_str:>10} | {cap_str:>10}")

    # Hourly
    for label, mu_p, sigma_p in [
        ("A vs BEST", mu_best_h, sigma_best_h),
        ("A vs AVG", mu_avg_h, sigma_avg_h),
    ]:
        dr, drisk, dt = delta_u(mu_p, sigma_p, mu_a_h, sigma_a_h, gamma)
        risk_share = abs(drisk / dt * 100) if abs(dt) > 0.01 else 0
        cap = COST_ANNUAL / (dt / 100) if dt > 0.01 else float('inf')
        cap_str = f"{cap/1e6:.1f} млн" if cap < 1e12 else "∞"
        risk_str = f"{risk_share:.0f}" if risk_share <= 100 else "100+"
        print(f"{'Часовой '+str(K_H)+'x+ФДР':<20} | {label:<12} | {gamma:>3} | {dr:>+8.2f} | {drisk:>+9.2f} | "
              f"{dt:>6.2f} | {risk_str:>10} | {cap_str:>10}")

    print("-" * 100)


# ══════════════════════════════════════════════════════════════════
# SECTION 6: LINEAR REGRESSION Δ(γ) = a + (γ/2)*b
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 5: Линейная параметризация Δ(γ) = a + (γ/2)·b")
print("=" * 80)

# The formula: Δ(γ) = ΔE(r) + (γ/2) * Δσ²
# This is linear in γ with:
#   a = ΔE(r) (intercept when γ=0)
#   b = Δσ² (slope coefficient for γ/2)

cases = [
    ("Daily BEST", mu_best_d, sigma_best_d, mu_a_d, sigma_a_d),
    ("Daily AVG",  mu_avg_d, sigma_avg_d, mu_a_d, sigma_a_d),
    ("Hourly BEST", mu_best_h, sigma_best_h, mu_a_h, sigma_a_h),
    ("Hourly AVG",  mu_avg_h, sigma_avg_h, mu_a_h, sigma_a_h),
]

print(f"\n{'Случай':<15} | {'a = ΔE(r), п.п.':>16} | {'b = Δσ², decimal':>17} | {'Формула'}")
print("-" * 80)
for label, mu_p, sigma_p, mu_base, sigma_base in cases:
    a = (mu_p - mu_base) * 100  # п.п.
    b = (sigma_base**2 - sigma_p**2)  # decimal
    print(f"{label:<15} | {a:>16.2f} | {b:>17.6f} | Δ(γ) = {a:+.2f} + (γ/2)·{b*100:.4f}/100")
    # Also express as: Δ(γ) = a + (γ/2) * b * 100 (п.п.)
    print(f"{'':15} |{'':>16} |{'':>17} | = {a:+.2f} + (γ/2)·{b*100:.4f} (в п.п. при σ² × 100)")

print("\nДля графика 4.16:")
for label, mu_p, sigma_p, mu_base, sigma_base in cases:
    a = (mu_p - mu_base) * 100
    b = (sigma_base**2 - sigma_p**2) * 100
    print(f"  delta_{label.lower().replace(' ', '_')} = {a:.2f} + (gamma / 2) * {b:.4f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 7: RATIO AVG/BEST
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 6: Отношение Δ(AVG) / Δ(BEST)")
print("=" * 80)

for gamma in [3, 5, 10]:
    _, _, dt_best_d = delta_u(mu_best_d, sigma_best_d, mu_a_d, sigma_a_d, gamma)
    _, _, dt_avg_d = delta_u(mu_avg_d, sigma_avg_d, mu_a_d, sigma_a_d, gamma)
    _, _, dt_best_h = delta_u(mu_best_h, sigma_best_h, mu_a_h, sigma_a_h, gamma)
    _, _, dt_avg_h = delta_u(mu_avg_h, sigma_avg_h, mu_a_h, sigma_a_h, gamma)

    ratio_d = dt_avg_d / dt_best_d * 100 if abs(dt_best_d) > 0.01 else 0
    ratio_h = dt_avg_h / dt_best_h * 100 if abs(dt_best_h) > 0.01 else 0
    print(f"  γ={gamma:>2}: Daily Δ(AVG)/Δ(BEST) = {dt_avg_d:.2f}/{dt_best_d:.2f} = {ratio_d:.0f}%"
          f"   |   Hourly = {dt_avg_h:.2f}/{dt_best_h:.2f} = {ratio_h:.0f}%")


# ══════════════════════════════════════════════════════════════════
# SECTION 8: WATERFALL VALUES for fig_4_17
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 7: Waterfall значения (γ=5) для обновления fig_4_17")
print("=" * 80)

GAMMA = 5
for tf_label, mu_a, sigma_a, mu_port, sigma_port, port_label in [
    (f"Daily {K_D}x+ФДР", mu_a_d, sigma_a_d, mu_best_d, sigma_best_d, "BEST"),
    (f"Daily {K_D}x+ФДР", mu_a_d, sigma_a_d, mu_avg_d, sigma_avg_d, "AVG"),
    (f"Hourly {K_H}x+ФДР", mu_a_h, sigma_a_h, mu_best_h, sigma_best_h, "BEST"),
    (f"Hourly {K_H}x+ФДР", mu_a_h, sigma_a_h, mu_avg_h, sigma_avg_h, "AVG"),
]:
    U_A = (mu_a - (GAMMA / 2) * sigma_a**2) * 100  # п.п.
    U_PORT = (mu_port - (GAMMA / 2) * sigma_port**2) * 100
    delta_er = (mu_port - mu_a) * 100
    delta_risk = (GAMMA / 2) * (sigma_a**2 - sigma_port**2) * 100

    print(f"\n  {tf_label}, A vs {port_label}:")
    print(f"    U(МСП-A)     = {U_A:.2f}")
    print(f"    ΔE(r)        = {delta_er:+.2f}")
    print(f"    −Δ(γ/2)σ²    = {delta_risk:+.2f}")
    print(f"    U(МСП-{port_label:4s})  = {U_PORT:.2f}")
    print(f"    Δ total       = {delta_er + delta_risk:.2f}")

print("\nDone.")
