#!/usr/bin/env python3
"""
Расчёт недостающих данных для раздела 4.3.
Использует уже готовые CSV (capital_efficiency_details.csv, hourly_...) для верифицированных портфелей.
Дополняет hourly META-C и META-MEAN(BCD).
Масштабирование и итоговые таблицы для ВСЕХ 6 МСП.
"""
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
BPY = 252

STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
BCD_YEARS = [2022, 2023, 2024, 2025]
COMM_H_INST = 0.0004
COMM_H_RETL = 0.0005
COMM_D_INST = 0.0005
COMM_D_RETL = 0.0006
K_VALS = [1, 2, 3, 5, 8]

BEST_MAP_HOURLY = {
    "S1_MeanRev": "D", "S2_Bollinger": "D", "S3_Donchian": "B",
    "S4_Supertrend": "B", "S5_PivotPoints": "D", "S6_VWAP": "D",
}

META_NAMES = ["META-A", "META-B", "META-C", "META-D", "META-MEAN(BCD)", "META-BEST"]

# ── Helpers ──────────────────────────────────────────────────────
def calc_metrics(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return dict(Return_pct=0, Vol_pct=0, Sharpe=0, MDD_pct=0, Calmar=0)
    ann_ret = np.mean(r) * BPY
    ann_vol = np.std(r, ddof=1) * np.sqrt(BPY)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0
    cum = np.cumprod(1 + r)
    rmax = np.maximum.accumulate(cum)
    mdd = np.min((cum - rmax) / np.where(rmax > 0, rmax, 1))
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0
    return dict(Return_pct=round(ann_ret*100, 2), Vol_pct=round(ann_vol*100, 2),
                Sharpe=round(sharpe, 2), MDD_pct=round(mdd*100, 2), Calmar=round(calmar, 2))

def net_returns_series(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm

def strategy_portfolio_returns(df_sub, comm):
    ticker_rets = {}
    for tkr, g in df_sub.groupby("ticker"):
        g = g.sort_values("date")
        nr = net_returns_series(g["position"].values, g["daily_gross_return"].values, comm)
        ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
    if not ticker_rets:
        return pd.Series(dtype=float)
    return pd.DataFrame(ticker_rets).sort_index().fillna(0.0).mean(axis=1)


# ══════════════════════════════════════════════════════════════════
# PART 0: LOAD EXISTING DATA
# ══════════════════════════════════════════════════════════════════
print("Loading existing data...")
det_d = pd.read_csv("results/capital_efficiency_details.csv", index_col=0, parse_dates=True)
det_h = pd.read_csv("results/hourly_capital_efficiency_details.csv", index_col=0, parse_dates=True)

print(f"  Daily details: {det_d.shape[0]} days, cols={det_d.shape[1]}")
print(f"  Hourly details: {det_h.shape[0]} days, cols={det_h.shape[1]}")

# Extract daily data for all 6 MSP
daily_data = {}
for p in META_NAMES:
    col_p = p
    daily_data[p] = {
        "strat_r": det_d[f"{col_p}_strat"].values,
        "exposure": det_d[f"{col_p}_exposure"].values,
        "mmf": det_d[f"{col_p}_mmf"].values,
        "total": det_d[f"{col_p}_total"].values,
    }

# Extract MMF daily rate from daily data
free_cap_a = np.maximum(0, 1 - daily_data["META-A"]["exposure"])
mask = free_cap_a > 0.01
mmf_rate_daily = np.where(mask, daily_data["META-A"]["mmf"] / free_cap_a, 0)

# Hourly: A, B, D, BEST exist; need C and MEAN(BCD)
hourly_data = {}
for p in ["META-A", "META-B", "META-D", "META-BEST"]:
    hourly_data[p] = {
        "strat_r": det_h[f"{p}_strat"].values,
        "exposure": det_h[f"{p}_exposure"].values,
        "mmf": det_h[f"{p}_mmf"].values,
        "total": det_h[f"{p}_total"].values,
    }

# Hourly MMF rate
free_cap_ha = np.maximum(0, 1 - hourly_data["META-A"]["exposure"])
mask_h = free_cap_ha > 0.01
mmf_rate_hourly = np.where(mask_h, hourly_data["META-A"]["mmf"] / free_cap_ha, 0)

n_d = len(det_d)
n_h = len(det_h)


# ══════════════════════════════════════════════════════════════════
# PART 1: BUILD MISSING HOURLY META-C AND META-MEAN(BCD)
# ══════════════════════════════════════════════════════════════════
print("\nBuilding hourly META-C and META-MEAN(BCD)...")

pos_all = pd.read_parquet("results/final/strategies/walkforward_v4/daily_positions.parquet")
pos_all["date"] = pd.to_datetime(pos_all["date"])
pos_all = pos_all[pos_all["test_year"].isin(BCD_YEARS)]
pos_hourly = pos_all[pos_all["tf"] == "hourly"].copy()
print(f"  Hourly positions: {len(pos_hourly):,} rows")

# Build hourly per-(strategy, approach) portfolios
strat_port_h = {}
for (strat, app), g in pos_hourly.groupby(["strategy", "approach"]):
    strat_port_h[(strat, app)] = strategy_portfolio_returns(g, COMM_H_INST)

# Also build with retail commission
strat_port_h_retl = {}
for (strat, app), g in pos_hourly.groupby(["strategy", "approach"]):
    strat_port_h_retl[(strat, app)] = strategy_portfolio_returns(g, COMM_H_RETL)

# META-C hourly
meta_c_h_parts = {}
for s in STRATEGIES:
    key = (s, "C")
    if key in strat_port_h:
        meta_c_h_parts[s] = strat_port_h[key]
meta_c_hourly = pd.DataFrame(meta_c_h_parts).fillna(0.0).mean(axis=1)
meta_c_hourly.index = pd.to_datetime(meta_c_hourly.index)
meta_c_hourly = meta_c_hourly.sort_index()

# META-MEAN(BCD) hourly
meta_mean_h_parts = {}
for s in STRATEGIES:
    bcd = {}
    for app in ["B", "C", "D"]:
        key = (s, app)
        if key in strat_port_h:
            bcd[app] = strat_port_h[key]
    if bcd:
        meta_mean_h_parts[s] = pd.DataFrame(bcd).fillna(0.0).mean(axis=1)
meta_mean_hourly = pd.DataFrame(meta_mean_h_parts).fillna(0.0).mean(axis=1)
meta_mean_hourly.index = pd.to_datetime(meta_mean_hourly.index)
meta_mean_hourly = meta_mean_hourly.sort_index()

# Retail versions
meta_c_h_retl_parts = {}
for s in STRATEGIES:
    key = (s, "C")
    if key in strat_port_h_retl:
        meta_c_h_retl_parts[s] = strat_port_h_retl[key]
meta_c_hourly_retl = pd.DataFrame(meta_c_h_retl_parts).fillna(0.0).mean(axis=1).sort_index()

meta_mean_h_retl_parts = {}
for s in STRATEGIES:
    bcd = {}
    for app in ["B", "C", "D"]:
        key = (s, app)
        if key in strat_port_h_retl:
            bcd[app] = strat_port_h_retl[key]
    if bcd:
        meta_mean_h_retl_parts[s] = pd.DataFrame(bcd).fillna(0.0).mean(axis=1)
meta_mean_hourly_retl = pd.DataFrame(meta_mean_h_retl_parts).fillna(0.0).mean(axis=1).sort_index()

# Aggregate hourly -> daily
def agg_h2d(series):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    dates_norm = s.index.normalize()
    return (1 + s).groupby(dates_norm).prod() - 1

meta_c_daily_from_h = agg_h2d(meta_c_hourly)
meta_mean_daily_from_h = agg_h2d(meta_mean_hourly)
meta_c_daily_from_h_retl = agg_h2d(meta_c_hourly_retl)
meta_mean_daily_from_h_retl = agg_h2d(meta_mean_hourly_retl)

# Hourly exposure for C and MEAN(BCD)
def hourly_exposure_single(approach):
    sub = pos_hourly[pos_hourly["approach"] == approach].copy()
    exp_per_ts = sub.groupby("date").apply(
        lambda g: (g["position"].abs() > 0.001).mean(), include_groups=False)
    exp_per_ts.index = pd.to_datetime(exp_per_ts.index)
    return exp_per_ts.groupby(exp_per_ts.index.normalize()).mean()

exp_c_h = hourly_exposure_single("C")
exp_b_h = hourly_exposure_single("B")
exp_d_h = hourly_exposure_single("D")

# MEAN(BCD) exposure = average of B, C, D exposures
exp_mean_h = pd.DataFrame({"B": exp_b_h, "C": exp_c_h, "D": exp_d_h}).fillna(0).mean(axis=1)

# Add to hourly_data
h_idx = det_h.index
hourly_data["META-C"] = {
    "strat_r": meta_c_daily_from_h.reindex(h_idx).fillna(0).values,
    "exposure": exp_c_h.reindex(h_idx).fillna(0).values,
}
hourly_data["META-C"]["mmf"] = np.maximum(0, 1 - hourly_data["META-C"]["exposure"]) * mmf_rate_hourly
hourly_data["META-C"]["total"] = hourly_data["META-C"]["strat_r"] + hourly_data["META-C"]["mmf"]

hourly_data["META-MEAN(BCD)"] = {
    "strat_r": meta_mean_daily_from_h.reindex(h_idx).fillna(0).values,
    "exposure": exp_mean_h.reindex(h_idx).fillna(0).values,
}
hourly_data["META-MEAN(BCD)"]["mmf"] = np.maximum(0, 1 - hourly_data["META-MEAN(BCD)"]["exposure"]) * mmf_rate_hourly
hourly_data["META-MEAN(BCD)"]["total"] = hourly_data["META-MEAN(BCD)"]["strat_r"] + hourly_data["META-MEAN(BCD)"]["mmf"]

print("  Done. All 6 hourly meta-portfolios ready.")


# ══════════════════════════════════════════════════════════════════
# PART 1B: RETAIL HOURLY RETURNS (for B and C too)
# ══════════════════════════════════════════════════════════════════
# Build retail hourly for ALL portfolios
h_retl_data = {}
for p in ["META-A", "META-B", "META-D", "META-BEST"]:
    # Re-build with retail commission
    pass  # Will build from strat_port_h_retl

def build_meta_h_retl(approach):
    parts = {}
    for s in STRATEGIES:
        key = (s, approach)
        if key in strat_port_h_retl:
            parts[s] = strat_port_h_retl[key]
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1).sort_index()

def build_meta_best_h_retl():
    parts = {}
    for s, app in BEST_MAP_HOURLY.items():
        key = (s, app)
        if key in strat_port_h_retl:
            parts[s] = strat_port_h_retl[key]
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1).sort_index()

h_retl_raw = {
    "META-A": build_meta_h_retl("A"),
    "META-B": build_meta_h_retl("B"),
    "META-C": meta_c_hourly_retl,
    "META-D": build_meta_h_retl("D"),
    "META-MEAN(BCD)": meta_mean_hourly_retl,
    "META-BEST": build_meta_best_h_retl(),
}
h_retl_daily = {k: agg_h2d(v) for k, v in h_retl_raw.items()}

# Also build daily retail returns
pos_daily = pos_all[pos_all["tf"] == "daily"].copy()
strat_port_d_retl = {}
for (strat, app), g in pos_daily.groupby(["strategy", "approach"]):
    strat_port_d_retl[(strat, app)] = strategy_portfolio_returns(g, COMM_D_RETL)

BEST_MAP_DAILY = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}

def build_meta_d_retl(approach):
    parts = {}
    for s in STRATEGIES:
        key = (s, approach)
        if key in strat_port_d_retl:
            parts[s] = strat_port_d_retl[key]
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1).sort_index()

def build_meta_best_d_retl():
    parts = {}
    for s, app in BEST_MAP_DAILY.items():
        key = (s, app)
        if key in strat_port_d_retl:
            parts[s] = strat_port_d_retl[key]
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1).sort_index()

def build_meta_mean_d_retl():
    parts = {}
    for s in STRATEGIES:
        bcd = {}
        for app in ["B", "C", "D"]:
            key = (s, app)
            if key in strat_port_d_retl:
                bcd[app] = strat_port_d_retl[key]
        if bcd:
            parts[s] = pd.DataFrame(bcd).fillna(0.0).mean(axis=1)
    return pd.DataFrame(parts).fillna(0.0).mean(axis=1).sort_index()

d_retl = {
    "META-A": build_meta_d_retl("A"),
    "META-B": build_meta_d_retl("B"),
    "META-C": build_meta_d_retl("C"),
    "META-D": build_meta_d_retl("D"),
    "META-MEAN(BCD)": build_meta_mean_d_retl(),
    "META-BEST": build_meta_best_d_retl(),
}


# ══════════════════════════════════════════════════════════════════
# SECTION 1: EXPOSURE ALL MSP
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1: ЭКСПОЗИЦИЯ ВСЕХ МСП (daily + hourly)")
print("="*70)
print(f"\n{'Портфель':<16} | {'Дн.ср.%':>7} | {'Дн.макс%':>8} | {'Дн.своб%':>8} | "
      f"{'Час.ср.%':>8} | {'Час.мак%':>8} | {'Час.своб%':>9}")
print("-"*85)

for p in META_NAMES:
    dd = daily_data[p]
    hd = hourly_data[p]
    d_avg = np.mean(dd["exposure"]) * 100
    d_max = np.max(dd["exposure"]) * 100
    h_avg = np.mean(hd["exposure"]) * 100
    h_max = np.max(hd["exposure"]) * 100
    print(f"{p:<16} | {d_avg:>7.1f} | {d_max:>8.1f} | {100-d_avg:>8.1f} | "
          f"{h_avg:>8.1f} | {h_max:>8.1f} | {100-h_avg:>9.1f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: CAPITAL EFFICIENCY (1x + ФДР)
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2: КАПИТАЛЬНАЯ ЭФФЕКТИВНОСТЬ (1x + ФДР)")
print("="*70)

def print_fdr_block(title, data_dict, mmf_rate_arr, n):
    print(f"\n### {title}")
    print(f"{'Портфель':<16} | {'Стратег%':>8} | {'+ФДР':>6} | {'Итого%':>7} | "
          f"{'Sharpe':>6} | {'MDD%':>6} | {'Calmar':>6}")
    print("-"*72)
    for p in META_NAMES:
        d = data_dict[p]
        strat_m = calc_metrics(d["strat_r"][:n])
        total_m = calc_metrics(d["total"][:n])
        fdr = total_m["Return_pct"] - strat_m["Return_pct"]
        print(f"{p:<16} | {strat_m['Return_pct']:>8.2f} | {fdr:>+6.2f} | "
              f"{total_m['Return_pct']:>7.2f} | {total_m['Sharpe']:>6.2f} | "
              f"{total_m['MDD_pct']:>6.2f} | {total_m['Calmar']:>6.2f}")

print_fdr_block("Daily (инст. 0,05%)", daily_data, mmf_rate_daily, n_d)
print_fdr_block("Hourly (инст. 0,04%)", hourly_data, mmf_rate_hourly, n_h)


# ══════════════════════════════════════════════════════════════════
# SECTION 3: SCALING ALL MSP
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3: МАСШТАБИРОВАНИЕ ВСЕХ МСП + ФДР")
print("="*70)

def scaling_table(strat_r, exp, mmf_rate, k_vals):
    rows = []
    for k in k_vals:
        scaled = k * strat_r
        eff_exp = k * exp
        free = np.maximum(0, 1 - eff_exp)
        mmf_r = free * mmf_rate
        total = scaled + mmf_r
        m = calc_metrics(total)
        rows.append({
            "k": k,
            "avg_exp": round(np.mean(eff_exp)*100, 1),
            "max_exp": round(np.max(eff_exp)*100, 1),
            "free_cap": round(np.mean(free)*100, 1),
            "pct_over": round(np.mean(eff_exp > 1.0)*100, 1),
            **m
        })
    return pd.DataFrame(rows)

def print_scaling_block(title, data_dict, mmf_rate_arr, n):
    for p in META_NAMES:
        d = data_dict[p]
        sr = d["strat_r"][:n]
        ex = d["exposure"][:n]
        mr = mmf_rate_arr[:n]
        df = scaling_table(sr, ex, mr, K_VALS)
        print(f"\n### {p} {title}")
        print(f"{'k':>3} | {'Ср.эксп%':>8} | {'Макс%':>6} | {'Своб%':>6} | "
              f"{'%>100':>5} | {'Ret%':>7} | {'Sharpe':>6} | {'MDD%':>7}")
        print("-"*65)
        for _, r in df.iterrows():
            print(f"{int(r['k']):>2}x | {r['avg_exp']:>8.1f} | {r['max_exp']:>6.1f} | "
                  f"{r['free_cap']:>6.1f} | {r['pct_over']:>5.1f} | "
                  f"{r['Return_pct']:>7.2f} | {r['Sharpe']:>6.2f} | {r['MDD_pct']:>7.2f}")

print_scaling_block("daily + ФДР (инст. 0,05%)", daily_data, mmf_rate_daily, n_d)
print_scaling_block("hourly + ФДР (инст. 0,04%)", hourly_data, mmf_rate_hourly, n_h)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: SUMMARY TABLE 3x daily / 5x hourly
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 4: ИТОГОВАЯ ТАБЛИЦА (3x daily / 5x hourly + ФДР)")
print("="*70)

def compute_at_k(data_dict, mmf_rate_arr, n, k):
    rows = []
    for p in META_NAMES:
        d = data_dict[p]
        sr = d["strat_r"][:n]
        ex = d["exposure"][:n]
        mr = mmf_rate_arr[:n]
        scaled = k * sr
        free = np.maximum(0, 1 - k * ex)
        total = scaled + free * mr
        m = calc_metrics(total)
        rows.append({"name": p, **m})
    return pd.DataFrame(rows)

def compute_retl_at_k(retl_rets_dict, exp_dict, mmf_rate_arr, n, k, daily_idx=None, hourly_idx=None):
    """Compute using retail returns + existing exposure."""
    rows = []
    for p in META_NAMES:
        # Get retail returns
        if p in retl_rets_dict:
            r_series = retl_rets_dict[p]
            if hasattr(r_series, "values"):
                if daily_idx is not None:
                    r_vals = r_series.reindex(daily_idx).fillna(0).values[:n]
                elif hourly_idx is not None:
                    r_vals = r_series.reindex(hourly_idx).fillna(0).values[:n]
                else:
                    r_vals = r_series.values[:n]
            else:
                r_vals = r_series[:n]
        else:
            r_vals = np.zeros(n)

        ex = exp_dict[p]["exposure"][:n]
        mr = mmf_rate_arr[:n]
        scaled = k * r_vals
        free = np.maximum(0, 1 - k * ex)
        total = scaled + free * mr
        m = calc_metrics(total)
        rows.append({"name": p, **m})
    return pd.DataFrame(rows)


def print_summary(title, d_rows, h_rows):
    print(f"\n### {title}")
    print(f"{'':>16} | {'Дн.Ret%':>7} | {'Дн.Vol%':>7} | {'Дн.Sh':>6} | {'Дн.MDD%':>7} | "
          f"{'Час.Ret%':>8} | {'Час.Vol%':>8} | {'Час.Sh':>6} | {'Час.MDD%':>8}")
    print("-"*105)
    print(f"{'IMOEX':<16} | {-3.54:>7.2f} | {29.17:>7.2f} | {-0.12:>6.2f} | {-50.76:>7.2f} | "
          f"{-3.54:>8.2f} | {29.17:>8.2f} | {-0.12:>6.2f} | {-50.76:>8.2f}")
    for i, p in enumerate(META_NAMES):
        dr = d_rows.iloc[i]
        hr = h_rows.iloc[i]
        label = f"{p}+ФДР"
        print(f"{label:<16} | {dr['Return_pct']:>7.2f} | {dr['Vol_pct']:>7.2f} | "
              f"{dr['Sharpe']:>6.2f} | {dr['MDD_pct']:>7.2f} | "
              f"{hr['Return_pct']:>8.2f} | {hr['Vol_pct']:>8.2f} | "
              f"{hr['Sharpe']:>6.2f} | {hr['MDD_pct']:>8.2f}")

# Institutional
d_inst_3x = compute_at_k(daily_data, mmf_rate_daily, n_d, 3)
h_inst_5x = compute_at_k(hourly_data, mmf_rate_hourly, n_h, 5)
print_summary("Институциональные издержки (3x daily / 5x hourly + ФДР)", d_inst_3x, h_inst_5x)

# Retail
d_retl_3x = compute_retl_at_k(d_retl, daily_data, mmf_rate_daily, n_d, 3,
                                daily_idx=det_d.index)
h_retl_5x = compute_retl_at_k(h_retl_daily, hourly_data, mmf_rate_hourly, n_h, 5,
                                hourly_idx=det_h.index)
print_summary("Розничные издержки (3x daily / 5x hourly + ФДР)", d_retl_3x, h_retl_5x)


# ══════════════════════════════════════════════════════════════════
# SECTION 5: SHARPE AT TWO COMMISSION LEVELS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 5: SHARPE ПРИ ДВУХ УРОВНЯХ КОМИССИЙ (без ФДР)")
print("="*70)

# Daily institutional: from existing CSV
# Daily retail: from d_retl
# Hourly inst: from hourly_data strat_r (already inst)
# Hourly retail: from h_retl_daily

print(f"\n{'Портфель':<16} | {'Дн.0,05%':>8} | {'Дн.0,06%':>8} | {'Час.0,04%':>9} | {'Час.0,05%':>9}")
print("-"*70)

for p in META_NAMES:
    # Daily inst
    d_i = calc_metrics(daily_data[p]["strat_r"])
    # Daily retail
    r_retl = d_retl[p]
    if hasattr(r_retl, "reindex"):
        d_r_vals = r_retl.reindex(det_d.index).fillna(0).values
    else:
        d_r_vals = r_retl
    d_r = calc_metrics(d_r_vals)
    # Hourly inst
    h_i = calc_metrics(hourly_data[p]["strat_r"])
    # Hourly retail
    h_r_series = h_retl_daily.get(p)
    if h_r_series is not None and hasattr(h_r_series, "reindex"):
        h_r_vals = h_r_series.reindex(det_h.index).fillna(0).values
    else:
        h_r_vals = np.zeros(n_h)
    h_r = calc_metrics(h_r_vals)

    print(f"{p:<16} | {d_i['Sharpe']:>8.2f} | {d_r['Sharpe']:>8.2f} | "
          f"{h_i['Sharpe']:>9.2f} | {h_r['Sharpe']:>9.2f}")


# ══════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

print("\nОжидаемые значения (из FINAL_RESULTS Section 11):")
expected = [
    ("BEST daily 3x+ФДР", "META-BEST", daily_data, mmf_rate_daily, n_d, 3, 27.94, 3.34, -5.08),
    ("D daily 3x+ФДР",    "META-D",    daily_data, mmf_rate_daily, n_d, 3, 27.45, 3.29, -5.20),
]
for label, p, dd, mr, n, k, exp_ret, exp_sh, exp_mdd in expected:
    sr = dd[p]["strat_r"][:n]
    ex = dd[p]["exposure"][:n]
    total = k * sr + np.maximum(0, 1 - k * ex) * mr[:n]
    m = calc_metrics(total)
    ok_ret = "✓" if abs(m["Return_pct"] - exp_ret) < 0.5 else "✗"
    ok_sh = "✓" if abs(m["Sharpe"] - exp_sh) < 0.3 else "✗"
    ok_mdd = "✓" if abs(m["MDD_pct"] - exp_mdd) < 0.5 else "✗"
    print(f"  {label}: Ret={m['Return_pct']} (exp {exp_ret}) {ok_ret}, "
          f"Sh={m['Sharpe']} (exp {exp_sh}) {ok_sh}, "
          f"MDD={m['MDD_pct']} (exp {exp_mdd}) {ok_mdd}")

# Hourly
expected_h = [
    ("BEST hourly 5x+ФДР", "META-BEST", hourly_data, mmf_rate_hourly, n_h, 5, 25.07, 4.87, -1.10),
    ("D hourly 5x+ФДР",    "META-D",    hourly_data, mmf_rate_hourly, n_h, 5, 41.22, 2.62, -7.74),
]
for label, p, dd, mr, n, k, exp_ret, exp_sh, exp_mdd in expected_h:
    sr = dd[p]["strat_r"][:n]
    ex = dd[p]["exposure"][:n]
    total = k * sr + np.maximum(0, 1 - k * ex) * mr[:n]
    m = calc_metrics(total)
    ok_ret = "✓" if abs(m["Return_pct"] - exp_ret) < 0.5 else "✗"
    ok_sh = "✓" if abs(m["Sharpe"] - exp_sh) < 0.3 else "✗"
    ok_mdd = "✓" if abs(m["MDD_pct"] - exp_mdd) < 0.5 else "✗"
    print(f"  {label}: Ret={m['Return_pct']} (exp {exp_ret}) {ok_ret}, "
          f"Sh={m['Sharpe']} (exp {exp_sh}) {ok_sh}, "
          f"MDD={m['MDD_pct']} (exp {exp_mdd}) {ok_mdd}")

print("\nDone.")
