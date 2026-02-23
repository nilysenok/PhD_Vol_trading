#!/usr/bin/env python3
"""
Расчёт ВСЕХ недостающих данных для раздела 4.3:
1. Экспозиция МСП-C и МСП-MEAN(BCD)
2. ФДР для МСП-C и МСП-MEAN(BCD)
3. Масштабирование A, B, C, MEAN (+ верификация D, BEST)
4. Итоговая таблица всех МСП при 3x daily / 5x hourly (инст. + розн.)
5. Розничные Sharpe для B и C
"""
import numpy as np
import pandas as pd
import os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

# ── Constants ────────────────────────────────────────────────────
BPY_D = 252
BPY_H = 2268
COMM_D_INST = 0.0005   # 0.05% institutional daily
COMM_D_RETL = 0.0006   # 0.06% retail daily
COMM_H_INST = 0.0004   # 0.04% institutional hourly
COMM_H_RETL = 0.0005   # 0.05% retail hourly

BCD_YEARS = [2022, 2023, 2024, 2025]
STRATEGIES = ["S1_MeanRev", "S2_Bollinger", "S3_Donchian",
              "S4_Supertrend", "S5_PivotPoints", "S6_VWAP"]
K_VALS = [1, 2, 3, 5, 8]

BEST_MAP_DAILY = {
    "S1_MeanRev": "C", "S2_Bollinger": "C", "S3_Donchian": "B",
    "S4_Supertrend": "D", "S5_PivotPoints": "C", "S6_VWAP": "D",
}
BEST_MAP_HOURLY = {
    "S1_MeanRev": "D", "S2_Bollinger": "D", "S3_Donchian": "B",
    "S4_Supertrend": "B", "S5_PivotPoints": "D", "S6_VWAP": "D",
}

# ── Helpers ──────────────────────────────────────────────────────
def net_returns_series(pos, gross_r, comm):
    dpos = np.diff(pos, prepend=0.0)
    return gross_r - np.abs(dpos) * comm

def calc_metrics(r, bpy):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return dict(Return_pct=0, Vol_pct=0, Sharpe=0, MDD_pct=0, Calmar=0)
    ann_ret = np.mean(r) * bpy
    ann_vol = np.std(r, ddof=1) * np.sqrt(bpy)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0
    cum = np.cumprod(1 + r)
    rmax = np.maximum.accumulate(cum)
    mdd = np.min((cum - rmax) / np.where(rmax > 0, rmax, 1))
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0
    return dict(Return_pct=round(ann_ret*100, 2), Vol_pct=round(ann_vol*100, 2),
                Sharpe=round(sharpe, 2), MDD_pct=round(mdd*100, 2), Calmar=round(calmar, 2))


def strategy_portfolio_returns(df_sub, comm):
    """EW portfolio across tickers for a given (strategy, approach) subset."""
    ticker_rets = {}
    for tkr, g in df_sub.groupby("ticker"):
        g = g.sort_values("date")
        nr = net_returns_series(g["position"].values, g["daily_gross_return"].values, comm)
        ticker_rets[tkr] = pd.Series(nr, index=g["date"].values)
    if not ticker_rets:
        return pd.Series(dtype=float)
    df_r = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
    return df_r.mean(axis=1)


def build_meta(pos_df, approach, comm):
    """META-X for single approach (A, B, C, D): EW across 6 strategies."""
    strat_rets = []
    for s in STRATEGIES:
        sub = pos_df[(pos_df["strategy"] == s) & (pos_df["approach"] == approach)]
        sr = strategy_portfolio_returns(sub, comm)
        if len(sr) > 0:
            strat_rets.append(sr)
    if not strat_rets:
        return pd.Series(dtype=float)
    df_all = pd.DataFrame(strat_rets).T.fillna(0.0)
    return df_all.mean(axis=1)


def build_meta_best(pos_df, best_map, comm):
    """META-BEST: for each strategy use its best approach."""
    strat_rets = []
    for s in STRATEGIES:
        app = best_map[s]
        sub = pos_df[(pos_df["strategy"] == s) & (pos_df["approach"] == app)]
        sr = strategy_portfolio_returns(sub, comm)
        if len(sr) > 0:
            strat_rets.append(sr)
    df_all = pd.DataFrame(strat_rets).T.fillna(0.0)
    return df_all.mean(axis=1)


def build_meta_mean_bcd(pos_df, comm):
    """META-MEAN(BCD): for each strategy, average B+C+D, then EW across strategies."""
    strat_rets = []
    for s in STRATEGIES:
        bcd_parts = []
        for app in ["B", "C", "D"]:
            sub = pos_df[(pos_df["strategy"] == s) & (pos_df["approach"] == app)]
            sr = strategy_portfolio_returns(sub, comm)
            if len(sr) > 0:
                bcd_parts.append(sr)
        if bcd_parts:
            df_bcd = pd.DataFrame(bcd_parts).T.fillna(0.0)
            strat_rets.append(df_bcd.mean(axis=1))
    df_all = pd.DataFrame(strat_rets).T.fillna(0.0)
    return df_all.mean(axis=1)


def compute_exposure_single(pos_df, approach):
    """Compute daily exposure for a single-approach meta-portfolio."""
    sub = pos_df[pos_df["approach"] == approach].copy()
    # Per date: fraction of (strategy, ticker) with |position| > 0.001
    exp = sub.groupby("date").apply(lambda g: (g["position"].abs() > 0.001).mean())
    return exp


def compute_exposure_best(pos_df, best_map):
    """Compute daily exposure for META-BEST."""
    parts = []
    for s, app in best_map.items():
        sub = pos_df[(pos_df["strategy"] == s) & (pos_df["approach"] == app)]
        exp_s = sub.groupby("date").apply(lambda g: (g["position"].abs() > 0.001).mean())
        parts.append(exp_s)
    df_exp = pd.DataFrame(parts).T.fillna(0)
    return df_exp.mean(axis=1)


def compute_exposure_mean_bcd(pos_df):
    """Compute daily exposure for META-MEAN(BCD): average exposure of B, C, D."""
    exps = []
    for app in ["B", "C", "D"]:
        exps.append(compute_exposure_single(pos_df, app))
    df_exp = pd.DataFrame(exps).T.fillna(0)
    return df_exp.mean(axis=1)


def aggregate_hourly_to_daily_returns(hourly_rets):
    """Compound hourly returns to daily: (1+r1)*(1+r2)*...*(1+r9) - 1."""
    dates_norm = hourly_rets.index.normalize()
    daily_r = (1 + hourly_rets).groupby(dates_norm).prod() - 1
    return daily_r


def aggregate_hourly_to_daily_exposure(hourly_exp):
    """Mean hourly exposure per day."""
    dates_norm = hourly_exp.index.normalize()
    return hourly_exp.groupby(dates_norm).mean()


def load_cbr_rate(dates_index):
    """Load CBR key rate and align to given dates."""
    kr_path = os.path.join(BASE, "..", "moex_discovery/data/external/macro/raw/key_rate_cbr.parquet")
    kr = pd.read_parquet(kr_path)
    kr["date"] = pd.to_datetime(kr["date"])
    kr = kr.sort_values("date")
    daily_dates = pd.DataFrame({"date": dates_index})
    kr_daily = pd.merge_asof(daily_dates, kr[["date", "value"]], on="date", direction="backward")
    kr_daily = kr_daily.set_index("date")["value"]
    mmf_ann = (kr_daily - 1.5) / 100
    mmf_daily = mmf_ann / 365
    return mmf_daily


def scaling_analysis(strat_r_daily, exp_daily, mmf_daily, k_vals, bpy):
    """Compute scaling metrics for given multipliers."""
    rows = []
    for k in k_vals:
        scaled_r = k * strat_r_daily
        eff_exp = k * exp_daily
        free_cap = np.maximum(0, 1 - eff_exp)
        mmf_r = free_cap * mmf_daily
        total_r = scaled_r + mmf_r

        m = calc_metrics(total_r.values, bpy)
        avg_exp = np.mean(eff_exp) * 100
        max_exp = np.max(eff_exp) * 100
        avg_free = np.mean(free_cap) * 100
        pct_over = np.mean(eff_exp > 1.0) * 100

        rows.append({
            "k": k, "avg_exp": round(avg_exp, 1), "max_exp": round(max_exp, 1),
            "free_cap": round(avg_free, 1), "pct_over100": round(pct_over, 1),
            **m
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
print("Loading data...")
pos_path = "results/final/strategies/walkforward_v4/daily_positions.parquet"
pos_all = pd.read_parquet(pos_path)
pos_all["date"] = pd.to_datetime(pos_all["date"])
pos_all = pos_all[pos_all["test_year"].isin(BCD_YEARS)]

pos_daily = pos_all[pos_all["tf"] == "daily"].copy()
pos_hourly = pos_all[pos_all["tf"] == "hourly"].copy()
print(f"  Daily: {len(pos_daily):,} rows, Hourly: {len(pos_hourly):,} rows")

# ── Build all meta-portfolio returns ─────────────────────────────
META_NAMES = ["META-A", "META-B", "META-C", "META-D", "META-MEAN", "META-BEST"]

def build_all_metas(pos_df, comm, best_map, is_hourly=False):
    """Build all 6 meta-portfolios. Returns dict of name -> Series."""
    metas = {}
    for app_letter in ["A", "B", "C", "D"]:
        metas[f"META-{app_letter}"] = build_meta(pos_df, app_letter, comm)
    metas["META-MEAN"] = build_meta_mean_bcd(pos_df, comm)
    metas["META-BEST"] = build_meta_best(pos_df, best_map, comm)
    return metas

def build_all_exposures(pos_df, best_map):
    """Build daily exposure for all 6 meta-portfolios."""
    exps = {}
    for app_letter in ["A", "B", "C", "D"]:
        exps[f"META-{app_letter}"] = compute_exposure_single(pos_df, app_letter)
    exps["META-MEAN"] = compute_exposure_mean_bcd(pos_df)
    exps["META-BEST"] = compute_exposure_best(pos_df, best_map)
    return exps


print("\n=== DAILY (institutional 0.05%) ===")
d_rets_inst = build_all_metas(pos_daily, COMM_D_INST, BEST_MAP_DAILY)
d_exps = build_all_exposures(pos_daily, BEST_MAP_DAILY)

print("\n=== DAILY (retail 0.06%) ===")
d_rets_retl = build_all_metas(pos_daily, COMM_D_RETL, BEST_MAP_DAILY)

print("\n=== HOURLY (institutional 0.04%) ===")
h_rets_inst_raw = build_all_metas(pos_hourly, COMM_H_INST, BEST_MAP_HOURLY, is_hourly=True)
h_exps_raw = build_all_exposures(pos_hourly, BEST_MAP_HOURLY)

# Aggregate hourly → daily
h_rets_inst = {k: aggregate_hourly_to_daily_returns(v) for k, v in h_rets_inst_raw.items()}
h_exps = {k: aggregate_hourly_to_daily_exposure(v) for k, v in h_exps_raw.items()}

print("\n=== HOURLY (retail 0.05%) ===")
h_rets_retl_raw = build_all_metas(pos_hourly, COMM_H_RETL, BEST_MAP_HOURLY, is_hourly=True)
h_rets_retl = {k: aggregate_hourly_to_daily_returns(v) for k, v in h_rets_retl_raw.items()}

# Load CBR rate
common_dates = d_rets_inst["META-A"].index
mmf_daily = load_cbr_rate(common_dates)

# For hourly, align MMF to hourly daily dates
h_dates = h_rets_inst["META-A"].index
mmf_daily_h = load_cbr_rate(h_dates)

print(f"\nMMF rate: mean {mmf_daily.mean()*365*100:.1f}% p.a.")

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Exposure for all 6 MSP
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1: ЭКСПОЗИЦИЯ ВСЕХ МСП")
print("="*70)

print("\n### Таблица: Экспозиция МСП (daily + hourly)")
print(f"{'Портфель':<14} | {'Дн.ср.%':>8} | {'Дн.макс.%':>9} | {'Дн.своб.%':>9} | "
      f"{'Час.ср.%':>8} | {'Час.макс.%':>10} | {'Час.своб.%':>10}")
print("-"*90)

for name in META_NAMES:
    d_e = d_exps[name]
    h_e = h_exps[name]
    d_avg, d_max, d_free = d_e.mean()*100, d_e.max()*100, (1-d_e.mean())*100
    h_avg, h_max, h_free = h_e.mean()*100, h_e.max()*100, (1-h_e.mean())*100
    print(f"{name:<14} | {d_avg:>8.1f} | {d_max:>9.1f} | {d_free:>9.1f} | "
          f"{h_avg:>8.1f} | {h_max:>10.1f} | {h_free:>10.1f}")


# ══════════════════════════════════════════════════════════════════
# SECTION 2: ФДР (Capital Efficiency) for all MSP
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2: КАПИТАЛЬНАЯ ЭФФЕКТИВНОСТЬ (стратегия + ФДР)")
print("="*70)

def print_fdr_table(title, rets, exps, mmf, bpy):
    print(f"\n### {title}")
    print(f"{'Портфель':<14} | {'Стратег.%':>9} | {'+ФДР':>6} | {'Итого%':>7} | "
          f"{'Sharpe':>6} | {'MDD%':>6} | {'Calmar':>6}")
    print("-"*75)
    for name in META_NAMES:
        r = rets[name]
        e = exps[name]
        # Align indices
        idx = r.index.intersection(e.index).intersection(mmf.index)
        r_a = r.loc[idx]
        e_a = e.loc[idx]
        m_a = mmf.loc[idx]

        strat_m = calc_metrics(r_a.values, bpy)
        free_cap = np.maximum(0, 1 - e_a.values)
        total_r = r_a.values + free_cap * m_a.values
        total_m = calc_metrics(total_r, bpy)
        fdr_add = total_m["Return_pct"] - strat_m["Return_pct"]
        print(f"{name:<14} | {strat_m['Return_pct']:>9.2f} | {fdr_add:>+6.2f} | "
              f"{total_m['Return_pct']:>7.2f} | {total_m['Sharpe']:>6.2f} | "
              f"{total_m['MDD_pct']:>6.2f} | {total_m['Calmar']:>6.2f}")

print_fdr_table("Daily (инст. 0,05%)", d_rets_inst, d_exps, mmf_daily, BPY_D)
print_fdr_table("Hourly (инст. 0,04%) — агрегировано в дневные", h_rets_inst, h_exps, mmf_daily_h, BPY_D)


# ══════════════════════════════════════════════════════════════════
# SECTION 3: Масштабирование всех МСП
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3: МАСШТАБИРОВАНИЕ ВСЕХ МСП + ФДР")
print("="*70)

def print_scaling(title, rets, exps, mmf, bpy):
    for name in META_NAMES:
        r = rets[name]
        e = exps[name]
        idx = r.index.intersection(e.index).intersection(mmf.index)
        r_a, e_a, m_a = r.loc[idx], e.loc[idx], mmf.loc[idx]

        df_sc = scaling_analysis(r_a, e_a, m_a, K_VALS, bpy)
        print(f"\n### {name} {title}")
        print(f"{'k':>3} | {'Ср.эксп%':>8} | {'Макс%':>6} | {'Своб.%':>6} | "
              f"{'%>100':>5} | {'Ret%':>6} | {'Sharpe':>6} | {'MDD%':>6}")
        print("-"*65)
        for _, row in df_sc.iterrows():
            print(f"{int(row['k']):>2}x | {row['avg_exp']:>8.1f} | {row['max_exp']:>6.1f} | "
                  f"{row['free_cap']:>6.1f} | {row['pct_over100']:>5.1f} | "
                  f"{row['Return_pct']:>6.2f} | {row['Sharpe']:>6.2f} | {row['MDD_pct']:>6.2f}")

print_scaling("daily + ФДР (инст. 0,05%)", d_rets_inst, d_exps, mmf_daily, BPY_D)
print_scaling("hourly + ФДР (инст. 0,04%)", h_rets_inst, h_exps, mmf_daily_h, BPY_D)


# ══════════════════════════════════════════════════════════════════
# SECTION 4: Итоговая таблица при 3x daily / 5x hourly
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 4: ИТОГОВАЯ ТАБЛИЦА — 3x daily / 5x hourly + ФДР")
print("="*70)

def summary_at_k(rets, exps, mmf, bpy, k):
    """Compute Return%, Vol%, Sharpe, MDD% for all MSP at multiplier k."""
    rows = []
    for name in META_NAMES:
        r = rets[name]
        e = exps[name]
        idx = r.index.intersection(e.index).intersection(mmf.index)
        r_a, e_a, m_a = r.loc[idx], e.loc[idx], mmf.loc[idx]

        scaled_r = k * r_a.values
        free_cap = np.maximum(0, 1 - k * e_a.values)
        total_r = scaled_r + free_cap * m_a.values
        m = calc_metrics(total_r, bpy)
        rows.append({"name": name, **m})
    return pd.DataFrame(rows)


def print_summary_table(title, d_rets, h_rets, d_exps_, h_exps_, mmf_d, mmf_h, k_d=3, k_h=5):
    print(f"\n### {title}")
    print(f"{'':>16} | {'Дн.Ret%':>7} | {'Дн.Vol%':>7} | {'Дн.Sh':>6} | {'Дн.MDD%':>7} | "
          f"{'Час.Ret%':>8} | {'Час.Vol%':>8} | {'Час.Sh':>6} | {'Час.MDD%':>8}")
    print("-"*110)
    # IMOEX
    print(f"{'IMOEX':<16} | {-3.54:>7.2f} | {29.17:>7.2f} | {-0.12:>6.2f} | {-50.76:>7.2f} | "
          f"{-3.54:>8.2f} | {29.17:>8.2f} | {-0.12:>6.2f} | {-50.76:>8.2f}")

    d_sum = summary_at_k(d_rets, d_exps_, mmf_d, BPY_D, k_d)
    h_sum = summary_at_k(h_rets, h_exps_, mmf_h, BPY_D, k_h)

    for i, name in enumerate(META_NAMES):
        dr = d_sum.iloc[i]
        hr = h_sum.iloc[i]
        label = f"{name} + ФДР"
        print(f"{label:<16} | {dr['Return_pct']:>7.2f} | {dr['Vol_pct']:>7.2f} | "
              f"{dr['Sharpe']:>6.2f} | {dr['MDD_pct']:>7.2f} | "
              f"{hr['Return_pct']:>8.2f} | {hr['Vol_pct']:>8.2f} | "
              f"{hr['Sharpe']:>6.2f} | {hr['MDD_pct']:>8.2f}")

print_summary_table("Институциональные издержки (3x daily / 5x hourly + ФДР)",
                     d_rets_inst, h_rets_inst, d_exps, h_exps, mmf_daily, mmf_daily_h)
print_summary_table("Розничные издержки (3x daily / 5x hourly + ФДР)",
                     d_rets_retl, h_rets_retl, d_exps, h_exps, mmf_daily, mmf_daily_h)


# ══════════════════════════════════════════════════════════════════
# SECTION 5: Sharpe при двух уровнях комиссий для B и C
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 5: РОЗНИЧНЫЕ SHARPE для B и C (+ все остальные для проверки)")
print("="*70)

print(f"\n{'Портфель':<14} | {'Дн.0.05%':>8} | {'Дн.0.06%':>8} | {'Час.0.04%':>9} | {'Час.0.05%':>9}")
print("-"*65)

for name in META_NAMES:
    d_i = calc_metrics(d_rets_inst[name].values, BPY_D)
    d_r = calc_metrics(d_rets_retl[name].values, BPY_D)
    h_i = calc_metrics(h_rets_inst[name].values, BPY_D)
    h_r = calc_metrics(h_rets_retl[name].values, BPY_D)
    print(f"{name:<14} | {d_i['Sharpe']:>8.2f} | {d_r['Sharpe']:>8.2f} | "
          f"{h_i['Sharpe']:>9.2f} | {h_r['Sharpe']:>9.2f}")


# ══════════════════════════════════════════════════════════════════
# VERIFICATION
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("VERIFICATION: сравнение с существующими данными")
print("="*70)

print("\nОжидаемые значения:")
print("  BEST daily  3x+ФДР: Ret=27.94, Sh=3.34, MDD=-5.08")
print("  BEST hourly 5x+ФДР: Ret=25.07, Sh=4.87, MDD=-1.10")
print("  D daily  3x+ФДР: Ret=27.45, Sh=3.29, MDD=-5.20")
print("  D hourly 5x+ФДР: Ret=41.22, Sh=2.62, MDD=-7.74")

# Compute verification
for name, k_d, k_h in [("META-BEST", 3, 5), ("META-D", 3, 5)]:
    # Daily
    r = d_rets_inst[name]; e = d_exps[name]
    idx = r.index.intersection(e.index).intersection(mmf_daily.index)
    total_d = k_d * r.loc[idx].values + np.maximum(0, 1 - k_d * e.loc[idx].values) * mmf_daily.loc[idx].values
    md = calc_metrics(total_d, BPY_D)
    print(f"\n  {name} daily  {k_d}x+ФДР: Ret={md['Return_pct']}, Sh={md['Sharpe']}, MDD={md['MDD_pct']}")

    # Hourly
    r = h_rets_inst[name]; e = h_exps[name]
    idx = r.index.intersection(e.index).intersection(mmf_daily_h.index)
    total_h = k_h * r.loc[idx].values + np.maximum(0, 1 - k_h * e.loc[idx].values) * mmf_daily_h.loc[idx].values
    mh = calc_metrics(total_h, BPY_D)
    print(f"  {name} hourly {k_h}x+ФДР: Ret={mh['Return_pct']}, Sh={mh['Sharpe']}, MDD={mh['MDD_pct']}")

print("\nDone.")
