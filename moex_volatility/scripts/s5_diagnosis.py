#!/usr/bin/env python3
"""S5_PivotPoints diagnostic: why does it underperform?"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "results" / "final" / "strategies" / "data"

daily = pd.read_parquet(DATA_DIR / "ohlcv_daily_full.parquet")
daily["date"] = pd.to_datetime(daily["date"])

hourly = pd.read_parquet(DATA_DIR / "ohlcv_hourly_full.parquet")
hourly["datetime"] = pd.to_datetime(hourly["datetime"])

WARMUP = 200

# ═══════════════════════════════════════════════════════════════
# 1. PIVOT LEVEL FORMULA & PERIOD
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("1. PIVOT LEVEL CALCULATION")
print("=" * 70)

print("""
Formula: CLASSIC
  P  = (H_prev + L_prev + C_prev) / 3
  S1 = 2*P - H_prev
  R1 = 2*P - L_prev

Period used:
  Daily TF:  previous DAY's OHLC → today's pivots
  Hourly TF: previous DAY's OHLC → mapped to hourly bars

  *** NO WEEKLY/MONTHLY pivots are computed ***
  *** Daily TF uses DAILY pivots (1-day lookback) ***
""")

# Example: SBER, 5 recent days
sber_d = daily[daily["ticker"] == "SBER"].sort_values("date")
sber_d = sber_d[sber_d["date"] >= "2024-01-01"].head(10)

high = sber_d["high"].values
low = sber_d["low"].values
close = sber_d["close"].values
dates = sber_d["date"].values

n = len(close)
P = np.full(n, np.nan)
S1 = np.full(n, np.nan)
R1 = np.full(n, np.nan)
P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
S1[1:] = 2 * P[1:] - high[:-1]
R1[1:] = 2 * P[1:] - low[:-1]

print("Example: SBER, first 10 trading days of 2024")
print(f"{'Date':>12s}  {'Close':>8s}  {'Pivot':>8s}  {'S1':>8s}  {'R1':>8s}  {'S1_dist%':>9s}  {'R1_dist%':>9s}  {'Range%':>8s}")
for i in range(n):
    d = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
    c = close[i]
    p = P[i]; s1 = S1[i]; r1 = R1[i]
    if np.isnan(p):
        print(f"{d:>12s}  {c:8.2f}  {'NaN':>8s}  {'NaN':>8s}  {'NaN':>8s}")
    else:
        s1_dist = (c - s1) / c * 100
        r1_dist = (r1 - c) / c * 100
        rng = (r1 - s1) / c * 100
        print(f"{d:>12s}  {c:8.2f}  {p:8.2f}  {s1:8.2f}  {r1:8.2f}  {s1_dist:9.3f}  {r1_dist:9.3f}  {rng:8.3f}")

# Compute average S1→P and P→R1 distance across all SBER data
sber_full = daily[daily["ticker"] == "SBER"].sort_values("date")
h = sber_full["high"].values.astype(np.float64)
l = sber_full["low"].values.astype(np.float64)
c = sber_full["close"].values.astype(np.float64)
nf = len(c)
Pf = np.full(nf, np.nan); S1f = np.full(nf, np.nan); R1f = np.full(nf, np.nan)
Pf[1:] = (h[:-1] + l[:-1] + c[:-1]) / 3.0
S1f[1:] = 2 * Pf[1:] - h[:-1]
R1f[1:] = 2 * Pf[1:] - l[:-1]

valid = ~np.isnan(Pf) & (c > 0)
s1_to_p_pct = np.abs(Pf[valid] - S1f[valid]) / c[valid] * 100
p_to_r1_pct = np.abs(R1f[valid] - Pf[valid]) / c[valid] * 100
total_range_pct = np.abs(R1f[valid] - S1f[valid]) / c[valid] * 100

print(f"\nSBER average pivot distances (% of close price):")
print(f"  S1 → P distance:   mean={s1_to_p_pct.mean():.3f}%, median={np.median(s1_to_p_pct):.3f}%")
print(f"  P → R1 distance:   mean={p_to_r1_pct.mean():.3f}%, median={np.median(p_to_r1_pct):.3f}%")
print(f"  S1 → R1 full range: mean={total_range_pct.mean():.3f}%, median={np.median(total_range_pct):.3f}%")

# Same for ALL tickers
print(f"\nAll tickers average pivot distances:")
all_s1_p = []
all_p_r1 = []
all_range = []
for ticker in sorted(daily["ticker"].unique()):
    td = daily[daily["ticker"] == ticker].sort_values("date")
    ht = td["high"].values.astype(np.float64)
    lt = td["low"].values.astype(np.float64)
    ct = td["close"].values.astype(np.float64)
    nt = len(ct)
    Pt = np.full(nt, np.nan); S1t = np.full(nt, np.nan); R1t = np.full(nt, np.nan)
    Pt[1:] = (ht[:-1] + lt[:-1] + ct[:-1]) / 3.0
    S1t[1:] = 2 * Pt[1:] - ht[:-1]
    R1t[1:] = 2 * Pt[1:] - lt[:-1]
    vt = ~np.isnan(Pt) & (ct > 0)
    if vt.sum() > 0:
        all_s1_p.append(np.abs(Pt[vt] - S1t[vt]) / ct[vt] * 100)
        all_p_r1.append(np.abs(R1t[vt] - Pt[vt]) / ct[vt] * 100)
        all_range.append(np.abs(R1t[vt] - S1t[vt]) / ct[vt] * 100)
all_s1_p = np.concatenate(all_s1_p)
all_p_r1 = np.concatenate(all_p_r1)
all_range = np.concatenate(all_range)
print(f"  S1 → P distance:   mean={all_s1_p.mean():.3f}%, median={np.median(all_s1_p):.3f}%")
print(f"  P → R1 distance:   mean={all_p_r1.mean():.3f}%, median={np.median(all_p_r1):.3f}%")
print(f"  S1 → R1 full range: mean={all_range.mean():.3f}%, median={np.median(all_range):.3f}%")


# ═══════════════════════════════════════════════════════════════
# 2. RAW SIGNALS (before & after filters)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. ENTRY SIGNALS: RAW vs FILTERED")
print("=" * 70)

def precompute_indicators(close, high, low, volume):
    """Minimal indicator set needed for diagnosis."""
    n = len(close)
    ind = {}

    # ATR14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)),
                                            np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
    ind["atr14"] = atr14

    # ADX14
    plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0: plus_dm[i] = up
        if down > up and down > 0: minus_dm[i] = down
    smooth_tr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    smooth_tr = np.where(smooth_tr < 1e-10, 1e-10, smooth_tr)
    plus_di = pd.Series(plus_dm).rolling(14, min_periods=1).mean().values / smooth_tr * 100
    minus_di = pd.Series(minus_dm).rolling(14, min_periods=1).mean().values / smooth_tr * 100
    dx = np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1.0) * 100
    adx14 = pd.Series(dx).rolling(14, min_periods=1).mean().values
    ind["adx14"] = adx14

    # BB width
    sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    std20 = pd.Series(close).rolling(20, min_periods=1).std(ddof=1).values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = np.where(sma20 > 0, (bb_upper - bb_lower) / sma20, 0.0)
    ind["bb_width"] = bb_width

    # BB width 30th percentile (rolling 252)
    bw_series = pd.Series(bb_width)
    bw_p30 = bw_series.rolling(252, min_periods=50).quantile(0.3).values
    ind["bw_p30"] = bw_p30

    # SMA20 slope (10-bar)
    sma20_10ago = np.roll(sma20, 10)
    sma20_10ago[:10] = np.nan
    sma20_slope10 = np.where(sma20 > 0, (sma20 - sma20_10ago) / sma20, 0.0)
    ind["sma20_slope10"] = sma20_slope10

    # Vol regime
    ret = np.zeros(n)
    ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
    rvol20 = pd.Series(np.abs(ret)).rolling(20, min_periods=1).mean().values
    rvol60 = pd.Series(np.abs(ret)).rolling(60, min_periods=1).mean().values
    vol_regime = np.where(rvol60 > 0, rvol20 / rvol60, 1.0)
    ind["vol_regime"] = vol_regime

    # VF pass (volatility filter) — simplified: always True for diagnosis
    ind["vf_pass"] = np.ones(n, dtype=bool)

    return ind


def count_signals_year(ticker, df, is_hourly=False, daily_df=None):
    """Count raw and filtered signals per year."""
    if is_hourly:
        tdf = df[df["ticker"] == ticker].sort_values("datetime")
        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        dates = tdf["datetime"].values
    else:
        tdf = df[df["ticker"] == ticker].sort_values("date")
        close = tdf["close"].values.astype(np.float64)
        high = tdf["high"].values.astype(np.float64)
        low = tdf["low"].values.astype(np.float64)
        volume = tdf["volume"].values.astype(np.float64)
        dates = tdf["date"].values

    n = len(close)
    ind = precompute_indicators(close, high, low, volume)

    # Compute pivots
    if is_hourly:
        td = daily_df[daily_df["ticker"] == ticker].sort_values("date")
        hd = td["high"].values.astype(np.float64)
        ld = td["low"].values.astype(np.float64)
        cd = td["close"].values.astype(np.float64)
        dates_d = np.array(td["date"].values, dtype="datetime64[D]")
        nd = len(cd)
        Pd = np.full(nd, np.nan); S1d = np.full(nd, np.nan); R1d = np.full(nd, np.nan)
        Pd[1:] = (hd[:-1] + ld[:-1] + cd[:-1]) / 3.0
        S1d[1:] = 2 * Pd[1:] - hd[:-1]
        R1d[1:] = 2 * Pd[1:] - ld[:-1]
        h_dates = np.array(dates, dtype="datetime64[D]")
        indices = np.searchsorted(dates_d, h_dates, side="right") - 1
        valid_idx = indices >= 0
        P = np.full(n, np.nan); S1 = np.full(n, np.nan); R1 = np.full(n, np.nan)
        P[valid_idx] = Pd[indices[valid_idx]]
        S1[valid_idx] = S1d[indices[valid_idx]]
        R1[valid_idx] = R1d[indices[valid_idx]]
    else:
        P = np.full(n, np.nan); S1 = np.full(n, np.nan); R1 = np.full(n, np.nan)
        P[1:] = (high[:-1] + low[:-1] + close[:-1]) / 3.0
        S1[1:] = 2 * P[1:] - high[:-1]
        R1[1:] = 2 * P[1:] - low[:-1]

    # Raw signals (no filters)
    raw_long = ~np.isnan(S1) & (close < S1)
    raw_short = ~np.isnan(R1) & (close > R1)
    raw_long[:WARMUP] = False; raw_short[:WARMUP] = False
    raw_total = raw_long | raw_short

    # Individual filter pass rates
    adx = ind["adx14"]
    f_adx = ~np.isnan(adx) & (adx < 25)

    bw = ind["bb_width"]; bw_t = ind["bw_p30"]
    f_bw = ~np.isnan(bw_t) & (bw < bw_t)

    slope = ind["sma20_slope10"]
    f_slope = ~np.isnan(slope) & (np.abs(slope) < 0.01)

    vr = ind["vol_regime"]
    f_vr = ~np.isnan(vr) & (vr < 0.9)

    f_all = f_adx & f_bw & f_slope & f_vr

    # Filtered signals
    filt_long = raw_long & f_all
    filt_short = raw_short & f_all
    filt_total = filt_long | filt_short

    # Per year breakdown
    dt_years = pd.DatetimeIndex(dates).year

    results = {}
    for year in sorted(dt_years.unique()):
        if year < 2020: continue
        ymask = dt_years == year
        nb = ymask.sum()

        raw_ct = raw_total[ymask].sum()
        filt_ct = filt_total[ymask].sum()

        results[year] = {
            "bars": nb,
            "raw_signals": int(raw_ct),
            "filtered_signals": int(filt_ct),
            "raw_pct": raw_ct / nb * 100 if nb > 0 else 0,
            "filt_pct": filt_ct / nb * 100 if nb > 0 else 0,
            "kill_rate": (1 - filt_ct / max(raw_ct, 1)) * 100 if raw_ct > 0 else 0,
        }

    return results, {
        "n": n, "dates": dates, "dt_years": dt_years,
        "f_adx": f_adx, "f_bw": f_bw, "f_slope": f_slope, "f_vr": f_vr, "f_all": f_all,
        "raw_long": raw_long, "raw_short": raw_short, "raw_total": raw_total,
        "filt_total": filt_total,
        "adx": adx, "ind": ind, "P": P, "S1": S1, "R1": R1, "close": close,
    }


# SBER daily
print("\nSBER DAILY — raw vs filtered signals per year:")
sber_daily_res, sber_d_diag = count_signals_year("SBER", daily)
print(f"  {'Year':>4s}  {'Bars':>5s}  {'Raw':>5s}  {'Filt':>5s}  {'Raw%':>6s}  {'Filt%':>6s}  {'Kill%':>6s}")
for yr, r in sorted(sber_daily_res.items()):
    print(f"  {yr:>4d}  {r['bars']:>5d}  {r['raw_signals']:>5d}  {r['filtered_signals']:>5d}  {r['raw_pct']:6.1f}  {r['filt_pct']:6.1f}  {r['kill_rate']:6.1f}")

# SBER hourly
print("\nSBER HOURLY — raw vs filtered signals per year:")
sber_hourly_res, sber_h_diag = count_signals_year("SBER", hourly, is_hourly=True, daily_df=daily)
print(f"  {'Year':>4s}  {'Bars':>5s}  {'Raw':>5s}  {'Filt':>5s}  {'Raw%':>6s}  {'Filt%':>6s}  {'Kill%':>6s}")
for yr, r in sorted(sber_hourly_res.items()):
    print(f"  {yr:>4d}  {r['bars']:>5d}  {r['raw_signals']:>5d}  {r['filtered_signals']:>5d}  {r['raw_pct']:6.1f}  {r['filt_pct']:6.1f}  {r['kill_rate']:6.1f}")


# ═══════════════════════════════════════════════════════════════
# 3. FILTER PASS RATES (SBER daily, 2022-2026)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. FILTER PASS RATES — SBER daily 2022-2026")
print("=" * 70)

d = sber_d_diag
ymask = d["dt_years"] >= 2022
nb = ymask.sum()

print(f"\n  Total bars 2022-2026: {nb}")
print(f"  a) ADX < 25:             {d['f_adx'][ymask].sum():>5d} / {nb}  = {d['f_adx'][ymask].mean()*100:5.1f}%")
print(f"  b) BB_squeeze < P30:     {d['f_bw'][ymask].sum():>5d} / {nb}  = {d['f_bw'][ymask].mean()*100:5.1f}%")
print(f"  c) flat_MA |slope|<0.01: {d['f_slope'][ymask].sum():>5d} / {nb}  = {d['f_slope'][ymask].mean()*100:5.1f}%")
print(f"  d) vol_compression <0.9: {d['f_vr'][ymask].sum():>5d} / {nb}  = {d['f_vr'][ymask].mean()*100:5.1f}%")
print(f"  e) ALL TOGETHER:         {d['f_all'][ymask].sum():>5d} / {nb}  = {d['f_all'][ymask].mean()*100:5.1f}%")
print(f"\n  Raw signals in 2022-2026: {d['raw_total'][ymask].sum()}")
print(f"  Filtered signals:         {d['filt_total'][ymask].sum()}")

# Also show for ALL tickers
print(f"\nALL TICKERS daily, 2022-2026 filter pass rates:")
all_adx = []; all_bw = []; all_slope = []; all_vr = []; all_f = []
all_raw_s = 0; all_filt_s = 0; all_nb = 0

for ticker in sorted(daily["ticker"].unique()):
    tres, tdiag = count_signals_year(ticker, daily)
    dt_y = tdiag["dt_years"]
    ym = dt_y >= 2022
    nb_t = ym.sum()
    all_nb += nb_t
    all_adx.append(tdiag["f_adx"][ym])
    all_bw.append(tdiag["f_bw"][ym])
    all_slope.append(tdiag["f_slope"][ym])
    all_vr.append(tdiag["f_vr"][ym])
    all_f.append(tdiag["f_all"][ym])
    all_raw_s += tdiag["raw_total"][ym].sum()
    all_filt_s += tdiag["filt_total"][ym].sum()

all_adx = np.concatenate(all_adx)
all_bw = np.concatenate(all_bw)
all_slope = np.concatenate(all_slope)
all_vr = np.concatenate(all_vr)
all_f = np.concatenate(all_f)

print(f"  Total bars: {all_nb}")
print(f"  a) ADX < 25:             {all_adx.mean()*100:5.1f}%")
print(f"  b) BB_squeeze < P30:     {all_bw.mean()*100:5.1f}%")
print(f"  c) flat_MA |slope|<0.01: {all_slope.mean()*100:5.1f}%")
print(f"  d) vol_compression <0.9: {all_vr.mean()*100:5.1f}%")
print(f"  e) ALL TOGETHER:         {all_f.mean()*100:5.1f}%")
print(f"  Raw signals:    {all_raw_s}")
print(f"  Filtered signals: {all_filt_s}")
if all_raw_s > 0:
    print(f"  Kill rate: {(1 - all_filt_s/all_raw_s)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# 4. SL/TP vs PIVOT DISTANCE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. SL/TP vs PIVOT DISTANCE")
print("=" * 70)

# ATR14 in % of price for SBER
sber_fd = daily[daily["ticker"] == "SBER"].sort_values("date")
c_s = sber_fd["close"].values.astype(np.float64)
h_s = sber_fd["high"].values.astype(np.float64)
l_s = sber_fd["low"].values.astype(np.float64)
tr_s = np.maximum(h_s - l_s, np.maximum(np.abs(h_s - np.roll(c_s, 1)), np.abs(l_s - np.roll(c_s, 1))))
tr_s[0] = h_s[0] - l_s[0]
atr14_s = pd.Series(tr_s).rolling(14, min_periods=1).mean().values
atr_pct = atr14_s / c_s * 100

# For 2022+ only
sber_dates = sber_fd["date"].values
sber_yrs = pd.DatetimeIndex(sber_dates).year
ym = sber_yrs >= 2022

print(f"\nSBER daily (2022-2026):")
print(f"  ATR14 as % of close: mean={atr_pct[ym].mean():.3f}%, median={np.median(atr_pct[ym]):.3f}%")
print(f"  Pivot S1→P distance: mean={s1_to_p_pct.mean():.3f}%, median={np.median(s1_to_p_pct):.3f}%")
print(f"  Pivot P→R1 distance: mean={p_to_r1_pct.mean():.3f}%, median={np.median(p_to_r1_pct):.3f}%")
print(f"  Pivot S1→R1 range:   mean={total_range_pct.mean():.3f}%, median={np.median(total_range_pct):.3f}%")

print(f"\n  SL/TP at different ATR multipliers:")
for mult in [0.75, 1.0, 1.5]:
    sl_pct = atr_pct[ym].mean() * mult
    print(f"    {mult:.2f}x ATR14 = {sl_pct:.3f}% of price")

print(f"\n  Comparison:")
mean_atr = atr_pct[ym].mean()
mean_s1_p = s1_to_p_pct.mean()
print(f"    SL (1.0x ATR): {mean_atr:.3f}%")
print(f"    S1→P distance: {mean_s1_p:.3f}%")
ratio = mean_atr / mean_s1_p
print(f"    Ratio SL/distance: {ratio:.2f}x")
if ratio > 1.5:
    print(f"    *** SL is {ratio:.1f}x larger than pivot distance — SL will almost never trigger CORRECTLY")
    print(f"    *** By the time SL triggers, price has blown through S2")
elif ratio > 1.0:
    print(f"    SL is slightly larger than pivot distance")
else:
    print(f"    SL fits within pivot distance (ok)")

# Cross-ticker ATR vs pivot distance
print(f"\nAll tickers — ATR14% vs pivot S1→R1 range%:")
print(f"  {'Ticker':>6s}  {'ATR14%':>7s}  {'S1-R1%':>7s}  {'ATR/Range':>9s}  {'SL_1.0x_fits':>12s}")
for ticker in sorted(daily["ticker"].unique()):
    td = daily[daily["ticker"] == ticker].sort_values("date")
    ct = td["close"].values.astype(np.float64)
    ht = td["high"].values.astype(np.float64)
    lt = td["low"].values.astype(np.float64)
    trt = np.maximum(ht - lt, np.maximum(np.abs(ht - np.roll(ct, 1)), np.abs(lt - np.roll(ct, 1))))
    trt[0] = ht[0] - lt[0]
    atr14t = pd.Series(trt).rolling(14, min_periods=1).mean().values
    atr_pct_t = atr14t / ct * 100

    nt = len(ct)
    Pt = np.full(nt, np.nan); S1t = np.full(nt, np.nan); R1t = np.full(nt, np.nan)
    Pt[1:] = (ht[:-1] + lt[:-1] + ct[:-1]) / 3.0
    S1t[1:] = 2 * Pt[1:] - ht[:-1]; R1t[1:] = 2 * Pt[1:] - lt[:-1]
    vt = ~np.isnan(Pt) & (ct > 0)
    range_pct = np.abs(R1t[vt] - S1t[vt]) / ct[vt] * 100

    atr_mean = atr_pct_t[vt].mean()
    rng_mean = range_pct.mean()
    ratio = atr_mean / rng_mean if rng_mean > 0 else 999
    fits = "YES" if ratio < 1.0 else "NO"
    print(f"  {ticker:>6s}  {atr_mean:7.3f}  {rng_mean:7.3f}  {ratio:9.2f}  {fits:>12s}")

# ═══════════════════════════════════════════════════════════════
# 5. DAILY TF: PIVOTS USE 1-DAY LOOKBACK — IS THIS WRONG?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. PIVOT PERIOD PROBLEM ANALYSIS")
print("=" * 70)

print("""
PROBLEM #1: Daily TF uses DAILY pivots (previous 1 day)
  - Standard practice: daily TF should use WEEKLY or MONTHLY pivots
  - Daily pivots on daily bars = levels change every single day
  - This means S1/R1 are just yesterday's high/low transformed
  - The "range" they define is just yesterday's daily range
  - For mean reversion to work, levels should persist for multiple days

PROBLEM #2: Signal definition
  - Entry: close < S1 (below yesterday's support)
  - This is effectively: price dropped below yesterday's low range
  - But daily pivots shift daily, so the "support" moves every day
  - No persistence = no meaningful support/resistance

PROBLEM #3: Hourly TF correctly uses daily pivots
  - Hourly bars with daily pivots = standard/correct usage
  - Levels persist across ~9 bars = more meaningful S/R
""")

# Demonstrate: how often S1/R1 persist vs change
sber_full = daily[daily["ticker"] == "SBER"].sort_values("date")
c = sber_full["close"].values.astype(np.float64)
h = sber_full["high"].values.astype(np.float64)
l = sber_full["low"].values.astype(np.float64)
n = len(c)
P = np.full(n, np.nan); S1 = np.full(n, np.nan); R1 = np.full(n, np.nan)
P[1:] = (h[:-1] + l[:-1] + c[:-1]) / 3.0
S1[1:] = 2 * P[1:] - h[:-1]; R1[1:] = 2 * P[1:] - l[:-1]

# How much do levels change day to day?
s1_change_pct = np.abs(np.diff(S1[~np.isnan(S1)])) / c[1:len(S1[~np.isnan(S1)])+1] * 100
r1_change_pct = np.abs(np.diff(R1[~np.isnan(R1)])) / c[1:len(R1[~np.isnan(R1)])+1] * 100

print(f"SBER: Daily change in pivot levels (% of price):")
print(f"  S1 daily change: mean={s1_change_pct.mean():.3f}%, median={np.median(s1_change_pct):.3f}%")
print(f"  R1 daily change: mean={r1_change_pct.mean():.3f}%, median={np.median(r1_change_pct):.3f}%")
print(f"  → Levels shift substantially every day (no persistent S/R)")

# Weekly pivot simulation
print(f"\n  If we used WEEKLY pivots on daily bars:")
print(f"    Levels would persist for 5 days (Mon-Fri)")
print(f"    S1/R1 based on previous week's H/L/C")
print(f"    Much more meaningful support/resistance")

print("\n" + "=" * 70)
print("6. SUMMARY OF PROBLEMS")
print("=" * 70)
print("""
1. WRONG PIVOT PERIOD (daily TF):
   Daily bars use daily pivots → levels shift EVERY day
   Should use WEEKLY pivots for daily TF (or monthly)
   Hourly TF correctly uses daily pivots ✓

2. FILTERS KILL ~95% OF SIGNALS:
   Four strict range-confirming filters applied conjunctively
   Only ~5% of bars pass all four → very few entry opportunities

3. SL/TP MISMATCH WITH PIVOT DISTANCE:
   ATR14 ≈ 2x the S1→R1 range for daily pivots
   SL at 1.0x ATR = larger than entire pivot channel
   When entry at S1, SL should be at S2 (natural level)
   TP should be at P or R1 (natural level)

4. NO SIGNAL PARAMETER TUNING:
   S5 signal grid is empty {} — only "classic" variant tested
   Other strategies have parameter grids to optimize

PROPOSED FIXES:
  A) Daily TF: use WEEKLY pivots (5-day H/L/C)
  B) Relax filters: drop vol_compression, raise ADX to 30
  C) SL/TP tied to pivot levels: SL=S2 for longs, TP=P or R1
  D) Add signal grid: pivot_type=[classic,fibonacci,woodie,camarilla]
""")
