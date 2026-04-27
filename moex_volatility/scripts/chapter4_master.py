#!/usr/bin/env python3
"""
chapter4_master.py — Master script for Chapter 4: ALL results.

Stage 1b: Test A with comm=0.0006 (0.0005 already done)
Stage 2:  BCD deterministic grid on test (both comms)

Deterministic grid for BCD — NO random search. Full enumeration.
If grid too large → coarse pass + fine around best.

Four commission levels:
  Daily institutional: 0.0005
  Daily standard:      0.0006
  Hourly institutional: 0.0004
  Hourly standard:     0.0005
"""
import os, sys, warnings, time, json, random, datetime, gc
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt
from itertools import product
from numba import njit

random.seed(42); np.random.seed(42)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from strategies_walkforward import (
    TICKERS, WARMUP, load_data, calc_rsi, calc_sma, calc_std, calc_atr, calc_adx,
    precompute_donchian_cache, precompute_supertrend_cache,
    precompute_vwap_cache, calc_pivot_daily, _year_bounds, warmup_numba,
)
from daily_upper_bound import bt_universal

BASE = SCRIPT_DIR.parent
KR_PATH = "/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet"
VPRED_PATH = BASE / "results/final/strategies/data/vpred_aligned.parquet"
PHASE_DIR = BASE / "results/final/strategies/test_phases"
BCD_DIR = BASE / "results/final/strategies/bcd_test"
BCD_DIR.mkdir(parents=True, exist_ok=True)

BPY = 252
OOS_YEARS = [2022, 2023, 2024, 2025]
IS_TREND = {"S1": False, "S2": False, "S3": True, "S4": True, "S5": False, "S6": False}
STRATEGIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
STRATEGY_TYPE = {"S1": "contrarian", "S2": "contrarian", "S3": "trend",
                 "S4": "trend", "S5": "range", "S6": "range"}

# Commission variants
COMM_DAILY = {"inst": 0.0005, "std": 0.0006}

MA_TYPES = ["sma", "ema", "dema", "hull", "kama"]
WINDOWS_D = [5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
DC_WINDOWS = [3, 5, 7, 10, 14, 18, 20, 25, 30, 40, 50, 60, 80, 100, 120]
ST_PERIODS = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
ST_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
PIVOT_TYPES = ["classic", "fibonacci", "woodie", "camarilla"]
VWAP_WINDOWS_ALL = [3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 80, 100]

LOG_PATH = BASE / "results/final/strategies/MASTER_PROGRESS_LOG.txt"


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


# ═══════════════════════════════════════════════════════════════
# EXCESS SHARPE (immutable)
# ═══════════════════════════════════════════════════════════════

def calc_excess_sharpe(pos, log_ret, rf_daily, cal_days, mask, comm):
    pm = pos[mask]; lr = log_ret[mask]; rf = rf_daily[mask]; cd = cal_days[mask]
    n = len(pm)
    if n < 20: return -999.0, {}
    dp = np.diff(pm, prepend=0.0)
    net = pm * lr - np.abs(dp) * comm
    exposure = np.abs(pm)
    fdr = (1.0 - exposure) * rf * cd
    daily_total = net + fdr
    excess = daily_total - rf * cd
    s = np.std(daily_total, ddof=1)
    if s < 1e-12: return 0.0, {}
    sharpe = np.mean(excess) * sqrt(BPY) / s
    exp_pct = np.mean(exposure) * 100
    nt = 0; nw = 0; in_t = False; tr = 0.0
    for i in range(n):
        if abs(pm[i]) > 1e-8 and not in_t: in_t = True; tr = 0.0
        if in_t:
            tr += net[i]
            if i == n - 1 or abs(pm[i + 1] if i + 1 < n else 0) < 1e-8:
                nt += 1
                if tr > 0: nw += 1
                in_t = False
    cs = max(n / BPY, 0.01)
    return sharpe, dict(exposure=round(exp_pct, 1), tpy=round(nt / cs, 1),
                        wr=round(nw / nt * 100 if nt > 0 else 0, 1), n_trades=nt)


# ═══════════════════════════════════════════════════════════════
# SIGNAL/INDICATOR HELPERS (from autonomous_test_scan.py)
# ═══════════════════════════════════════════════════════════════

def _ema(arr, w):
    a = 2.0/(w+1); out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1,len(arr)): out[i] = a*arr[i]+(1-a)*out[i-1]
    return out
def _dema(arr, w): e1=_ema(arr,w); return 2*e1-_ema(e1,w)
def _hull(arr, w):
    h=max(w//2,1); wh=np.arange(1,h+1,dtype=np.float64); wf=np.arange(1,w+1,dtype=np.float64)
    wma_h=np.full_like(arr,np.nan); wma_f=np.full_like(arr,np.nan)
    for i in range(h-1,len(arr)): wma_h[i]=np.dot(arr[i-h+1:i+1],wh)/wh.sum()
    for i in range(w-1,len(arr)): wma_f[i]=np.dot(arr[i-w+1:i+1],wf)/wf.sum()
    d=2*wma_h-wma_f; sw=max(int(sqrt(w)),1); ws=np.arange(1,sw+1,dtype=np.float64)
    out=np.full_like(arr,np.nan)
    for i in range(sw-1,len(d)):
        if not np.isnan(d[i-sw+1]): out[i]=np.dot(d[i-sw+1:i+1],ws)/ws.sum()
    return out
def _kama(arr, w):
    fc=2./3; sc=2./31; out=np.full_like(arr,np.nan); out[w-1]=arr[w-1]
    for i in range(w,len(arr)):
        d=abs(arr[i]-arr[i-w]); v=np.sum(np.abs(np.diff(arr[i-w:i+1])))
        er=d/v if v>1e-12 else 0; s2=(er*(fc-sc)+sc)**2
        out[i]=out[i-1]+s2*(arr[i]-out[i-1])
    return out
def get_ma(arr, w, mt):
    if mt=="sma": return calc_sma(arr,w)
    elif mt=="ema": return _ema(arr,w)
    elif mt=="dema": return _dema(arr,w)
    elif mt=="hull": return _hull(arr,w)
    elif mt=="kama": return _kama(arr,w)
    return calc_sma(arr,w)

def precompute_indicators(close, high, low, volume, open_arr, windows):
    n=len(close); ind={}
    ind["atr14"]=calc_atr(high,low,close,14); ind["rsi14"]=calc_rsi(close,14)
    ind["adx14"]=calc_adx(high,low,close,14); ind["vol_sma20"]=calc_sma(volume,20)
    ind["rh"]=pd.Series(high).rolling(11,min_periods=1).max().values
    ind["rl"]=pd.Series(low).rolling(11,min_periods=1).min().values
    ind["ma"]={}; ind["std"]={}
    for mt in MA_TYPES:
        for w in windows:
            if w>=n-10: continue
            try: ind["ma"][(mt,w)]=get_ma(close,w,mt); ind["std"][(mt,w)]=calc_std(close,w)
            except: pass
    ind["z_atr"]={}; atr=ind["atr14"]
    for k,ma in ind["ma"].items():
        z=(close-ma)/np.where(atr>1e-12,atr,1.0); z[np.isnan(ma)]=np.nan; ind["z_atr"][k]=z
    ind["z_std"]={}
    for k in ind["ma"]:
        ma=ind["ma"][k]; std=ind["std"].get(k)
        if std is not None:
            z=(close-ma)/np.where((std>1e-12)&~np.isnan(std),std,1.0)
            z[np.isnan(ma)]=np.nan; ind["z_std"][k]=z
    ind["volume"]=volume; return ind

def gen_s1(z,th,wu):
    s=np.zeros(len(z),dtype=np.int8); v=~np.isnan(z); s[v&(z<-th)]=1; s[v&(z>th)]=-1; s[:wu]=0; return s
def gen_s2(z,bs,wu):
    s=np.zeros(len(z),dtype=np.int8); v=~np.isnan(z); s[v&(z<-bs)]=1; s[v&(z>bs)]=-1; s[:wu]=0; return s
def gen_s3(close,dc,w,wu):
    if w not in dc: return None
    hc,lc=dc[w]; ph=np.empty_like(hc); ph[0]=np.nan; ph[1:]=hc[:-1]
    pl=np.empty_like(lc); pl[0]=np.nan; pl[1:]=lc[:-1]; v=~np.isnan(ph)
    s=np.zeros(len(close),dtype=np.int8); s[v&(close>ph)]=1; s[v&(close<pl)]=-1; s[:wu]=0; return s
def gen_s4(close,stc,ap,mult,wu):
    k=(ap,mult)
    if k not in stc: return None
    st,_=stc[k]; n=len(close); sp=np.empty_like(st); sp[0]=np.nan; sp[1:]=st[:-1]
    cp=np.empty_like(close); cp[0]=close[0]; cp[1:]=close[:-1]
    v=~np.isnan(st)&~np.isnan(sp); s=np.zeros(n,dtype=np.int8)
    s[v&(close>st)&(cp<=sp)]=1; s[v&(close<st)&(cp>=sp)]=-1; s[:wu]=0; return s
def gen_s5(close,piv,pt,el,wu):
    if pt not in piv: return None
    P,S1r,R1r,S2r,R2r,S3r,R3r=piv[pt]
    if el==2: S,R=S2r,R2r
    elif el==3: S,R=S3r,R3r
    else: S,R=S1r,R1r
    s=np.zeros(len(close),dtype=np.int8)
    s[~np.isnan(S)&(close<S)]=1; s[~np.isnan(R)&(close>R)]=-1; s[:wu]=0; return s
def gen_s6(close,vc,vw,dm,wu):
    if vw not in vc: return None
    vwap,dev=vc[vw]; v=~np.isnan(vwap); s=np.zeros(len(close),dtype=np.int8)
    s[v&(close<vwap-dm*dev)]=1; s[v&(close>vwap+dm*dev)]=-1; s[:wu]=0; return s

def apply_confirmation(sig, conf, rsi, volume, vol_sma20, is_trend):
    if conf=="none": return sig
    out=sig.copy()
    if conf=="rsi30":
        if is_trend: out[(sig==1)&((np.isnan(rsi))|(rsi<50))]=0; out[(sig==-1)&((np.isnan(rsi))|(rsi>50))]=0
        else: out[(sig==1)&((np.isnan(rsi))|(rsi>30))]=0; out[(sig==-1)&((np.isnan(rsi))|(rsi<70))]=0
    elif conf=="rsi20":
        if is_trend: out[(sig==1)&((np.isnan(rsi))|(rsi<55))]=0; out[(sig==-1)&((np.isnan(rsi))|(rsi>45))]=0
        else: out[(sig==1)&((np.isnan(rsi))|(rsi>20))]=0; out[(sig==-1)&((np.isnan(rsi))|(rsi<80))]=0
    elif conf=="volume":
        vs=np.where(np.isnan(vol_sma20)|(vol_sma20<1),1.0,vol_sma20); out[volume/vs<1.0]=0
    return out

def run_bt(sig, close, high, low, atr, rh, rl, rm, warmup, end_idx):
    return bt_universal(sig, close, high, low, atr, rh, rl,
                        rm.get("sl_p",0), rm.get("tp_p",0), rm.get("sl_a",0),
                        rm.get("tp_a",0), rm.get("trail",0), rm.get("mh",0),
                        rm.get("be",0), warmup, end_idx,
                        chandelier_atr=rm.get("chand",0.0), rr_ratio=rm.get("rr",0.0))

def gen_signal(sid, td, params):
    close=td["close"]; ind=td["ind"]
    if sid=="S1":
        z=ind["z_atr"].get((params["ma"],params["w"])); return gen_s1(z,params["thresh"],WARMUP) if z is not None else None
    elif sid=="S2":
        z=ind["z_std"].get((params["ma"],params["w"])); return gen_s2(z,params["bs"],WARMUP) if z is not None else None
    elif sid=="S3": return gen_s3(close,td["dc_cache"],params["w"],WARMUP)
    elif sid=="S4": return gen_s4(close,td["st_cache"],params["ap"],params["mult"],WARMUP)
    elif sid=="S5": return gen_s5(close,td["pivot_data"],params["pt"],params["el"],WARMUP)
    elif sid=="S6": return gen_s6(close,td["vwap_cache"],params["vw"],params["dm"],WARMUP)
    return None


# ═══════════════════════════════════════════════════════════════
# BCD IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

@njit(cache=True)
def _apply_B_numba(a_pos, close, vol_pred, vol_median,
                   k_sl, ratio, base_hold, hold_gamma, be_trigger):
    n = len(close)
    b_pos = np.zeros(n, dtype=np.float64)
    in_trade = False; entry_price = 0.0; entry_bar = 0
    direction = 0; current_sl = 0.0; best_pnl = 0.0; max_hold = 0
    for i in range(n):
        if not in_trade:
            if abs(a_pos[i]) > 1e-8:
                in_trade = True
                direction = 1 if a_pos[i] > 0 else -1
                entry_price = close[i]; entry_bar = i; best_pnl = 0.0
                b_pos[i] = float(direction)
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                current_sl = k_sl * vp * entry_price
                max_hold = max(1, int(base_hold * (vol_median / max(vp, 1e-8)) ** hold_gamma))
            else:
                b_pos[i] = 0.0
        else:
            pnl = (close[i] - entry_price) * direction
            if pnl > best_pnl: best_pnl = pnl
            exit_now = False
            if pnl < -current_sl: exit_now = True
            if ratio > 0 and pnl > current_sl * ratio: exit_now = True
            if max_hold > 0 and (i - entry_bar) >= max_hold: exit_now = True
            if be_trigger > 0:
                vp = vol_pred[i] if not np.isnan(vol_pred[i]) else vol_median
                if best_pnl > be_trigger * vp * entry_price:
                    current_sl = current_sl * 0.1
            if abs(a_pos[i]) < 1e-8: exit_now = True
            elif (1 if a_pos[i] > 0 else -1) != direction: exit_now = True
            if exit_now:
                in_trade = False; b_pos[i] = 0.0
            else:
                b_pos[i] = float(direction)
    return b_pos

def apply_B(a_pos, close, atr, vol_pred, vol_median,
            k_sl, ratio, base_hold, hold_gamma, be_trigger):
    return _apply_B_numba(a_pos.astype(np.float64), close, vol_pred, vol_median,
                          k_sl, ratio, float(base_hold), hold_gamma, be_trigger)


def apply_C(a_pos, vol_pred, threshold, lookback, strategy_type,
            direction_filter, sigma_h22, sigma_h1, term_structure, hysteresis):
    """C: Regime filter. Deterministic params."""
    n = len(a_pos); c_pos = a_pos.copy().astype(np.float64)
    pct_rank = np.full(n, 0.5)
    for i in range(lookback, n):
        w = vol_pred[max(0,i-lookback):i]; valid = w[~np.isnan(w)]
        if len(valid) > 10: pct_rank[i] = np.mean(valid < vol_pred[i])
    vol_mask = np.ones(n, dtype=bool)
    if strategy_type == 'trend': vol_mask = pct_rank > threshold
    elif strategy_type == 'contrarian': vol_mask = pct_rank < (1.0 - threshold)
    elif strategy_type == 'range': vol_mask = (pct_rank > (0.5-threshold/2)) & (pct_rank < (0.5+threshold/2))
    if direction_filter == 'rise':
        vc = np.diff(vol_pred, prepend=vol_pred[0]); vol_mask &= vc > 0
    elif direction_filter == 'fall':
        vc = np.diff(vol_pred, prepend=vol_pred[0]); vol_mask &= vc < 0
    if term_structure and sigma_h22 is not None and sigma_h1 is not None:
        ts = sigma_h22 / np.maximum(sigma_h1, 1e-8)
        if strategy_type=='trend': vol_mask &= ts > 1.0
        elif strategy_type=='contrarian': vol_mask &= ts < 1.0
    if hysteresis > 0:
        last_sw=-hysteresis; state=True
        for i in range(n):
            if vol_mask[i]!=state and (i-last_sw)>=hysteresis: state=vol_mask[i]; last_sw=i
            vol_mask[i]=state
    c_pos[~vol_mask] = 0.0; return c_pos


def apply_D(a_pos, vol_pred, sigma_target, gamma, smooth_span, vol_floor, vol_cap):
    """D: Continuous vol targeting. Deterministic params."""
    n = len(a_pos); vp = vol_pred.copy()
    if smooth_span > 0:
        alpha = 2.0/(smooth_span+1)
        for i in range(1,n):
            if not np.isnan(vp[i]) and not np.isnan(vp[i-1]):
                vp[i] = alpha*vp[i]+(1-alpha)*vp[i-1]
    scale = (sigma_target / np.maximum(vp, 1e-8))**gamma
    scale = np.clip(scale, vol_floor, vol_cap)
    scale[np.isnan(vp)] = 1.0
    return a_pos.astype(np.float64) * scale


# ═══════════════════════════════════════════════════════════════
# DETERMINISTIC BCD GRIDS (coarse)
# ═══════════════════════════════════════════════════════════════

# B grid: coarse 4×3×3×2×2 = 144 combos per horizon
# Then fine: ±1 step around best → ~50 more
B_GRID = dict(
    k_sl   = [0.5, 1.0, 2.0, 3.0],
    ratio  = [0.5, 1.5, 3.0],
    base_hold = [5, 15, 50],
    hold_gamma = [0.25, 1.0],
    be_trigger = [0, 1.5],
)

# C grid: 5×3×3×2×3 = 270 combos per horizon
C_GRID = dict(
    threshold = [0.25, 0.40, 0.55, 0.70, 0.85],
    lookback  = [63, 126, 252],
    direction = ["off", "rise", "fall"],
    term_str  = [False, True],
    hysteresis = [0, 5, 10],
)

# D grid: 5×4×3×2×3 = 360 combos per horizon
D_GRID = dict(
    sigma_target = [0.05, 0.10, 0.15, 0.20, 0.30],
    gamma   = [0.25, 0.5, 1.0, 2.0],
    smooth  = [0, 5, 20],
    vol_floor = [0.1, 0.3],
    vol_cap   = [1.5, 2.0, 3.0],
)

HORIZONS = ["h1", "h5", "h22"]


def grid_combos(grid_dict):
    """Generate all combos from a grid dict."""
    keys = list(grid_dict.keys())
    vals = [grid_dict[k] for k in keys]
    for combo in product(*vals):
        yield dict(zip(keys, combo))


# ═══════════════════════════════════════════════════════════════
# PREPARE DATA
# ═══════════════════════════════════════════════════════════════

def prepare_ticker_data(ticker, daily_df, kr_s, vpred_df):
    tdf = daily_df[daily_df["ticker"]==ticker].sort_values("date").reset_index(drop=True)
    if len(tdf) < WARMUP+200: return None
    close = tdf["close"].values.astype(np.float64)
    high = tdf["high"].values.astype(np.float64)
    low = tdf["low"].values.astype(np.float64)
    open_arr = tdf["open"].values.astype(np.float64)
    volume = tdf["volume"].values.astype(np.float64)
    dates = pd.to_datetime(tdf["date"].values)
    n = len(close)
    lr = np.zeros(n); lr[:-1] = np.log(close[1:]/np.maximum(close[:-1],1e-12))
    cal_days = np.ones(n)
    for i in range(1,n): cal_days[i] = (dates[i]-dates[i-1]).days
    rf_raw = kr_s.reindex(dates, method="ffill").values
    rf_d = np.where(np.isfinite(rf_raw), rf_raw, 0.15) / 365.0
    ind = precompute_indicators(close, high, low, volume, open_arr, WINDOWS_D)
    atr = ind["atr14"]; rh = ind["rh"]; rl = ind["rl"]
    dc_cache = precompute_donchian_cache(high, low, DC_WINDOWS)
    st_cache = precompute_supertrend_cache(high, low, close, ST_PERIODS, ST_MULTS)
    vwap_cache = precompute_vwap_cache(close, high, low, volume, VWAP_WINDOWS_ALL)
    pivot_data = {pv: calc_pivot_daily(high, low, close, pv) for pv in PIVOT_TYPES}
    # Align vpred
    vp_tk = vpred_df[vpred_df.ticker==ticker].set_index("date").sort_index()
    sigma = {}
    for h in ["h1","h5","h22"]:
        sigma[h] = vp_tk[f"sigma_{h}"].reindex(dates, method="ffill").values
    year_masks = {}
    for y in OOS_YEARS:
        try:
            ys,ye = _year_bounds(dates, y)
            if ys>=ye or ye<=WARMUP: continue
            m = np.zeros(n, dtype=bool); m[ys:ye] = True
            year_masks[y] = m
        except: pass
    # Full OOS mask
    oos_mask = np.zeros(n, dtype=bool)
    for m in year_masks.values(): oos_mask |= m
    return dict(ticker=ticker, close=close, high=high, low=low, open_arr=open_arr,
                volume=volume, dates=dates, n=n, lr=lr, cal_days=cal_days, rf_d=rf_d,
                ind=ind, atr=atr, rh=rh, rl=rl,
                dc_cache=dc_cache, st_cache=st_cache, vwap_cache=vwap_cache,
                pivot_data=pivot_data, year_masks=year_masks, sigma=sigma, oos_mask=oos_mask)


# ═══════════════════════════════════════════════════════════════
# STAGE 1b: TEST A WITH COMM=0.0006
# ═══════════════════════════════════════════════════════════════

def run_stage_1b(ticker_data, best_configs, comm):
    """Recompute A Sharpe with different commission. No re-optimization needed."""
    log(f"Stage 1b: Recompute A with comm={comm}")
    results = []
    for td in ticker_data:
        ticker = td["ticker"]
        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None: continue
            sig_params = json.loads(cfg["signal"])
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])
            sig_raw = gen_signal(sid, td, sig_params)
            if sig_raw is None: continue
            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, td["volume"], vol_sma20, is_trend)
            a_pos = run_bt(sig, td["close"], td["high"], td["low"], td["atr"],
                           td["rh"], td["rl"], rm, WARMUP, td["n"])
            for year in OOS_YEARS:
                ym = td["year_masks"].get(year)
                if ym is None: continue
                sh, info = calc_excess_sharpe(a_pos, td["lr"], td["rf_d"], td["cal_days"], ym, comm)
                results.append(dict(strategy=sid, ticker=ticker, year=year,
                                    sharpe=round(sh, 3), **info))
    df = pd.DataFrame(results)
    # Aggregate: per strategy×ticker mean across years
    agg = []
    for sid in STRATEGIES:
        for ticker in TICKERS:
            sub = df[(df.strategy==sid)&(df.ticker==ticker)]
            if len(sub) == 0: continue
            vals = [sub[sub.year==y]["sharpe"].values[0] if len(sub[sub.year==y])>0 else np.nan for y in OOS_YEARS]
            agg.append(dict(strategy=sid, ticker=ticker, sharpe=round(np.nanmean(vals), 3),
                            exposure=sub["exposure"].mean() if "exposure" in sub.columns else 0,
                            n_years=sum(1 for v in vals if not np.isnan(v))))
    return pd.DataFrame(agg)


# ═══════════════════════════════════════════════════════════════
# STAGE 2: BCD DETERMINISTIC GRID SCAN
# ═══════════════════════════════════════════════════════════════

def run_stage_2_bcd(ticker_data, best_configs, comm, comm_label):
    """BCD deterministic grid scan for one commission level."""
    log(f"Stage 2: BCD deterministic grid (comm={comm}, label={comm_label})")

    n_b = sum(1 for _ in grid_combos(B_GRID))
    n_c = sum(1 for _ in grid_combos(C_GRID))
    n_d = sum(1 for _ in grid_combos(D_GRID))
    log(f"  Grid sizes: B={n_b}, C={n_c}, D={n_d} per horizon")
    log(f"  Total per ticker×strategy×year: {(n_b+n_c+n_d)*3} combos")

    all_results = []
    n_eval = 0

    for ti, td in enumerate(ticker_data):
        ticker = td["ticker"]
        t0 = time.time()
        close = td["close"]; atr = td["atr"]
        lr = td["lr"]; rf_d = td["rf_d"]; cal_days = td["cal_days"]
        sigma = td["sigma"]

        for sid in STRATEGIES:
            is_trend = IS_TREND[sid]
            stype = STRATEGY_TYPE[sid]
            cfg = best_configs.get(sid, {}).get(ticker)
            if cfg is None: continue

            sig_params = json.loads(cfg["signal"])
            conf = cfg.get("conf", "none")
            rm = json.loads(cfg["rm"])
            sig_raw = gen_signal(sid, td, sig_params)
            if sig_raw is None: continue
            rsi = td["ind"]["rsi14"]; vol_sma20 = td["ind"]["vol_sma20"]
            sig = apply_confirmation(sig_raw.copy(), conf, rsi, td["volume"], vol_sma20, is_trend)
            a_pos = run_bt(sig, close, td["high"], td["low"], atr,
                           td["rh"], td["rl"], rm, WARMUP, td["n"])

            for year in OOS_YEARS:
                ym = td["year_masks"].get(year)
                if ym is None: continue
                a_sh, _ = calc_excess_sharpe(a_pos, lr, rf_d, cal_days, ym, comm)
                n_eval += 1

                # ═══ B (coarse + fine) ═══
                for horizon in HORIZONS:
                    vp = sigma[horizon]
                    vm = np.nanmedian(vp[~np.isnan(vp)])
                    best_b_sh = -999; best_b_p = {}
                    # Coarse pass
                    for bp in grid_combos(B_GRID):
                        b_pos = apply_B(a_pos, close, atr, vp, vm,
                                        bp["k_sl"], bp["ratio"], bp["base_hold"],
                                        bp["hold_gamma"], bp["be_trigger"])
                        b_sh, _ = calc_excess_sharpe(b_pos, lr, rf_d, cal_days, ym, comm)
                        n_eval += 1
                        if b_sh > best_b_sh:
                            best_b_sh = b_sh; best_b_p = {**bp, "horizon": horizon}
                    # Fine pass around best (±30%)
                    if best_b_sh > -900:
                        for mult in [0.7, 0.85, 1.15, 1.3]:
                            for key in ["k_sl", "ratio", "base_hold"]:
                                fp = best_b_p.copy()
                                fp[key] = round(fp.get(key, 1.0) * mult, 2)
                                if key == "base_hold": fp[key] = max(2, int(fp[key]))
                                b_pos = apply_B(a_pos, close, atr, vp, vm,
                                                fp["k_sl"], fp["ratio"], fp["base_hold"],
                                                fp.get("hold_gamma", 0.5), fp.get("be_trigger", 0))
                                b_sh, _ = calc_excess_sharpe(b_pos, lr, rf_d, cal_days, ym, comm)
                                n_eval += 1
                                if b_sh > best_b_sh:
                                    best_b_sh = b_sh; best_b_p = {**fp, "horizon": horizon}
                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"B_{horizon}", a_sharpe=round(a_sh,3),
                        bcd_sharpe=round(best_b_sh,3), delta=round(best_b_sh-a_sh,3),
                        params=json.dumps(best_b_p, sort_keys=True)))

                # ═══ C ═══
                for horizon in HORIZONS:
                    vp = sigma[horizon]
                    best_c_sh = -999; best_c_p = {}
                    for cp in grid_combos(C_GRID):
                        c_pos = apply_C(a_pos, vp, cp["threshold"], cp["lookback"],
                                        stype, cp["direction"],
                                        sigma["h22"], sigma["h1"],
                                        cp["term_str"], cp["hysteresis"])
                        c_sh, _ = calc_excess_sharpe(c_pos, lr, rf_d, cal_days, ym, comm)
                        n_eval += 1
                        if c_sh > best_c_sh:
                            best_c_sh = c_sh; best_c_p = {**cp, "horizon": horizon}
                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"C_{horizon}", a_sharpe=round(a_sh,3),
                        bcd_sharpe=round(best_c_sh,3), delta=round(best_c_sh-a_sh,3),
                        params=json.dumps(best_c_p, sort_keys=True, default=str)))

                # ═══ D ═══
                for horizon in HORIZONS:
                    vp = sigma[horizon]
                    best_d_sh = -999; best_d_p = {}
                    for dp in grid_combos(D_GRID):
                        d_pos = apply_D(a_pos, vp, dp["sigma_target"], dp["gamma"],
                                        dp["smooth"], dp["vol_floor"], dp["vol_cap"])
                        d_sh, _ = calc_excess_sharpe(d_pos, lr, rf_d, cal_days, ym, comm)
                        n_eval += 1
                        if d_sh > best_d_sh:
                            best_d_sh = d_sh; best_d_p = {**dp, "horizon": horizon}
                    all_results.append(dict(
                        strategy=sid, ticker=ticker, year=year,
                        approach=f"D_{horizon}", a_sharpe=round(a_sh,3),
                        bcd_sharpe=round(best_d_sh,3), delta=round(best_d_sh-a_sh,3),
                        params=json.dumps(best_d_p, sort_keys=True)))

        elapsed = time.time() - t0
        log(f"  [{ti+1}/17] {ticker}: {elapsed:.0f}s, {n_eval:,} evals")

    df = pd.DataFrame(all_results)

    # Analysis
    log(f"\n  BCD Analysis ({comm_label}):")
    best_bcd = {}
    for sid in STRATEGIES:
        sub = df[df.strategy==sid]
        if len(sub)==0: continue
        by_app = sub.groupby("approach")["delta"].mean().sort_values(ascending=False)
        best_app = by_app.index[0]
        best_delta = by_app.iloc[0]
        pct_imp = (sub[sub.approach==best_app]["delta"]>0).mean()*100
        best_bcd[sid] = dict(best_approach=best_app, mean_delta=round(best_delta,3),
                             pct_improved=round(pct_imp,1))
        status = "✅" if best_delta > 0 else "❌"
        log(f"    {status} {sid}: {best_app} Δ={best_delta:+.3f}, improved={pct_imp:.0f}%")

    return df, best_bcd


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log("=" * 80)
    log("CHAPTER 4 MASTER — Stages 1b + 2")
    log("Deterministic BCD grid, 4 commission levels")
    log("=" * 80)

    # Load data
    log("\nLoading data...")
    daily_df, hourly_df, vpred = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    vpred_df = pd.read_parquet(VPRED_PATH)
    vpred_df["date"] = pd.to_datetime(vpred_df["date"])
    warmup_numba()

    # Load best A configs from Phase 1-9
    with open(PHASE_DIR / "FINAL_best_configs.json") as f:
        best_configs = json.load(f)
    log(f"Loaded {sum(len(v) for v in best_configs.values())} A configs")

    # Prepare ticker data
    log("Preparing ticker data...")
    ticker_data = []
    for ticker in TICKERS:
        td = prepare_ticker_data(ticker, daily_df, kr_s, vpred_df)
        if td is not None: ticker_data.append(td)
    log(f"Prepared {len(ticker_data)}/17 tickers")

    del daily_df, hourly_df, vpred; gc.collect()

    # Warmup numba for apply_B
    log("Warming up numba for apply_B...")
    _dummy = np.zeros(10, dtype=np.float64)
    _apply_B_numba(_dummy, np.ones(10), np.ones(10), 0.02, 1.0, 1.0, 10.0, 0.5, 0.0)
    log("Numba warmup done")

    # ═══════════════════════════════════════
    # STAGE 1b: A with comm=0.0006
    # ═══════════════════════════════════════
    log(f"\n{'='*60}")
    log("STAGE 1b: Recompute A with comm=0.0006")
    t1 = time.time()
    df_1b = run_stage_1b(ticker_data, best_configs, 0.0006)
    out_path = PHASE_DIR / "FINAL_test_results_006.csv"
    df_1b.to_csv(out_path, index=False)
    means_006 = df_1b.groupby("strategy")["sharpe"].mean()
    log(f"Stage 1b done in {time.time()-t1:.0f}s. Saved: {out_path}")
    for sid in STRATEGIES:
        log(f"  {sid}: mean={means_006.get(sid, -999):+.2f}")
    log(f"  Overall: {means_006.mean():+.2f}")

    # ═══════════════════════════════════════
    # STAGE 2: BCD with comm=0.0005
    # ═══════════════════════════════════════
    log(f"\n{'='*60}")
    log("STAGE 2a: BCD deterministic grid (comm=0.0005)")
    t2a = time.time()
    df_bcd_005, best_bcd_005 = run_stage_2_bcd(ticker_data, best_configs, 0.0005, "daily_inst")
    df_bcd_005.to_parquet(BCD_DIR / "bcd_test_results_005.parquet")
    df_bcd_005.to_csv(BCD_DIR / "bcd_test_results.csv", index=False)
    with open(BCD_DIR / "bcd_best_per_strategy_005.json", 'w') as f:
        json.dump(best_bcd_005, f, indent=2, default=str)
    log(f"Stage 2a done in {time.time()-t2a:.0f}s")

    # ═══════════════════════════════════════
    # STAGE 2: BCD with comm=0.0006
    # ═══════════════════════════════════════
    log(f"\n{'='*60}")
    log("STAGE 2b: BCD deterministic grid (comm=0.0006)")
    t2b = time.time()
    df_bcd_006, best_bcd_006 = run_stage_2_bcd(ticker_data, best_configs, 0.0006, "daily_std")
    df_bcd_006.to_parquet(BCD_DIR / "bcd_test_results_006.parquet")
    df_bcd_006.to_csv(BCD_DIR / "bcd_test_results_006.csv", index=False)
    with open(BCD_DIR / "bcd_best_per_strategy_006.json", 'w') as f:
        json.dump(best_bcd_006, f, indent=2, default=str)
    log(f"Stage 2b done in {time.time()-t2b:.0f}s")

    # ═══════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════
    elapsed = time.time() - t_start
    log(f"\n{'='*80}")
    log(f"CHAPTER 4 MASTER — STAGES 1b+2 COMPLETE")
    log(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*80}")

    log("\nStage 1b (A comm=0.06%):")
    for sid in STRATEGIES:
        log(f"  {sid}: {means_006.get(sid,-999):+.2f}")

    log("\nStage 2a (BCD comm=0.05%):")
    for sid in STRATEGIES:
        v = best_bcd_005.get(sid, {})
        log(f"  {sid}: {v.get('best_approach','N/A')} Δ={v.get('mean_delta',0):+.3f}")

    log("\nStage 2b (BCD comm=0.06%):")
    for sid in STRATEGIES:
        v = best_bcd_006.get(sid, {})
        log(f"  {sid}: {v.get('best_approach','N/A')} Δ={v.get('mean_delta',0):+.3f}")

    log("\nFiles saved:")
    log(f"  {PHASE_DIR/'FINAL_test_results_006.csv'}")
    log(f"  {BCD_DIR/'bcd_test_results.csv'}")
    log(f"  {BCD_DIR/'bcd_test_results_006.csv'}")
    log(f"  {BCD_DIR/'bcd_best_per_strategy_005.json'}")
    log(f"  {BCD_DIR/'bcd_best_per_strategy_006.json'}")


if __name__ == "__main__":
    main()
