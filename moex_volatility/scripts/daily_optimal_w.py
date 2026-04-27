#!/usr/bin/env python3
"""Daily: alpha/unit analysis + optimal w sweep for Excess Sharpe."""
import numpy as np, pandas as pd, warnings
from pathlib import Path
from math import sqrt
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
POS_PATH = BASE / "results/final/strategies/walkforward_v4/daily_a/daily_a_positions.parquet"
KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")

pos_all = pd.read_parquet(POS_PATH)
pos_all["date"] = pd.to_datetime(pos_all["date"])
kr = pd.read_parquet(KR_PATH)
kr["date"] = pd.to_datetime(kr["date"])
kr_s = kr.set_index("date")["key_rate"].sort_index()

BPY = 252
STRATEGIES = ["S1_MeanRev","S2_Bollinger","S3_Donchian","S4_Supertrend","S5_PivotPoints","S6_VWAP"]
N_TICKERS = 17
OOS_YEARS = [2022,2023,2024,2025]
W_GRID = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
CAP_GRID = [0.15, 0.25, 0.50, 0.75, 1.0]

def analyze(comm):
    comm_label = f"{comm*100:.2f}%"
    print(f"\n{'='*90}")
    print(f"COMM = {comm_label}")
    print(f"{'='*90}")

    results = []
    for strat in STRATEGIES:
        sub = pos_all[pos_all["strategy"] == strat].copy()
        tickers = sorted(sub["ticker"].unique())
        ticker_rets = {}; ticker_pos_dict = {}
        for tkr in tickers:
            g = sub[sub["ticker"] == tkr].sort_values("date")
            p = g["position"].values; gross = g["daily_gross_return"].values
            dpos = np.diff(p, prepend=0.0)
            net = gross - np.abs(dpos) * comm
            ticker_rets[tkr] = pd.Series(net, index=g["date"].values)
            ticker_pos_dict[tkr] = pd.Series(p, index=g["date"].values)

        ret_df = pd.DataFrame(ticker_rets).sort_index().fillna(0.0)
        pos_df = pd.DataFrame(ticker_pos_dict).sort_index().fillna(0.0)
        dates = ret_df.index
        signal = (pos_df.abs() > 1e-8).astype(float)
        pos_abs = pos_df.abs().replace(0, np.nan)
        rpu = (ret_df / pos_abs).fillna(0.0)
        T, N = signal.shape
        years = pd.to_datetime(dates).year
        oos_mask = np.isin(years, OOS_YEARS)
        rf_daily = kr_s.reindex(pd.to_datetime(dates), method="ffill").values / BPY
        rf_daily = np.where(np.isfinite(rf_daily), rf_daily, 0.0)[:T]

        # Alpha per unit at w=1/17
        w0 = np.full(N, 1.0/N_TICKERS)
        raw_exp = (signal.values * w0[None,:]).sum(axis=1)
        scale = np.where(raw_exp > 1, 1/np.maximum(raw_exp,1e-12), 1.0)
        eff_w = signal.values * w0[None,:] * scale[:,None]
        trad_ret = (eff_w * rpu.values).sum(axis=1)
        exposure = eff_w.sum(axis=1)
        avg_exp = np.mean(exposure[oos_mask])
        ann_trad = np.mean(trad_ret[oos_mask]) * BPY
        alpha_per_unit = ann_trad / max(avg_exp, 0.001)
        avg_rf = np.mean(rf_daily[oos_mask]) * BPY

        # Sweep w × cap
        best_sh = -999; best_w = 0; best_cap = 1.0; best_info = {}
        for w_val in W_GRID:
            for cap in CAP_GRID:
                w = np.full(N, w_val)
                raw_exp = (signal.values * w[None,:]).sum(axis=1)
                scale = np.where(raw_exp > cap, cap/np.maximum(raw_exp,1e-12), 1.0)
                eff_w = signal.values * w[None,:] * scale[:,None]
                trad = (eff_w * rpu.values).sum(axis=1)
                exp = eff_w.sum(axis=1)
                fdr = np.maximum(0, 1-exp) * rf_daily
                lev = np.maximum(0, exp-1) * (rf_daily*BPY+0.02)/BPY
                total = trad + fdr - lev

                oos_total = total[oos_mask]
                excess = oos_total - rf_daily[oos_mask]
                ann_excess = np.mean(excess) * BPY
                ann_vol = np.std(oos_total, ddof=1) * sqrt(BPY)
                sh = ann_excess / ann_vol if ann_vol > 1e-12 else 0

                ann_ret = np.mean(oos_total) * BPY
                ann_trad_w = np.mean(trad[oos_mask]) * BPY
                ann_fdr_w = np.mean(fdr[oos_mask]) * BPY
                cum = np.cumprod(1 + pd.Series(oos_total.copy()))
                mdd = ((cum / np.maximum.accumulate(cum)) - 1).min() * 100

                if sh > best_sh:
                    best_sh = sh; best_w = w_val; best_cap = cap
                    best_info = dict(ret=ann_ret*100, trad=ann_trad_w*100, fdr=ann_fdr_w*100,
                                     exp=np.mean(exp[oos_mask])*100, vol=ann_vol*100, mdd=mdd)

        alpha_flag = "ALPHA>RF" if alpha_per_unit > avg_rf else "alpha<rf"
        print(f"\n{strat:>16s}: alpha/unit={alpha_per_unit*100:+.1f}%  rf={avg_rf*100:.1f}%  [{alpha_flag}]")
        print(f"  Optimal w={best_w:.3f} cap={best_cap:.2f}: ExSharpe={best_sh:+.2f}")
        print(f"    Ret={best_info.get('ret',0):.1f}%  Trad={best_info.get('trad',0):.1f}%  "
              f"FDR={best_info.get('fdr',0):.1f}%  Exp={best_info.get('exp',0):.1f}%  "
              f"Vol={best_info.get('vol',0):.1f}%  MDD={best_info.get('mdd',0):.1f}%")
        results.append(dict(strategy=strat, comm=comm_label, alpha_per_unit=round(alpha_per_unit*100,1),
                            rf=round(avg_rf*100,1), best_w=best_w, best_cap=best_cap,
                            excess_sharpe=round(best_sh,4), **{k:round(v,2) for k,v in best_info.items()}))

    return results

all_res = []
all_res.extend(analyze(0.0005))
all_res.extend(analyze(0.0006))

df = pd.DataFrame(all_res)
print("\n" + "="*90)
print("SUMMARY TABLE")
print("="*90)
print(df[["strategy","comm","alpha_per_unit","rf","best_w","best_cap","excess_sharpe",
          "ret","trad","fdr","exp","vol","mdd"]].to_string(index=False))
