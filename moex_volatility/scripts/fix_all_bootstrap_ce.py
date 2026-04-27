#!/usr/bin/env python3
"""
fix_all_bootstrap_ce.py — Full recalculation of bootstrap and CE after MSP-B fix.
Uses corrected MSP daily returns from perstrat_daily_returns/MSP_{A,B,C,D}_inst.csv.
Also computes MSP-A vs IMOEX.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

BASE = Path(__file__).resolve().parent.parent
DR = BASE / "results/final/strategies/msp/perstrat_daily_returns"
BPY = 252
N_BOOT = 10_000


def sharpe(xs):
    s = np.std(xs, ddof=1)
    return np.mean(xs) * sqrt(BPY) / s if s > 1e-12 else 0.0


def bootstrap_sharpe_diff(ret_test, ret_base, n_boot=N_BOOT):
    n = len(ret_test)
    obs = sharpe(ret_test) - sharpe(ret_base)
    np.random.seed(42)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        deltas[i] = sharpe(ret_test[idx]) - sharpe(ret_base[idx])
    p = np.mean(deltas <= 0)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return obs, p, ci_lo, ci_hi


def ce(r, gamma):
    w = np.maximum(1.0 + r, 1e-10)
    if gamma == 1:
        return np.exp(np.mean(np.log(w))) - 1
    u = np.mean(w ** (1 - gamma))
    return u ** (1 / (1 - gamma)) - 1


def ce_bootstrap(ret_test, ret_base, gamma, n_boot=N_BOOT):
    obs = (ce(ret_test, gamma) - ce(ret_base, gamma)) * BPY * 100
    np.random.seed(42)
    n = len(ret_test)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        deltas[i] = (ce(ret_test[idx], gamma) - ce(ret_base[idx], gamma)) * BPY * 100
    p = np.mean(deltas <= 0)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    return obs, p, ci_lo, ci_hi


def stars(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))


def load_msp(app, comm="inst"):
    df = pd.read_csv(DR / f"MSP_{app}_{comm}.csv", parse_dates=["date"])
    return df.set_index("date")


def msp_metrics(df, label):
    xs = df["excess_ret"].values
    tr = df["trading_ret"].values
    tot = df["total_ret"].values
    exp = df["exposure"].values if "exposure" in df.columns else np.full(len(df), np.nan)
    cum = np.cumsum(tot)
    mdd = (cum - np.maximum.accumulate(cum)).min() * 100
    print(f"  {label}: XS Sh={sharpe(xs):.2f}  Tr Sh={sharpe(tr):.2f}  "
          f"Ret={np.mean(tot)*BPY*100:.1f}%  Vol={np.std(tot,ddof=1)*sqrt(BPY)*100:.1f}%  "
          f"MDD={mdd:.1f}%  Exp={np.nanmean(exp)*100:.1f}%")
    # Per-year
    df2 = df.copy()
    df2["year"] = df2.index.year
    for yr in [2022, 2023, 2024, 2025]:
        ydf = df2[df2["year"] == yr]
        if len(ydf) < 20:
            continue
        yxs = sharpe(ydf["excess_ret"].values)
        yret = ydf["total_ret"].mean() * BPY * 100
        print(f"    {yr}: XS Sh={yxs:+.2f}  Ret={yret:+.1f}%")


def main():
    # Load all MSP daily returns
    a = load_msp("A")
    b = load_msp("B")
    c = load_msp("C")
    d = load_msp("D")

    print("=" * 70)
    print("MSP ALL6×EW, inst 0.05%, CORRECTED")
    print("=" * 70)
    for df, lbl in [(a, "A"), (b, "B"), (c, "C"), (d, "D")]:
        msp_metrics(df, lbl)
    print()

    # Align dates
    common = a.index
    for df in [b, c, d]:
        common = common.intersection(df.index)
    print(f"Common dates: {len(common)}")

    ra = a.loc[common, "excess_ret"].values
    rb = b.loc[common, "excess_ret"].values
    rc = c.loc[common, "excess_ret"].values
    rd = d.loc[common, "excess_ret"].values

    ta = a.loc[common, "total_ret"].values
    tb = b.loc[common, "total_ret"].values
    tc = c.loc[common, "total_ret"].values
    td = d.loc[common, "total_ret"].values

    # ── Bootstrap Sharpe differences ──────────────────────────────
    print("\n" + "=" * 70)
    print("Bootstrap ΔSharpe (10,000 replications)")
    print("=" * 70)
    for label, ret_test in [("B>A", rb), ("C>A", rc), ("D>A", rd)]:
        delta, p, ci_lo, ci_hi = bootstrap_sharpe_diff(ret_test, ra)
        print(f"  {label}: ΔSh={delta:+.4f}  p={p:.4f} {stars(p)}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # D vs C
    delta, p, ci_lo, ci_hi = bootstrap_sharpe_diff(rd, rc)
    print(f"  D>C: ΔSh={delta:+.4f}  p={p:.4f} {stars(p)}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── MSP-A vs IMOEX ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MSP-A vs IMOEX")
    print("=" * 70)
    features = pd.read_parquet(BASE / "data/features/features_all.parquet")
    imoex = features[["date", "close_IMOEX"]].drop_duplicates("date").sort_values("date")
    imoex["date"] = pd.to_datetime(imoex["date"])
    imoex = imoex.set_index("date")
    imoex["log_ret"] = np.log(imoex["close_IMOEX"] / imoex["close_IMOEX"].shift(1))
    imoex = imoex.dropna()

    # IMOEX on common dates with MSP-A
    common_imoex = a.index.intersection(imoex.index)
    imoex_ret = imoex.loc[common_imoex, "log_ret"].values
    msp_a_xs = a.loc[common_imoex, "excess_ret"].values

    # IMOEX excess = log_ret - rf_daily
    # Use key rate per year
    kr = {2022: 0.106, 2023: 0.099, 2024: 0.174, 2025: 0.210}
    imoex_dates = pd.Series(common_imoex)
    imoex_rf = imoex_dates.dt.year.map(kr).values / 252.0
    imoex_xs = imoex_ret - imoex_rf

    imoex_sh = sharpe(imoex_xs)
    msp_a_sh = sharpe(msp_a_xs)
    print(f"  IMOEX XS Sharpe: {imoex_sh:.2f}")
    print(f"  MSP-A XS Sharpe: {msp_a_sh:.2f}")

    delta, p, ci_lo, ci_hi = bootstrap_sharpe_diff(msp_a_xs, imoex_xs)
    print(f"  MSP-A > IMOEX: ΔSh={delta:+.4f}  p={p:.4f} {stars(p)}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    # ── CE Bootstrap ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CE Bootstrap (total_ret)")
    print("=" * 70)
    for label, ret_test in [("B>A", tb), ("C>A", tc), ("D>A", td)]:
        for gamma in [3, 5, 10]:
            ce_diff, p, ci_lo, ci_hi = ce_bootstrap(ret_test, ta, gamma)
            print(f"  {label} γ={gamma:2d}: ΔCE={ce_diff:+.2f}%/yr  p={p:.4f} {stars(p)}  CI=[{ci_lo:.2f}, {ci_hi:.2f}]")
        print()

    # D vs C CE
    print("  D vs C:")
    for gamma in [3, 5, 10]:
        ce_diff, p, ci_lo, ci_hi = ce_bootstrap(td, tc, gamma)
        print(f"  D>C γ={gamma:2d}: ΔCE={ce_diff:+.2f}%/yr  p={p:.4f} {stars(p)}  CI=[{ci_lo:.2f}, {ci_hi:.2f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
