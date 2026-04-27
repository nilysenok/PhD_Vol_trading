#!/usr/bin/env python3
"""
economic_value.py — Section 4.4: Economic value of volatility forecasts.

Computes: utility Δ(γ), min capital, decomposition, sensitivity, segmentation.
Uses MSP-A and MSP-MEAN at optimal k (no leverage).
"""
import os, sys, warnings, time, random
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["NUMBA_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from math import sqrt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msp_correct import (
    process_tf, build_portfolio, compute_correct_sharpe,
    SIDS, SNAMES, SID_MAP, IS_TREND,
)
from msp_portfolio_opt import search_portfolio_bcd
from strategies_walkforward import load_data, warmup_numba

KR_PATH = Path("/Users/nilysenok/Desktop/PhD/results/final/strategies/pairs/data/cbr_key_rate.parquet")
BASE = Path(__file__).resolve().parent.parent
BPY = 252; N_BOOT = 10000
random.seed(42); np.random.seed(42)
COST_YEAR1 = 3_000_000  # rubles
COST_ANNUAL = 1_800_000


def scale_portfolio(base_daily, rf_daily, avg_exp, k, rf_ann=0.21):
    """Scale portfolio returns with FDR and leverage cost adjustments."""
    # Approximate: at k=1, base_daily includes trading + FDR
    # FDR_base ≈ (1 - avg_exp) * rf_ann / 252
    fdr_base = (1 - avg_exp) * rf_ann / 252
    trading = base_daily - fdr_base  # strip FDR to get pure trading

    scaled_exp = avg_exp * k
    free = max(0, 1.0 - scaled_exp)
    lev = max(0, scaled_exp - 1.0)

    fdr_new = free * rf_ann / 252
    lev_cost = lev * (rf_ann + 0.02) / 252

    return trading * k + fdr_new - lev_cost


def main():
    t0 = time.time()
    print("=" * 120)
    print("ECONOMIC VALUE OF VOLATILITY FORECASTS — Section 4.4")
    print("=" * 120)

    daily_df, hourly_df, _ = load_data()
    kr = pd.read_parquet(KR_PATH); kr["date"] = pd.to_datetime(kr["date"])
    kr_s = kr.set_index("date")["key_rate"].sort_index()
    warmup_numba()

    d_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/daily_a/upper_bound_params.csv")
    h_ub = pd.read_csv(BASE / "results/final/strategies/walkforward_v4/hourly_a/hourly_upper_bound.csv")

    # Average exposures (from previous runs)
    avg_exps = {"DAILY": {"A": 0.528, "C": 0.453},
                "HOURLY": {"A": 0.131, "C": 0.112}}

    for tf_label, ub_df, data_df, is_hourly in [
        ("DAILY", d_ub, daily_df, False),
        ("HOURLY", h_ub, hourly_df, True),
    ]:
        print(f"\n{'='*120}")
        print(f"  {tf_label}")
        print(f"{'='*120}")

        sd, rf_d = process_tf(ub_df, data_df, kr_s, is_hourly,
                               daily_df if is_hourly else None)

        # Build A and get MEAN via portfolio-level BCD
        port_a, rf_a = build_portfolio(sd, "A", rf_d)
        if port_a is None: print("  No data"); continue
        n = len(port_a)

        # Search B/C/D at portfolio level (same as msp_portfolio_opt)
        print("  Searching portfolio B/C/D...")
        (sh_b, rets_b), (sh_c, rets_c), (sh_d, rets_d) = \
            search_portfolio_bcd(port_a, rf_a, 5000)

        ml = min(len(rets_b), len(rets_c), len(rets_d))
        rets_mean = (rets_b[:ml] + rets_c[:ml] + rets_d[:ml]) / 3
        rf_mean = rf_a[:ml]
        port_a_aligned = port_a[:ml]

        avg_exp_a = avg_exps[tf_label]["A"]
        avg_exp_c = avg_exps[tf_label]["C"]
        avg_exp_mean = (avg_exp_a + avg_exp_c) / 2  # approximate

        # Optimal k: max k such that avg_exp * k < 0.85
        k_opt_a = min(int(0.85 / avg_exp_a), 8)
        k_opt_mean = min(int(0.85 / avg_exp_mean), 8)

        print(f"\n  ── 1. MSP CHARACTERISTICS AT OPTIMAL k ──")
        print(f"  k_opt: A={k_opt_a}, MEAN={k_opt_mean}")

        # Compute scaled returns
        scaled_a = scale_portfolio(port_a_aligned, rf_mean, avg_exp_a, k_opt_a)
        scaled_mean = scale_portfolio(rets_mean, rf_mean, avg_exp_mean, k_opt_mean)

        # Metrics
        sh_a, info_a = compute_correct_sharpe(scaled_a, rf_mean)
        sh_m, info_m = compute_correct_sharpe(scaled_mean, rf_mean)

        er_a = info_a["ann_ret"] / 100  # decimal
        er_m = info_m["ann_ret"] / 100
        vol_a = info_a["vol"] / 100
        vol_m = info_m["vol"] / 100
        var_a = vol_a ** 2
        var_m = vol_m ** 2

        print(f"\n  {'Config':>25s} {'E(r),%':>8s} {'σ,%':>7s} {'σ²,%²':>8s} {'Sharpe':>8s} {'k':>3s}")
        print(f"  {tf_label+' MSP-A':>25s} {er_a*100:8.2f} {vol_a*100:7.2f} {var_a*100:8.4f} {sh_a:+8.2f} {k_opt_a:3d}")
        print(f"  {tf_label+' MSP-MEAN':>25s} {er_m*100:8.2f} {vol_m*100:7.2f} {var_m*100:8.4f} {sh_m:+8.2f} {k_opt_mean:3d}")

        # ── 2. Economic value Δ(γ) ──
        print(f"\n  ── 2. ECONOMIC VALUE Δ(γ) ──")
        delta_er = er_m - er_a  # excess return component
        delta_var = var_m - var_a  # variance component

        print(f"  ΔE(r) = {delta_er*100:+.2f} п.п.")
        print(f"  Δσ² = {delta_var*100:+.4f} п.п.²")

        print(f"\n  {'γ':>4s} {'ΔE(r),п.п.':>11s} {'-(γ/2)Δσ²':>10s} {'Δ,п.п.':>8s} "
              f"{'Доля риска%':>11s} {'Мин.кап.,млн':>13s} {'Мин.кап.2г,млн':>15s}")

        for gamma in [3, 5, 10]:
            risk_comp = -(gamma / 2) * delta_var
            delta = delta_er + risk_comp
            delta_pct = delta * 100  # percentage points
            risk_share = risk_comp / delta * 100 if abs(delta) > 1e-10 else 0

            # Min capital = Cost / Delta(decimal)
            if delta > 1e-8:
                min_cap_y1 = COST_YEAR1 / delta / 1e6  # millions
                min_cap_y2 = COST_ANNUAL / delta / 1e6
            else:
                min_cap_y1 = float('inf'); min_cap_y2 = float('inf')

            print(f"  {gamma:4d} {delta_er*100:+11.2f} {risk_comp*100:+10.4f} {delta_pct:+8.2f} "
                  f"{risk_share:11.1f} {min_cap_y1:13.1f} {min_cap_y2:15.1f}")

        # ── 3. Bootstrap Δ significance ──
        print(f"\n  ── 3. BOOTSTRAP Δ SIGNIFICANCE (γ=5) ──")
        gamma = 5
        boot_deltas = np.zeros(N_BOOT)
        for b in range(N_BOOT):
            idx = np.random.randint(0, ml, size=ml)
            a_b = scaled_a[idx]; m_b = scaled_mean[idx]
            er_ab = np.mean(a_b) * BPY; er_mb = np.mean(m_b) * BPY
            vol_ab = np.std(a_b, ddof=1) * sqrt(BPY)
            vol_mb = np.std(m_b, ddof=1) * sqrt(BPY)
            d_er = er_mb - er_ab
            d_var = vol_mb**2 - vol_ab**2
            boot_deltas[b] = d_er - (gamma/2) * d_var

        delta_point = delta_er - (gamma/2) * delta_var
        ci_lo = np.percentile(boot_deltas, 2.5)
        ci_hi = np.percentile(boot_deltas, 97.5)
        p_val = np.mean(boot_deltas <= 0)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))

        print(f"  Δ(γ=5) = {delta_point*100:+.2f} п.п.")
        print(f"  95% CI: [{ci_lo*100:+.2f}, {ci_hi*100:+.2f}] п.п.")
        print(f"  p-value: {p_val:.4f} {sig}")

        # ── 4. Cost sensitivity ──
        delta_g5 = delta_er - (5/2) * delta_var
        print(f"\n  ── 4. COST SENSITIVITY (γ=5, Δ={delta_g5*100:+.2f} п.п.) ──")
        print(f"  {'Издержки,тыс.руб.':>20s} {'Мин.капитал,млн.руб.':>22s}")
        for cost in [1_800_000, 2_000_000, 3_000_000, 5_000_000]:
            if delta_g5 > 1e-8:
                mc = cost / delta_g5 / 1e6
                print(f"  {cost/1000:20.0f} {mc:22.1f}")
            else:
                print(f"  {cost/1000:20.0f} {'∞':>22s}")

        # ── 5. Market segmentation ──
        print(f"\n  ── 5. MARKET SEGMENTATION (γ=5) ──")
        if delta_g5 > 1e-8:
            print(f"  {'Категория':>30s} {'Капитал,млн':>12s} {'TF':>8s} {'Ценность/год,млн':>17s} {'Окупаемость':>12s}")

            segments = [
                ("Крупные институциональные", ">500", 500, "оба"),
                ("Средние УК", "100-500", 200, "оба"),
                ("Квалиф. частные", "45-100", 70, tf_label.lower()),
                ("Розничные", "<45", 20, "—"),
            ]
            for name, cap_str, cap, tf_rec in segments:
                annual_value = cap * delta_g5  # millions
                if annual_value > 0:
                    payback = COST_YEAR1 / 1e6 / annual_value  # years
                    payback_months = payback * 12
                    print(f"  {name:>30s} {cap_str:>12s} {tf_rec:>8s} {annual_value:17.2f} "
                          f"{payback_months:10.1f} мес.")
                else:
                    print(f"  {name:>30s} {cap_str:>12s} {tf_rec:>8s} {'<0':>17s} {'∞':>12s}")

    # ── 6. IMOEX benchmark ──
    print(f"\n{'='*120}")
    print("  IMOEX BENCHMARK (2022-2025)")
    print(f"{'='*120}")

    # Load IMOEX data (approximate from MOEX ticker or manual)
    # IMOEX 2022-01-03 ≈ 3787, 2025-12-31 ≈ 2800 (approximate)
    # Manual: 2022: -43%, 2023: +44%, 2024: -7%, 2025: +2% (approx)
    imoex_yearly = {"2022": -0.43, "2023": 0.44, "2024": -0.07, "2025": 0.02}
    cum = 1.0
    for y, r in imoex_yearly.items():
        cum *= (1 + r)
    total = cum - 1
    annual = cum ** (1.0 / len(imoex_yearly)) - 1
    # Approximate vol from yearly returns
    vol_imoex = np.std(list(imoex_yearly.values()), ddof=1)
    sh_imoex = (annual - 0.15) / vol_imoex if vol_imoex > 0 else 0  # rf ≈ 15%

    print(f"  IMOEX:")
    print(f"    Yearly returns: {imoex_yearly}")
    print(f"    Cumulative: {total*100:+.1f}%")
    print(f"    Annual: {annual*100:+.1f}%")
    print(f"    Vol (from yearly): {vol_imoex*100:.1f}%")
    print(f"    Sharpe (excess over rf=15%): {sh_imoex:+.2f}")
    print(f"\n  МСП vs IMOEX:")
    print(f"    {'':>15s} {'MSP-A дн.':>12s} {'MSP-BEST дн.':>13s} {'MSP-A час.':>12s} {'MSP-BEST час.':>14s} {'IMOEX':>8s}")
    print(f"    {'Sharpe':>15s} {'5.32':>12s} {'8.88':>13s} {'3.52':>12s} {'7.38':>14s} {sh_imoex:+8.2f}")
    print(f"    {'Annual Ret%':>15s} {'+52.1':>12s} {'+72.3':>13s} {'+23.7':>12s} {'+34.4':>14s} {annual*100:+8.1f}")
    print(f"    {'MDD%':>15s} {'1.5':>12s} {'1.5':>13s} {'0.6':>12s} {'0.9':>14s} {'~50':>8s}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
