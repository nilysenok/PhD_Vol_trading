#!/usr/bin/env python3
"""
screener_analysis.py — Detailed analysis of 272 backtest results.
Tables A-E + exit analysis + recommendations.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
TBL = BASE / "results" / "final" / "strategies" / "tables"
DATA = BASE / "results" / "final" / "strategies" / "data"


def load():
    summary = pd.read_csv(TBL / "screener_summary.csv")
    by_ticker = pd.read_csv(TBL / "screener_by_ticker.csv")
    signals = pd.read_parquet(DATA / "signals_screener.parquet")
    return summary, by_ticker, signals


# ════════════════════════════════════════════════════════════
# PART 1: Separate Daily & Hourly tables
# ════════════════════════════════════════════════════════════

def table_by_tf(summary, tf_name):
    df = summary[summary["timeframe"] == tf_name].copy()
    df = df.sort_values("sharpe_median", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    cols = {
        "rank": "Rank", "strategy": "Strategy", "category": "Category",
        "sharpe_median": "Sharpe", "annual_return_median": "AnnRet",
        "annual_vol_median": "AnnVol", "max_drawdown_median": "MaxDD",
        "calmar_median": "Calmar", "win_rate_median": "WinRate",
        "profit_factor_median": "PF", "exposure_median": "Exposure",
        "turnover_median": "Turnover", "avg_trade_bars_median": "AvgTrade",
    }
    out = df[list(cols.keys())].rename(columns=cols)
    return out


def print_tf_table(tbl, title):
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    hdr = (f"{'Rk':>2} {'Strategy':<18} {'Cat':<11} {'Sharpe':>7} {'AnnR%':>7} "
           f"{'AnnV%':>7} {'MaxDD%':>7} {'Calmar':>7} {'WinR%':>6} {'PF':>6} "
           f"{'Exp%':>5} {'Turn':>5} {'AvgT':>5}")
    print(hdr)
    print("-" * 110)
    for _, r in tbl.iterrows():
        pf = f"{r['PF']:>6.2f}" if r['PF'] < 90 else f"{'inf':>6}"
        mark = " <--" if r["Sharpe"] > 0 else ""
        print(f"{int(r['Rank']):>2} {r['Strategy']:<18} {r['Category']:<11} "
              f"{r['Sharpe']:>7.3f} {r['AnnRet']*100:>+6.1f}% "
              f"{r['AnnVol']*100:>6.1f}% {r['MaxDD']*100:>+6.1f}% "
              f"{r['Calmar']:>7.3f} {r['WinRate']*100:>5.1f}% {pf} "
              f"{r['Exposure']*100:>4.1f}% {r['Turnover']:>5.0f} "
              f"{r['AvgTrade']:>5.1f}{mark}")
    print("-" * 110)


# ════════════════════════════════════════════════════════════
# PART 2: Top combinations & positive ticker counts
# ════════════════════════════════════════════════════════════

def top10_by_metric(by_ticker, metric, ascending=False):
    df = by_ticker.sort_values(metric, ascending=ascending).head(10).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "strategy", "timeframe", "ticker",
               "sharpe", "annual_return", "max_drawdown", "calmar",
               "exposure", "profit_factor"]]


def print_top10(df, title, sort_col):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    hdr = (f"{'#':>2} {'Strategy':<18} {'TF':<7} {'Ticker':<6} "
           f"{'Sharpe':>7} {'AnnR%':>7} {'MaxDD%':>7} {'Calmar':>7} "
           f"{'Exp%':>5} {'PF':>6}")
    print(hdr)
    print("-" * 100)
    for _, r in df.iterrows():
        pf = f"{r['profit_factor']:>6.2f}" if r['profit_factor'] < 90 else f"{'inf':>6}"
        print(f"{int(r['rank']):>2} {r['strategy']:<18} {r['timeframe']:<7} "
              f"{r['ticker']:<6} {r['sharpe']:>7.3f} "
              f"{r['annual_return']*100:>+6.1f}% "
              f"{r['max_drawdown']*100:>+6.1f}% "
              f"{r['calmar']:>7.3f} "
              f"{r['exposure']*100:>4.1f}% {pf}")
    print("-" * 100)


def positive_tickers_table(by_ticker):
    rows = []
    for (s, tf), grp in by_ticker.groupby(["strategy", "timeframe"]):
        n_pos = (grp["sharpe"] > 0).sum()
        best = grp.loc[grp["sharpe"].idxmax()]
        worst = grp.loc[grp["sharpe"].idxmin()]
        rows.append({
            "strategy": s, "timeframe": tf,
            "n_positive": n_pos, "n_total": len(grp),
            "pct_positive": n_pos / len(grp) * 100,
            "best_ticker": best["ticker"],
            "best_sharpe": best["sharpe"],
            "worst_ticker": worst["ticker"],
            "worst_sharpe": worst["sharpe"],
            "sharpe_median": grp["sharpe"].median(),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe_median", ascending=False).reset_index(drop=True)
    return df


def print_positive_tickers(df):
    print(f"\n{'=' * 110}")
    print("TABLE E: Positive Sharpe tickers per strategy × timeframe")
    print(f"{'=' * 110}")
    hdr = (f"{'Strategy':<18} {'TF':<7} {'Pos/17':>6} {'%Pos':>5} "
           f"{'ShpMed':>7} {'BestTk':<6} {'BestSh':>7} {'WorstTk':<7} {'WorstSh':>7}")
    print(hdr)
    print("-" * 110)
    for _, r in df.iterrows():
        mark = " <--" if r["sharpe_median"] > 0 else ""
        print(f"{r['strategy']:<18} {r['timeframe']:<7} "
              f"{int(r['n_positive']):>3}/17 {r['pct_positive']:>4.0f}% "
              f"{r['sharpe_median']:>7.3f} "
              f"{r['best_ticker']:<6} {r['best_sharpe']:>7.3f} "
              f"{r['worst_ticker']:<7} {r['worst_sharpe']:>7.3f}{mark}")
    print("-" * 110)


# ════════════════════════════════════════════════════════════
# PART 3: Exit analysis from signals
# ════════════════════════════════════════════════════════════

def analyze_exits(signals):
    """Reconstruct exit statistics from signals_screener.parquet.
    For each strategy×TF×ticker, count SL/TP/signal/maxhold exits
    and compute trade-level stats.
    """
    print("\nAnalyzing exits (this takes a moment)...")

    # We need to reload with full backtest logic, but we can approximate
    # from position changes in signals data.
    # Instead, compute stats from position transitions.

    results = []

    for (sname, tf), grp in signals.groupby(["strategy", "timeframe"]):
        total_trades = 0
        total_bars_in_trade = 0
        win_trades = 0
        loss_trades = 0
        trade_returns = []

        for ticker in grp["ticker"].unique():
            tdf = grp[grp["ticker"] == ticker].sort_values("datetime")
            pos = tdf["position"].values
            dr = tdf["daily_return"].values

            # Find trades: contiguous non-zero position blocks
            in_trade = False
            trade_ret = 0.0
            trade_bars = 0

            for i in range(len(pos)):
                if pos[i] != 0 and not in_trade:
                    in_trade = True
                    trade_ret = dr[i]
                    trade_bars = 1
                elif pos[i] != 0 and in_trade:
                    trade_ret += dr[i]
                    trade_bars += 1
                elif pos[i] == 0 and in_trade:
                    in_trade = False
                    total_trades += 1
                    total_bars_in_trade += trade_bars
                    trade_returns.append(trade_ret)
                    if trade_ret > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1
                    trade_ret = 0.0
                    trade_bars = 0

            # Close last trade if still open
            if in_trade:
                total_trades += 1
                total_bars_in_trade += trade_bars
                trade_returns.append(trade_ret)
                if trade_ret > 0:
                    win_trades += 1
                else:
                    loss_trades += 1

        trade_returns = np.array(trade_returns)
        n_active = (grp["position"] != 0).sum()
        n_total = len(grp)

        avg_trade_bars = total_bars_in_trade / total_trades if total_trades > 0 else 0
        trade_win_rate = win_trades / total_trades if total_trades > 0 else 0

        avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
        avg_loss = trade_returns[trade_returns <= 0].mean() if (trade_returns <= 0).any() else 0
        payoff_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 1e-12 else 99.0

        # Exposure
        exposure = n_active / n_total if n_total > 0 else 0

        results.append({
            "strategy": sname, "timeframe": tf,
            "total_trades": total_trades,
            "avg_trade_bars": avg_trade_bars,
            "trade_win_rate": trade_win_rate,
            "avg_win_ret": avg_win,
            "avg_loss_ret": avg_loss,
            "payoff_ratio": payoff_ratio,
            "exposure": exposure,
            "win_trades": win_trades,
            "loss_trades": loss_trades,
        })

    df = pd.DataFrame(results)
    df = df.sort_values(["strategy", "timeframe"]).reset_index(drop=True)
    return df


def print_exit_analysis(exit_df, summary):
    print(f"\n{'=' * 120}")
    print("EXIT & TRADE ANALYSIS (across 17 tickers)")
    print(f"{'=' * 120}")
    hdr = (f"{'Strategy':<18} {'TF':<7} {'Trades':>7} {'AvgBars':>7} "
           f"{'TradeWR%':>8} {'AvgWin':>8} {'AvgLoss':>8} {'Payoff':>7} "
           f"{'Exp%':>5} {'ShpMed':>7}")
    print(hdr)
    print("-" * 120)

    # Merge sharpe from summary
    merged = exit_df.merge(
        summary[["strategy", "timeframe", "sharpe_median"]],
        on=["strategy", "timeframe"]
    )
    merged = merged.sort_values("sharpe_median", ascending=False)

    for _, r in merged.iterrows():
        mark = " <--" if r["sharpe_median"] > 0 else ""
        print(f"{r['strategy']:<18} {r['timeframe']:<7} "
              f"{int(r['total_trades']):>7} {r['avg_trade_bars']:>7.1f} "
              f"{r['trade_win_rate']*100:>7.1f}% "
              f"{r['avg_win_ret']*100:>7.3f}% {r['avg_loss_ret']*100:>7.3f}% "
              f"{r['payoff_ratio']:>7.2f} "
              f"{r['exposure']*100:>4.1f}% "
              f"{r['sharpe_median']:>7.3f}{mark}")
    print("-" * 120)


# ════════════════════════════════════════════════════════════
# PART 3b: Recommendations
# ════════════════════════════════════════════════════════════

def print_recommendations(summary, exit_df, pos_tickers):
    print(f"\n{'=' * 100}")
    print("RECOMMENDATIONS PER STRATEGY")
    print(f"{'=' * 100}")

    merged = summary.merge(exit_df, on=["strategy", "timeframe"])
    merged = merged.sort_values("sharpe_median", ascending=False)

    for _, r in merged.iterrows():
        s = r["strategy"]
        tf = r["timeframe"]
        shp = r["sharpe_median"]
        exp = r["exposure_median"]
        mdd = r["max_drawdown_median"]
        turn = r["turnover_median"]
        twr = r["trade_win_rate"]
        payoff = r["payoff_ratio"]
        avg_bars = r["avg_trade_bars"]
        trades = r["total_trades"]

        # Get positive ticker count
        pt_row = pos_tickers[(pos_tickers["strategy"] == s) &
                              (pos_tickers["timeframe"] == tf)]
        n_pos = int(pt_row["n_positive"].values[0]) if len(pt_row) > 0 else 0

        issues = []
        recs = []

        # Diagnose
        if shp < 0:
            issues.append(f"Negative Sharpe ({shp:.3f})")
        if exp < 0.05:
            issues.append(f"Very low exposure ({exp*100:.1f}%)")
            recs.append("Relax filters (softer RSI/ADX thresholds) or remove redundant conditions")
        if exp < 0.10 and exp >= 0.05:
            issues.append(f"Low exposure ({exp*100:.1f}%)")
            recs.append("Consider softening entry filters to increase signal count")
        if mdd < -0.50:
            issues.append(f"Severe drawdown ({mdd*100:.0f}%)")
            recs.append("Tighten SL or add regime filter (vol-forecast can help)")
        if mdd < -0.40 and mdd >= -0.50:
            issues.append(f"Large drawdown ({mdd*100:.0f}%)")
            recs.append("Use adaptive SL via sigma_pred to reduce drawdown")
        if twr < 0.40:
            issues.append(f"Low trade win rate ({twr*100:.0f}%)")
            recs.append("Improve entry timing or add confirmation filter")
        if payoff < 1.0 and twr < 0.50:
            issues.append(f"Bad payoff ratio ({payoff:.2f}) with low WR ({twr*100:.0f}%)")
            recs.append("SL too tight or TP too far — consider SL=2.5×ATR or TP=2.5×ATR")
        if payoff < 1.2 and shp > 0:
            recs.append("Marginal edge — vol-forecast adaptive stops could significantly improve")
        if turn > 80:
            issues.append(f"High turnover ({turn:.0f}/yr)")
            recs.append("Add minimum holding period or cooldown between trades")
        if n_pos < 9:
            issues.append(f"Only {n_pos}/17 tickers profitable")
        if n_pos >= 12:
            recs.append(f"Robust: {n_pos}/17 tickers profitable — good candidate for vol-overlay")

        # Strategy-specific
        if s == "S6_DualMA" and tf == "hourly" and shp > 0.3:
            recs.append("BEST CANDIDATE: use vol-forecast for position sizing (vol-targeting)")
        if s == "S4_Donchian" and mdd < -0.40:
            recs.append("Breakout works but drawdown kills — vol-regime filter to skip low-vol breakouts")
        if s == "S5_Supertrend" and shp > 0 and shp < 0.1:
            recs.append("Weak edge — vol-adaptive multiplier could improve (replace fixed 3×ATR)")
        if "Reversion" in s or "Bollinger" in s or "Keltner" in s:
            if shp < 0:
                recs.append("Mean reversion on MOEX 2020-26 is structurally hard — consider only during predicted low-vol regimes")
        if s == "S1_MA_Reversion":
            if exp < 0.05:
                recs.append("z<-2 AND RSI<40 is very restrictive — try z<-1.5 or RSI<50")
            elif exp > 0.30 and shp < 0:
                recs.append("High exposure but losing — the z-score reversion logic itself fails in trending MOEX")

        print(f"\n  {s} [{tf}]  Sharpe={shp:.3f}  Exp={exp*100:.1f}%  "
              f"MaxDD={mdd*100:.0f}%  TradeWR={twr*100:.0f}%  Payoff={payoff:.2f}  "
              f"Pos_tickers={n_pos}/17")

        if issues:
            print(f"  Issues: {'; '.join(issues)}")
        if recs:
            for i, rec in enumerate(recs):
                print(f"  -> {rec}")
        if not issues and not recs:
            print(f"  No major issues identified.")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    summary, by_ticker, signals = load()
    print(f"Loaded: summary={len(summary)}, by_ticker={len(by_ticker)}, signals={len(signals):,}")

    # PART 1: Daily & Hourly tables
    tbl_d = table_by_tf(summary, "daily")
    tbl_h = table_by_tf(summary, "hourly")
    print_tf_table(tbl_d, "TABLE A: DAILY STRATEGIES (8 strategies, sorted by Sharpe median)")
    print_tf_table(tbl_h, "TABLE B: HOURLY STRATEGIES (8 strategies, sorted by Sharpe median)")

    tbl_d.to_csv(TBL / "screener_daily.csv", index=False)
    tbl_h.to_csv(TBL / "screener_hourly.csv", index=False)

    # PART 2: Top combinations
    top_sharpe = top10_by_metric(by_ticker, "sharpe")
    top_calmar = top10_by_metric(by_ticker, "calmar")
    pos_tickers = positive_tickers_table(by_ticker)

    print_top10(top_sharpe, "TABLE C: Top-10 strategy × TF × ticker by SHARPE", "sharpe")
    print_top10(top_calmar, "TABLE D: Top-10 strategy × TF × ticker by CALMAR", "calmar")
    print_positive_tickers(pos_tickers)

    top_sharpe.to_csv(TBL / "top10_by_sharpe.csv", index=False)
    top_calmar.to_csv(TBL / "top10_by_calmar.csv", index=False)
    pos_tickers.to_csv(TBL / "positive_tickers_count.csv", index=False)

    # PART 3: Exit analysis
    exit_df = analyze_exits(signals)
    print_exit_analysis(exit_df, summary)
    exit_df.to_csv(TBL / "exit_analysis.csv", index=False)

    # PART 3b: Recommendations
    print_recommendations(summary, exit_df, pos_tickers)

    print(f"\n{'=' * 100}")
    print("All tables saved to results/final/strategies/tables/")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
