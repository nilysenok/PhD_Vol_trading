"""Разложение МСП-A по годам (решение владельца 25.10, 18:50): торговая компонента отдельно от
дохода на кэш, сделки, оборот, комиссии, экспозиция, доля прибыльных сделок, вклад S1–S6.

python3 -m fastbt.decomp      (после chapter4_a и chapter4_a --leak-mask-old)
Вход: <основная копия>/data/fastbt/a_honest.parquet (маска [0]), a_mask1.parquet (маска [1]).
Выход: docs/tables/tabl_4_a_razlozhenie.csv, tabl_4_a_vklad_strategiy.csv.

Тождество (доходности логарифмические, суммы за год в % капитала, экспозиция ≤ 1):
excess = trade_net + cash − rf = trade_net − exp_cost, где
  trade_net = Σ_дни mean_пар(pos·lr − |Δpos|·c)   (торговая компонента после комиссии),
  cash      = Σ (1 − exp)·rf·cal_days              (доход на свободный капитал),
  exp_cost  = Σ exp·rf·cal_days                    (ставка, недополученная на занятом капитале).
Вклад стратегии s в excess = Σ_дни Σ_{пары s} (net − |pos|·rf·cal) / n_пар(день); сумма по s = excess.
Сделка — участок постоянной ненулевой позиции пары; год сделки — год входа; прибыльная — Σ pos·lr > 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import msp
from fastbt.cache import MAIN

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "docs" / "tables"
SRC = {"маска [0]": MAIN / "data" / "fastbt" / "a_honest.parquet",
       "маска [1] (утечка)": MAIN / "data" / "fastbt" / "a_mask1.parquet"}
COMM = 0.0005


def trades(pos: pd.DataFrame) -> pd.DataFrame:
    """Сделки всех пар: strategy, ticker, year (входа), ret (Σ pos·lr)."""
    out = []
    for (s, t), g in pos.sort_values("date").groupby(["strategy", "ticker"], sort=False):
        p = g["position"].values
        r = g["daily_gross_return"].values
        y = g["date"].dt.year.values
        new = (p != 0) & (np.r_[0.0, p[:-1]] != p)          # вход или разворот
        tid = np.cumsum(new)
        m = p != 0
        if not m.any():
            continue
        df = pd.DataFrame({"tid": tid[m], "r": r[m], "y": y[m]})
        a = df.groupby("tid").agg(ret=("r", "sum"), year=("y", "first"), days=("r", "size"))
        out.append(a.assign(strategy=s, ticker=t))
    return pd.concat(out, ignore_index=True)


def decompose(pos: pd.DataFrame, rate: pd.Series):
    port = msp.portfolio(pos, rate, COMM)
    rfcal = port["total"] - port["excess"]
    g = pos.sort_values(["ticker", "strategy", "date"]).copy()
    dp = g.groupby(["ticker", "strategy"])["position"].diff().fillna(g["position"])
    g["comm"] = dp.abs() * COMM
    g["net"] = g["daily_gross_return"] - g["comm"]
    n_pairs = g.groupby("date")["position"].size()
    g["w"] = 1.0 / g["date"].map(n_pairs)
    g["rfcal"] = g["date"].map(rfcal)
    g["xs"] = (g["net"] - g["position"].abs() * g["rfcal"]) * g["w"]
    g["netw"] = g["net"] * g["w"]
    g["year"] = g["date"].dt.year
    tr = trades(pos)
    rows, contrib = [], []
    for y, p in port.groupby(port.index.year):
        rf_y = rfcal.loc[p.index]
        gy = g[g["year"] == y]
        ty = tr[tr["year"] == y]
        n_pair_years = gy.groupby(["ticker", "strategy"]).ngroups
        rows.append(dict(
            year=y, n_days=len(p), sharpe_excess=msp.sharpe(p["excess"]),
            excess_pct=p["excess"].sum() * 100, total_pct=p["total"].sum() * 100,
            trade_gross_pct=(gy["daily_gross_return"] * gy["w"]).sum() * 100,
            commission_pct=(gy["comm"] * gy["w"]).sum() * 100,
            trade_net_pct=p["r"].sum() * 100, cash_pct=p["fdr"].sum() * 100,
            rf_pct=rf_y.sum() * 100, exp_cost_pct=(p["exposure"] * rf_y).sum() * 100,
            exposure_pct=p["exposure"].mean() * 100,
            turnover_pct_day=p["turnover"].mean() * 100,
            turnover_x_year=p["turnover"].sum(),
            n_trades=len(ty), trades_per_pair=len(ty) / max(n_pair_years, 1),
            win_rate_pct=(ty["ret"] > 0).mean() * 100 if len(ty) else np.nan,
            avg_trade_ret_pct=ty["ret"].mean() * 100 if len(ty) else np.nan,
            avg_trade_days=ty["days"].mean() if len(ty) else np.nan,
            trade_net_on_exposure_pct=p["r"].sum() / max(p["exposure"].mean(), 1e-12) * 100))
        for s, gs in gy.groupby("strategy"):
            ts = ty[ty["strategy"] == s]
            contrib.append(dict(year=y, strategy=s, trade_net_pct=gs["netw"].sum() * 100,
                                excess_contrib_pct=gs["xs"].sum() * 100,
                                commission_pct=(gs["comm"] * gs["w"]).sum() * 100,
                                n_trades=len(ts),
                                win_rate_pct=(ts["ret"] > 0).mean() * 100 if len(ts) else np.nan,
                                exposure_pct=(gs["position"].abs() * gs["w"]).groupby(gs["date"])
                                .sum().mean() * 100))
    return pd.DataFrame(rows), pd.DataFrame(contrib)


def main(argv=None):
    rate = msp.rate_cbr()
    r_all, c_all = [], []
    for name, path in SRC.items():
        pos = pd.read_parquet(path)
        pos["date"] = pd.to_datetime(pos["date"])
        pos = pos[pos["approach"] == "A"]
        r, c = decompose(pos, rate)
        r_all.append(r.assign(variant=name))
        c_all.append(c.assign(variant=name))
    r, c = pd.concat(r_all, ignore_index=True), pd.concat(c_all, ignore_index=True)
    chk = (r["trade_net_pct"] - r["exp_cost_pct"] - r["excess_pct"]).abs().max()
    chk2 = (c.groupby(["variant", "year"])["excess_contrib_pct"].sum().values
            - r.set_index(["variant", "year"]).sort_index()["excess_pct"].values)
    print(f"проверка тождества: {chk:.2e}; вклады: {np.abs(chk2).max():.2e}")
    TABLES.mkdir(parents=True, exist_ok=True)
    r.to_csv(TABLES / "tabl_4_a_razlozhenie.csv", index=False)
    c.to_csv(TABLES / "tabl_4_a_vklad_strategiy.csv", index=False)
    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))
    print(c.pivot_table(index=["variant", "strategy"], columns="year",
                        values="excess_contrib_pct").round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
