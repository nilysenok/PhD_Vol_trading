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

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import msp
from fastbt.cache import MAIN

ROOT = Path(__file__).resolve().parent.parent
TABLES = Path(os.environ.get("CH4_TABLES", ROOT / "docs" / "tables"))   # тест: docs/tables/strat_test
SRC = {"маска [0]": MAIN / "data" / "fastbt" / "a_honest.parquet",
       "маска [1] (утечка)": MAIN / "data" / "fastbt" / "a_mask1.parquet"}
COMM = 0.0005


REENTRY_N = (1, 3, 5)


def annotate(g: pd.DataFrame) -> pd.DataFrame:
    """Строки одной пары (по дате): номер сделки, направление, перезаход.
    Сделка — участок постоянной ненулевой позиции; разворот = выход + вход на том же баре.
    Комиссия бара делится точно: |p[t−1]|·c — выход прежней сделки, |p[t]|·c — вход новой.
    gap — баров от бара выхода прежней сделки (первый бар не в ней) до входа (разворот: 0)."""
    p = g["position"].values
    n = len(p)
    prev = np.r_[0.0, p[:-1]]
    chg = p != prev
    entry = (p != 0) & chg
    tid = np.where(p != 0, np.cumsum(entry), 0)
    # бар выхода сделки k — первый бар после её последнего бара
    gap = np.full(n, np.nan)
    last_exit = -1
    for t in range(n):
        if chg[t] and prev[t] != 0:
            last_exit = t
        if entry[t]:
            if last_exit >= 0:
                gap[t] = t - last_exit
    g = g.assign(tid=tid, entry=entry, gap=gap, side=np.sign(p),
                 c_exit=np.where(chg, np.abs(prev), 0.0), c_entry=np.where(chg, np.abs(p), 0.0),
                 exit_tid=np.where(chg & (prev != 0), np.r_[0, tid[:-1]], 0),
                 exit_side=np.where(chg, np.sign(prev), 0.0))
    return g


def trades(pos: pd.DataFrame) -> pd.DataFrame:
    """Сделки всех пар: strategy, ticker, year (входа), side, ret (Σ pos·lr), gap до входа."""
    out = []
    for (s, t), g in pos.sort_values("date").groupby(["strategy", "ticker"], sort=False):
        g = annotate(g)
        m = g["tid"] > 0
        if not m.any():
            continue
        a = g[m].groupby("tid").agg(ret=("daily_gross_return", "sum"),
                                    year=("date", lambda d: d.iloc[0].year),
                                    days=("position", "size"), side=("side", "first"),
                                    gap=("gap", "first"))
        out.append(a.assign(strategy=s, ticker=t))
    return pd.concat(out, ignore_index=True)


def decompose(pos: pd.DataFrame, rate: pd.Series):
    port = msp.portfolio(pos, rate, COMM)
    rfcal = port["total"] - port["excess"]
    g = pos.sort_values(["ticker", "strategy", "date"]).copy()
    g = pd.concat([annotate(x) for _, x in g.groupby(["ticker", "strategy"], sort=False)])
    dp = g.groupby(["ticker", "strategy"])["position"].diff().fillna(g["position"])
    g["comm"] = dp.abs() * COMM
    # перезаходы: сделка с gap ≤ N; её строки (доход) и её комиссии (вход + выход)
    gap_of = g[g["entry"]].set_index(["ticker", "strategy", "tid"])["gap"]
    key = pd.MultiIndex.from_arrays([g["ticker"], g["strategy"], g["tid"]])
    g["trade_gap"] = gap_of.reindex(key).values
    ekey = pd.MultiIndex.from_arrays([g["ticker"], g["strategy"], g["exit_tid"]])
    g["exit_gap"] = gap_of.reindex(ekey).values
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
            trade_net_on_exposure_pct=p["r"].sum() / max(p["exposure"].mean(), 1e-12) * 100,
            **longshort(gy, ty), **reentry(gy, ty)))
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


def longshort(gy, ty) -> dict:
    """Лонг/шорт: торговля нетто (комиссия — по направлению сделки входа/выхода), сделки."""
    out = {}
    for name, sd in (("long", 1), ("short", -1)):
        m = gy["side"] == sd
        gross = (gy.loc[m, "daily_gross_return"] * gy.loc[m, "w"]).sum()
        # комиссия входа — сторона текущей позиции; выхода — сторона прежней (знак prev)
        ce = (gy["c_entry"] * COMM * gy["w"])[m].sum()
        cx = (gy["c_exit"] * COMM * gy["w"])[gy["exit_side"] == sd].sum()
        t = ty[ty["side"] == sd]
        out.update({f"trade_net_{name}_pct": (gross - ce - cx) * 100,
                    f"n_trades_{name}": len(t),
                    f"win_{name}_pct": (t["ret"] > 0).mean() * 100 if len(t) else np.nan})
    return out


def reentry(gy, ty) -> dict:
    """Перезаходы в ту же пару в пределах N баров после выхода: доля сделок, вклад в
    торговлю нетто (доход строк сделки − её комиссии входа и выхода), их комиссии."""
    out = {}
    for N in REENTRY_N:
        t = ty[ty["gap"] <= N]
        m = gy["trade_gap"] <= N
        ce = (gy["c_entry"] * COMM * gy["w"])[m].sum()
        cx = (gy["c_exit"] * COMM * gy["w"])[gy["exit_gap"] <= N].sum()
        gross = (gy.loc[m, "daily_gross_return"] * gy.loc[m, "w"]).sum()
        out.update({f"reentry{N}_share_pct": len(t) / max(len(ty), 1) * 100,
                    f"reentry{N}_net_pct": (gross - ce - cx) * 100,
                    f"reentry{N}_comm_pct": (ce + cx) * 100,
                    f"reentry{N}_win_pct": (t["ret"] > 0).mean() * 100 if len(t) else np.nan})
    return out


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
