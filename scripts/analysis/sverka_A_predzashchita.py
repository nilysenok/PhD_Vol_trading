"""Сверка A «предзащита → сейчас», только 2018–2021 (владелец 27.10, 11:30). Движок не правится.

  SDF_CODE=<дерево кода> FASTBT_MAIN=<своя копия> FASTBT_CACHE=<папка> python3 sverka_A_predzashchita.py pos \
      --out p.parquet [--mask1] [--sel-comm 0.004] [--no-corp] [--tf] [--workers 6]
  python3 sverka_A_predzashchita.py port p.parquet [--comm 0] [--rate back|cbr] [--cal back|fwd] \
      [--agg pairs|ticker] [--exit-fee] [--lev-cal]
  python3 sverka_A_predzashchita.py            (самопроверка на синтетике)

pos  — позиции A (6 семей x 17 тикеров, 2018–2021) кодом из SDF_CODE (git archive нужного коммита).
port — слой портфеля со своими переключателями (определения, не ошибки: комиссия ряда, ставка, агрегирование;
       ошибки Н2/Н4/Н5 включаются флагами --cal fwd, --lev-cal, --exit-fee; в движок флаги не добавлены).
"""
from __future__ import annotations

import argparse
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd

SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
YEARS = (2018, 2019, 2020, 2021)


def sharpe(x) -> float:
    x = np.asarray(x, float)
    s = np.std(x, ddof=1)
    return float(np.mean(x) * sqrt(252) / s) if s > 1e-12 else 0.0


def cal_days(index, fwd: bool) -> np.ndarray:
    v = pd.DatetimeIndex(index).values
    cal = np.ones(len(v))
    dd = np.diff(v).astype("timedelta64[D]").astype(float)
    if fwd:
        cal[:-1] = dd
    else:
        cal[1:] = dd
    return cal


def portfolio(pos: pd.DataFrame, rate: pd.Series, comm: float = 0.0, agg: str = "pairs",
              fwd: bool = True, exit_fee: bool = True, lev_cal: bool = True) -> pd.DataFrame:
    """pos: date, ticker, strategy, position, gross (доходность позиции d→d+1 до комиссии).
    Те же формулы, что msp.portfolio, но каждое решение — свой переключатель."""
    g = pos[["date", "ticker", "strategy", "position", "gross"]].copy()
    if agg == "ticker":
        g = g.groupby(["date", "ticker"]).agg(position=("position", "mean"), gross=("gross", "mean")).reset_index()
        keys = ["ticker"]
    else:
        keys = ["ticker", "strategy"]
    g = g.sort_values(keys + ["date"])
    sides = g.groupby(keys)["position"].diff().fillna(g["position"]).abs()
    if exit_fee:                                   # Н5: выход на последнем баре окна
        sides = sides + g["position"].abs().where(~g.duplicated(keys, keep="last"), 0.0)
    g["net"] = g["gross"] - sides * comm
    g["ap"] = g["position"].abs()
    if agg == "tickw":                             # издержки по парам, вес 1/17 на тикер (а не 1/102 на пару)
        g = g.groupby(["date", "ticker"])[["net", "ap"]].mean().reset_index()
    p = g.groupby("date").agg(r=("net", "mean"), exposure=("ap", "mean")).sort_index()
    kr = rate.reindex(rate.index.union(p.index)).sort_index().ffill().bfill().reindex(p.index).values
    cal = cal_days(p.index, fwd)
    mcal = cal if lev_cal else np.ones(len(p))     # Н4: плата за плечо за календарные дни
    ex = p["exposure"].values
    p["total"] = p["r"] + np.maximum(0, 1 - ex) * kr / 365 * cal - np.maximum(0, ex - 1) * (kr + .02) / 365 * mcal
    p["excess"] = p["total"] - kr / 365 * cal
    return p


def by_year(p: pd.DataFrame) -> dict:
    return {y: sharpe(p["excess"][p.index.year == y]) for y in YEARS}


def load_rate(mode: str) -> pd.Series:
    sys.path.insert(0, os.environ["SDF_CODE"])
    from fastbt import msp
    return msp.rate_restored() if mode == "back" else msp.rate_cbr()


def produce(a) -> None:
    import dataclasses
    import multiprocessing as mp
    os.environ["NUMBA_NUM_THREADS"] = "1"
    jobs = [(t, s, a.mask1, a.sel_comm, a.no_corp, a.tf) for t in _tickers() for s in SIDS]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        parts = [x for x in pool.imap_unordered(_unit, jobs) if x is not None]
    pd.concat(parts).to_parquet(a.out)
    print("позиций:", sum(len(x) for x in parts), "->", a.out)


def _tickers():
    sys.path.insert(0, os.environ["SDF_CODE"])
    from fastbt.data import TICKERS
    return list(TICKERS)


def _unit(j):
    sys.path.insert(0, os.environ["SDF_CODE"])
    os.environ["NUMBA_NUM_THREADS"] = "1"
    import dataclasses
    from fastbt import design as dsg
    from fastbt import pipeline, turnover
    from fastbt.approach_a import ApproachA
    from fastbt.config import Config
    from fastbt.data import load_ticker
    from fastbt.unit import make_unit
    t, sid, mask1, comm, no_corp, tf = j
    cfg = dataclasses.replace(Config(), mask_old=mask1, comm=comm, corp_actions=not no_corp, years_a=YEARS)
    if tf:
        import yaml
        d = yaml.safe_load(open(os.path.join(os.environ["SDF_CODE"], "fastbt/designs/final/main.yaml")))
        from fastbt import dev
        cfg = dataclasses.replace(dev.make_config(d, "dev"), mask_old=mask1, comm=comm, corp_actions=not no_corp,
                                  years_a=YEARS)
    dsg.apply(cfg.design)
    td = load_ticker(t, cfg.corp_actions)
    u = make_unit(t, sid, td)
    res = ApproachA(u, cfg).run()
    if tf:
        res = turnover.apply(res, u, cfg, "A")
    parts = pipeline._rows(u, sid, "A", res, YEARS)
    if not parts:
        return None
    df = pd.concat(parts).rename(columns={"daily_gross_return": "gross"})
    return df[["date", "ticker", "strategy", "position", "gross"]]


def selftest() -> None:
    n = 55                                      # последний бар с открытой позицией: Н5 виден
    idx = pd.bdate_range("2019-01-01", periods=n)
    rows = []
    for t in ("X", "Y"):
        for s in ("S1", "S2"):
            pos = np.where(np.arange(n) % 10 < 5, 1.0, 0.0) * (1 if t == "X" else -1)
            gr = np.random.default_rng(len(t + s)).normal(0, .01, n) * pos
            rows.append(pd.DataFrame(dict(date=idx, ticker=t, strategy=s, position=pos, gross=gr)))
    pos = pd.concat(rows)
    rate = pd.Series(.1, index=idx)
    base = portfolio(pos, rate, 0.0, "pairs", True, False)
    assert portfolio(pos, rate, .01, "pairs", True, False)["excess"].mean() < base["excess"].mean(), "комиссия не действует"
    assert not np.allclose(portfolio(pos, rate, 0, "pairs", True, False)["excess"], portfolio(pos, rate, 0, "pairs", False, False)["excess"]), "дни ставки"
    nx = pos[~((pos.ticker == "Y") & (pos.strategy == "S2"))]          # пара пропала: 1/N по парам != EW по тикерам
    assert not np.allclose(portfolio(nx, rate, 0, "pairs")["r"], portfolio(nx, rate, 0, "ticker")["r"]), "агрегирование"
    assert np.allclose(portfolio(pos, rate, 0, "pairs")["r"], portfolio(pos, rate, 0, "ticker")["r"]), "при полной панели равны"
    assert not np.allclose(portfolio(pos, rate, .01, "pairs", exit_fee=True)["r"], portfolio(pos, rate, .01, "pairs", exit_fee=False)["r"]), "выход Н5"
    big = pos.assign(position=pos["position"] * 3, gross=pos["gross"] * 3)   # плечо 3x: Н4 меняет результат
    assert not np.allclose(portfolio(big, rate, 0, lev_cal=True)["total"], portfolio(big, rate, 0, lev_cal=False)["total"]), "Н4"
    bad = base.assign(excess=base["excess"] + .001)                    # подложенная ошибка: ставка не вычтена
    assert sharpe(bad["excess"]) != sharpe(base["excess"]) and abs(sharpe(sharpe_wrong(base)) - sharpe(base["excess"])) > 1e-9
    print("самопроверка OK")


def sharpe_wrong(p):
    return p["total"]                                                  # сырая вместо избыточной — ловится


def main() -> None:
    if len(sys.argv) == 1:
        return selftest()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pos")
    q.add_argument("--out", required=True)
    q.add_argument("--mask1", action="store_true")
    q.add_argument("--sel-comm", type=float, default=0.0004)
    q.add_argument("--no-corp", action="store_true")
    q.add_argument("--tf", action="store_true")
    q.add_argument("--workers", type=int, default=6)
    r = sub.add_parser("port")
    r.add_argument("pos")
    r.add_argument("--comm", type=float, default=0.0)
    r.add_argument("--rate", default="back")
    r.add_argument("--cal", default="back")
    r.add_argument("--agg", default="pairs")
    r.add_argument("--exit-fee", action="store_true")
    r.add_argument("--lev-cal", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pos":
        return produce(a)
    sys.path.insert(0, os.environ["SDF_CODE"])
    p = portfolio(pd.read_parquet(a.pos), load_rate(a.rate), a.comm, a.agg, a.cal == "fwd", a.exit_fee, a.lev_cal)
    print("sharpe %.4f" % sharpe(p["excess"]), {y: round(v, 3) for y, v in by_year(p).items()})


if __name__ == "__main__":
    main()
