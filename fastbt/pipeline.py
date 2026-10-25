"""Полный walk-forward: тикер → стратегии → A, B, C, D → дневные позиции (формат V4).

python3 -m fastbt.pipeline [--leak-mask-old] [--tickers SBER,IRAO] [--sids S1,S3]
        [--workers 12] [--out data/fastbt/positions.parquet]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import grids
from fastbt.cache import MAIN
from fastbt.config import Config, add_args, from_args
from fastbt.data import TICKERS, year_bounds

ROOT = Path(__file__).resolve().parent.parent
COLS = ["date", "strategy", "tf", "ticker", "approach", "test_year", "position",
        "daily_gross_return"]


def _rows(u, sname, approach, res, years):
    out = []
    for Y in years:
        if Y not in res:
            continue
        pos = res[Y][1]
        ys = year_bounds(u.dates, Y)[0]
        ye = year_bounds(u.dates, Y + 1)[0] if Y < 2026 else u.n
        p = np.round(pos[ys:ye].astype(np.float64), 6) + 0.0
        r = np.round(pos[ys:ye] * u.lr[ys:ye], 8) + 0.0
        out.append(pd.DataFrame({"date": pd.to_datetime(u.dates[ys:ye]), "strategy": sname,
                                 "tf": "daily", "ticker": u.ticker, "approach": approach,
                                 "test_year": Y, "position": p, "daily_gross_return": r}))
    return out


def run_unit(ticker, sid, cfg, td=None):
    """Возвращает (DataFrame позиций, сводка выбора по годам)."""
    from fastbt.approach_a import ApproachA
    from fastbt.approach_b import run_b
    from fastbt.approach_cd import run_c, run_d
    from fastbt.data import load_ticker
    from fastbt.design import apply, prepare_td
    from fastbt.unit import make_unit
    apply(cfg.design)
    td = load_ticker(ticker, cfg.corp_actions) if td is None else td
    u = make_unit(ticker, sid, prepare_td(td, cfg.design))
    A = ApproachA(u, cfg)
    a = A.run()
    ap = cfg.approaches
    b = run_b(u, cfg, a) if "B" in ap else {}
    c = run_c(u, cfg, a) if "C" in ap else {}
    d = run_d(u, cfg, a) if "D" in ap else {}
    sname = grids.STRATEGY_NAMES[sid]
    parts = (_rows(u, sname, "A", a, cfg.years_a) + _rows(u, sname, "B", b, cfg.years_bcd)
             + _rows(u, sname, "C", c, cfg.years_bcd) + _rows(u, sname, "D", d, cfg.years_bcd))
    meta = {"A": {Y: [[int(x) for x in t] for t in v[0]] for Y, v in a.items()},
            "A_info": A.info,
            "B": {Y: v[0] for Y, v in b.items()},
            "C": {Y: v[0] for Y, v in c.items()},
            "D": {Y: v[0] for Y, v in d.items()}}
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    return df, meta


def run_ticker(args):
    ticker, sids, cfg = args
    t0 = time.time()
    dfs, metas = [], {}
    for sid in sids:
        df, meta = run_unit(ticker, sid, cfg)
        dfs.append(df)
        metas[sid] = meta
    return ticker, pd.concat(dfs, ignore_index=True), metas, time.time() - t0


def run_all(cfg, tickers=TICKERS, sids=tuple(grids.STRATEGY_NAMES), workers=1, log=print):
    jobs = [(t, list(sids), cfg) for t in tickers]
    out, metas = [], {}
    if workers <= 1:
        it = map(run_ticker, jobs)
    else:
        import multiprocessing as mp
        os.environ["NUMBA_NUM_THREADS"] = "1"
        pool = mp.get_context("spawn").Pool(workers)
        it = pool.imap_unordered(run_ticker, jobs)
    for tk, df, meta, dt in it:
        out.append(df)
        metas[tk] = meta
        log(f"{tk}: {len(df)} строк, {dt:.1f} с")
    if workers > 1:
        pool.close()
        pool.join()
    return pd.concat(out, ignore_index=True), metas


def main(argv=None):
    p = argparse.ArgumentParser()
    add_args(p)
    p.add_argument("--tickers", default=",".join(TICKERS))
    p.add_argument("--sids", default="S1,S2,S3,S4,S5,S6")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", default=str(MAIN / "data" / "fastbt" / "positions.parquet"))
    a = p.parse_args(argv)
    cfg = from_args(a)
    t0 = time.time()
    df, metas = run_all(cfg, a.tickers.split(","), a.sids.split(","), a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    Path(a.out).with_suffix(".meta.json").write_text(
        json.dumps({"config": cfg.to_dict(), "meta": metas}, default=str, ensure_ascii=False))
    print(f"готово: {len(df)} строк, {time.time() - t0:.1f} с → {a.out}")


if __name__ == "__main__":
    main()
