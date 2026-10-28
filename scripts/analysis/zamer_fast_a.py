#!/usr/bin/env python3
"""Стадия 3 (A, затем B/C/D поверх A), comm=0,0005, параллельно по (стратегия x тикер) через joblib.

Семантика не меняется: каждая задача вызывает исходные walkforward_A/B/C/D из stage3_walkforward.py
с m.STRATEGIES=[sid], m.TICKERS=[tk]; выбор 10 лучших, сетки, комиссии те же. Опционально --fast
подменяет calc_sharpe_comm на numba-версию с той же формулой (zamer_fast.py).

python3 zamer_fast_a.py <sbx/moex_volatility> --workers N [--fast] [--tickers SBER,LKOH] [--tag имя]
Выход: data/strat_zamer/run_<tag>/ (parquet позиций A-D, csv строк, timing.csv).
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
here = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("sbx")
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--fast", action="store_true")
ap.add_argument("--tickers", default="")
ap.add_argument("--strategies", default="S1,S2,S3,S4,S5,S6")
ap.add_argument("--tag", default="run")
args = ap.parse_args()
SBX = Path(args.sbx)
OUT = here.parents[1] / "data/strat_zamer" / f"run_{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)

_cache = {}


def _setup():
    if "m" in _cache:
        return _cache["m"]
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(SBX / "scripts"))
    sys.path.insert(0, str(here))
    import stage3_walkforward as m
    if args.fast:
        import zamer_fast
        zamer_fast.patch(m)
    m.warmup_numba()
    p = np.zeros(50); p[10:15] = 1.0
    m._apply_B_numba(p, np.random.randn(50).cumsum() + 100, np.ones(50) * 0.02, 0.02, 1.0, 1.5, 15, 0.5, 1.0)
    daily_df, _, _ = m.load_data()
    kr = pd.read_parquet(m.KR_PATH)
    kr["date"] = pd.to_datetime(kr["date"])
    vp = pd.read_parquet(m.VPRED_PATH)
    vp["date"] = pd.to_datetime(vp["date"])
    _cache.update(m=m, daily=daily_df, kr=kr.set_index("date")["key_rate"].sort_index(), vp=vp, td={})
    return m


def task(sid, tk):
    import io
    import contextlib
    t0 = time.time()
    m = _setup()
    if tk not in _cache["td"]:
        _cache["td"][tk] = m.prepare_ticker_data(tk, _cache["daily"], _cache["kr"], _cache["vp"])
    td = _cache["td"][tk]
    t_prep = time.time() - t0
    m.STRATEGIES = [sid]
    m.TICKERS = [tk]
    tdd = {tk: td}
    comm = 0.0005
    tt = {}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        t = time.time(); a_rows, a_pos, _ = m.walkforward_A(tdd, {}, comm); tt["A"] = time.time() - t
        t = time.time(); b_rows, b_pos = m.walkforward_B(tdd, a_pos, comm); tt["B"] = time.time() - t
        t = time.time(); c_rows, c_pos = m.walkforward_C(tdd, a_pos, comm); tt["C"] = time.time() - t
        t = time.time(); d_rows, d_pos = m.walkforward_D(tdd, a_pos, comm); tt["D"] = time.time() - t
    return dict(sid=sid, tk=tk, prep=t_prep, **tt, total=time.time() - t0,
                rows=dict(A=a_rows, B=b_rows, C=c_rows, D=d_rows), pos=dict(A=a_pos, B=b_pos, C=c_pos, D=d_pos),
                dates=td["dates"], lr=td["lr"], n=td["n"])


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    sys.path.insert(0, str(SBX / "scripts"))
    import stage3_walkforward as m0
    tickers = args.tickers.split(",") if args.tickers else list(m0.TICKERS)
    sids = args.strategies.split(",")
    jobs = [(s, t) for t in tickers for s in sids]
    t0 = time.time()
    res = Parallel(n_jobs=args.workers, backend="loky", verbose=5)(delayed(task)(s, t) for s, t in jobs)
    wall = time.time() - t0
    pd.DataFrame([{k: r[k] for k in ("sid", "tk", "prep", "A", "B", "C", "D", "total")} for r in res]).to_csv(OUT / "timing.csv", index=False)
    for ap_ in "ABCD":
        rows, prow = [], []
        for r in res:
            rows += r["rows"][ap_]
            for (sid, tk, yr), pos in r["pos"][ap_].items():
                ts = np.searchsorted(r["dates"], pd.Timestamp(f"{yr}-01-01"))
                te = min(np.searchsorted(r["dates"], pd.Timestamp(f"{yr + 1}-01-01")), r["n"])
                prow.append(pd.DataFrame(dict(date=r["dates"][ts:te], ticker=tk, strategy=sid, year=yr,
                                              position=np.round(pos[ts:te], 6),
                                              daily_gross_return=np.round(pos[ts:te] * r["lr"][ts:te], 8))))
        pd.DataFrame(rows).to_csv(OUT / f"wf_{ap_}_results.csv", index=False)
        if prow:
            pd.concat(prow).sort_values(["strategy", "ticker", "date"]).to_parquet(OUT / f"wf_positions_{ap_}_full.parquet", index=False)
    print(f"WALL {wall:.1f}s workers={args.workers} fast={args.fast} tasks={len(jobs)} sum_task={sum(r['total'] for r in res):.1f}s")
