"""FS: эквивалентность быстрого слоя (fastbt.fast_*) и действующего пути fastbt.s9 на разработке
2018–2021 — N случайных конфигураций × тикеры × 4 подхода (как шаг 3, `fastbt.equiv`).

Подходы и конфигурации (на тикер, по N каждой):
- A: (стратегия, год, удержание, запрет входа) — A без фильтра оборота из кэша быстрого слоя
     против кэша старого пути (s9._job), затем фильтр оборота (turnover.tfilter);
- B: точка сетки B (σ-источник, cap, полоса) — веса S9.b_weights;
- C: точка C1 / C2 / доп. точки C (--c-dop: C2W, C*1, C*2) — затворы по категории стратегии;
- D: точка D1–D7 (S9.d_points) — затвор или вес.
Для каждой конфигурации и случайной пары-года (стратегия, Y ∈ 2018–2021) сравниваются:
pos_eq — позиции A·g на окне валидации и году Y совпадают точно; ret_diff — max|Δ(pos·lr)|;
sharpe_rel — Шарп на валидации 2017…Y−1 (быстрое ядро fast_grid.grid_val_sharpe против
approach_cd.masked_eval / scaled_eval) и Шарп года Y (stats.sharpe_window). Для B/C/D ещё
портфель МСП по выбранным тикерам (Engine.port): Шарп избыточной доходности, p IID-бутстрапа
(report.boot_delta против fast_boot.boot_delta) и, у каждой 10-й, p Ледуа–Вольфа.
Допуск — 1e-9 (Шарпы относительный), позиции и p — точно.

python3 scripts/analysis/fastbt_equiv_fast.py [--tickers SBER,IRAO,MOEX,MTLR,AFLT] [--n 200]
        [--sigma-h1 final3_full] [--c-dop] [--out csv] [--workers 3]
σ после 31.12.2021 не читается (fastbt.s9.CUT); окна — только 2017–2021.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOL = 1e-9


def _rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-12)) if b != 0 else float(abs(a))


def points(x, src, with_dop):
    """{подход: [(метка, G, вид, cat)]} для σ-источника src (тем же кодом S9)."""
    from fastbt import s9
    out = {"B": [], "C": [], "D": []}
    for cap in (1.0, 1.5, 2.0):
        for band in (0.0, 0.10, 0.20):
            out["B"].append((f"B:cap{cap:g}_band{int(band*100)}", x.b_weights(src, cap, band),
                             "scale", False))
    p1, p2 = x.c_points(src)
    out["C"] += [(f"C1:{lab}", G, "mask", True) for lab, G in p1]
    out["C"] += [(f"C2:{lab}", G, "scale", True) for lab, G in p2]
    if with_dop and s9.C_DOP is not None:
        rv = None if src.startswith("naive") else x.e.rvw()
        for fam, pts in s9.c_dop_gates(x.e.sigma(src), rv, x.e.n_t, s9.C_DOP):
            kind = "mask" if fam in ("C1", "C*1") else "scale"
            out["C"] += [(f"{fam}:{lab}", G, kind, True) for lab, G in pts]
    for fam, pts in x.d_points(src).items():
        out["D"] += [(f"{fam}:{lab}", G, kind, False) for lab, G, kind in pts]
    return out


def val_sharpe_old(apos, row, kind, lr, vs, ve):
    from fastbt.approach_cd import _eval, scaled_eval
    from fastbt.s9 import COMM
    r = np.ascontiguousarray(row[None, :])
    if kind == "mask":
        return float(_eval(apos, np.ascontiguousarray(r > 0.5), lr, vs, ve, COMM)[0][0])
    sh = np.zeros(1)
    scaled_eval(apos, r, lr, vs, ve, COMM, sh)
    return float(sh[0])


def val_sharpe_new(apos, row, kind, lr, vs, ve):
    from fastbt.fast_grid import grid_val_sharpe
    from fastbt.s9 import COMM
    n = len(apos)
    out = np.zeros((1, 1))
    grid_val_sharpe(np.ascontiguousarray(apos[None, :], np.float64),
                    np.ascontiguousarray(lr[None, :n], np.float64), np.zeros(1, np.int64),
                    np.zeros(1, np.int64), np.array([vs]), np.array([ve]),
                    np.ascontiguousarray(row[None, None, None, :n], np.float64),
                    0 if kind == "mask" else 1, COMM, out)
    return float(out[0, 0])


def run(tickers, n, seed, with_dop, workers):
    from fastbt import fast_boot, fast_signals, report, s9, s9_stats
    from fastbt.data import year_bounds
    from fastbt.fast_grid import patched
    from fastbt.stats import sharpe_window
    from fastbt.turnover import tfilter
    t0 = time.time()
    import multiprocessing as mp
    import os
    os.environ["FASTBT_VPRED"] = str(s9.SOURCES["hyb_h1"][0])      # как s9.build_cache
    srcs = {lab: (s9.SOURCES[lab][2], s9.load_source(lab)) for lab in s9.SOURCES}
    jobs = [(t, {lab: (h, df[df["ticker"] == t]) for lab, (h, df) in srcs.items()})
            for t in tickers]
    nt_env = os.environ.get("NUMBA_NUM_THREADS")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    pool = mp.get_context("spawn").Pool(max(1, min(workers, len(jobs))))
    if nt_env is None:
        os.environ.pop("NUMBA_NUM_THREADS")
    else:
        os.environ["NUMBA_NUM_THREADS"] = nt_env
    old_cache = dict(pool.map(s9._job, jobs))          # старый путь: s9._job как есть
    pool.close()
    pool.join()
    new_cache = fast_signals.build(workers, tickers)
    print(f"кэши: {time.time() - t0:.0f} с", flush=True)
    labs = [l for l in ("hyb_h1", "hyb_h5", "harj_h1", "naive_h1") if l in s9.SOURCES]
    eo = s9.Engine(old_cache)
    xo = s9.S9(eo, 200)
    P_old = {lab: points(xo, lab, with_dop) for lab in labs}
    with patched():
        en = s9.Engine(new_cache)
        xn = s9.S9(en, 200)
        P_new = {lab: points(xn, lab, with_dop) for lab in labs}
    rng = np.random.default_rng(seed)
    recs = []
    for ti, t in enumerate(eo.tickers):
        assert en.tickers[ti] == t
        dates = eo.cache[t]["dates"]
        lr = eo.cache[t]["lr"]
        nt = len(dates)
        for ap in "ABCD":
            for _ in range(n):
                sid = s9.SIDS[rng.integers(len(s9.SIDS))]
                Y = s9.YEARS[rng.integers(len(s9.YEARS))]
                vs, ve = eo.vwin[(ti, Y)]
                ys, ye = year_bounds(dates, Y)[0], year_bounds(dates, Y + 1)[0]
                rec = dict(ticker=t, approach=ap, sid=sid, year=Y)
                if ap == "A":
                    h = [3, 5][rng.integers(2)]
                    nb = [0, 1, 3, 5, 10][rng.integers(5)]
                    rec["point"] = f"A0+tf(h={h},ban={nb})"
                    a_old = np.asarray(eo.cache[t]["A"][sid][Y], float)
                    a_new = np.asarray(en.cache[t]["A"][sid][Y], float)
                    po = tfilter(a_old, h, nb)
                    pn = tfilter(a_new, h, nb)
                    rec["pos_eq"] = bool(np.array_equal(po, pn) and np.array_equal(a_old, a_new))
                    one = np.ones(nt)
                    so = val_sharpe_old(po, one, "mask", lr, vs, ve)
                    sn = val_sharpe_new(pn, one, "mask", lr, vs, ve)
                else:
                    lab = labs[rng.integers(len(labs))]
                    k = rng.integers(len(P_old[lab][ap]))
                    name, Go, kind, cat = P_old[lab][ap][k]
                    name2, Gn, _, _ = P_new[lab][ap][k]
                    assert name == name2
                    rec["point"] = f"{name} на {lab}"
                    ro = Go[s9.CATS[sid], ti, :nt] if cat else Go[ti, :nt]
                    rn = Gn[s9.CATS[sid], ti, :nt] if cat else Gn[ti, :nt]
                    a = np.asarray(eo.cache[t]["A"][sid][Y], float)
                    po, pn = a * ro, a * rn
                    rec["pos_eq"] = bool(np.array_equal(np.nan_to_num(ro, nan=-7.0),
                                                        np.nan_to_num(rn, nan=-7.0)))
                    so = val_sharpe_old(a, ro, kind, lr, vs, ve)
                    sn = val_sharpe_new(a, rn, kind, lr, vs, ve)
                    # МСП по выбранным тикерам: портфель A·g, Шарп, p IID, p ЛВ
                    go, gn = eo.g_row(Go), en.g_row(Gn)
                    pto, ptn = eo.port(eo.A_row * go), en.port(en.A_row * gn)
                    rec["msp_sharpe_rel"] = _rel(s9.sharpe(ptn["excess"]), s9.sharpe(pto["excess"]))
                    bo = report.boot_delta(pto["excess"].values, xo.A["excess"].values, 1000)
                    bn = fast_boot.boot_delta(ptn["excess"].values, xn.A["excess"].values, 1000)
                    rec["p_iid_eq"] = bool(bo[1] == bn[1] and bo[0] == bn[0])
                    if len(recs) % 10 == 0:
                        lo = s9_stats.lw_sharpe_diff(pto["excess"].values, xo.A["excess"].values,
                                                     1000, 5)
                        ln = fast_boot.lw_sharpe_diff(ptn["excess"].values, xn.A["excess"].values,
                                                      1000, 5)
                        rec["p_lw_eq"] = bool(lo[2] == ln[2])
                        rec["lw_d_rel"] = _rel(ln[0], lo[0])
                rec["ret_diff"] = float(np.max(np.abs(pn[vs:ye] * lr[vs:ye] - po[vs:ye] * lr[vs:ye])))
                rec["sharpe_rel"] = _rel(sn, so)
                so_y = sharpe_window(np.asarray(po, float), lr, ys, ye, s9.COMM)
                sn_y = sharpe_window(np.asarray(pn, float), lr, ys, ye, s9.COMM)
                rec["sharpe_y_rel"] = _rel(sn_y, so_y)
                recs.append(rec)
    df = pd.DataFrame(recs)
    print(f"сверка: {time.time() - t0:.0f} с", flush=True)
    return df


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SBER,IRAO,MOEX,MTLR,AFLT")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma-h1", default="final3_full")
    p.add_argument("--c-dop", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    from fastbt import s9
    s9.set_scope("dev")
    s9.use_sigma_h1(a.sigma_h1)
    if a.c_dop:
        s9.load_c_dop()
    df = run(a.tickers.split(","), a.n, a.seed, a.c_dop, a.workers)
    if a.out:
        df.to_csv(a.out, index=False)
    for c in ("msp_sharpe_rel", "lw_d_rel"):
        if c not in df:
            df[c] = np.nan
    for c in ("p_iid_eq", "p_lw_eq"):
        if c not in df:
            df[c] = True
    bad = df[(~df.pos_eq) | (df.ret_diff > TOL) | (df.sharpe_rel > TOL) | (df.sharpe_y_rel > TOL)
             | (df.msp_sharpe_rel.fillna(0) > TOL) | (df.p_iid_eq.fillna(True) == False)  # noqa: E712
             | (df.p_lw_eq.fillna(True) == False)]  # noqa: E712
    print(df.groupby("approach").agg(n=("pos_eq", "size"), pos_eq=("pos_eq", "mean"),
                                     ret_max=("ret_diff", "max"), sh_rel_max=("sharpe_rel", "max"),
                                     shy_rel_max=("sharpe_y_rel", "max"),
                                     msp_rel_max=("msp_sharpe_rel", "max"),
                                     p_iid_eq=("p_iid_eq", lambda s: s.dropna().mean()),
                                     p_lw_eq=("p_lw_eq", lambda s: s.dropna().mean()),
                                     n_lw=("p_lw_eq", lambda s: int(s.notna().sum())))
          .to_string())
    print(f"тикеров {df.ticker.nunique()}, конфигураций {len(df)}; расхождений: {len(bad)}")
    if len(bad):
        print(bad.head(20).to_string(index=False))
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
