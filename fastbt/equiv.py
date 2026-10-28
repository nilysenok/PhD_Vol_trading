"""Эквивалентность конфигураций fastbt и c22a4b3 (шаг 3): случайные конфигурации ×
тикеры × подходы; сравниваются дневные позиции (точно) и доходности pos·lr (1e-9),
а также Шарп отбора (calc_sharpe_v4) — относительная 1e-9.

python3 -m fastbt.equiv --tickers SBER,IRAO,MOEX,MTLR,GAZP --n 200 [--out csv]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from fastbt import grids
from fastbt.approach_b import HZ, _median, b_matrix
from fastbt.batch import trades_one
from fastbt.config import Config
from fastbt.data import year_bounds
from fastbt.execlayer import exec_layer
from fastbt.legacy import load_legacy
from fastbt.masks import c_masks, d_gates
from fastbt.stats import pos_from_trades, sharpe_window
from fastbt.unit import make_unit

SIDS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def _legacy_raw(L, u, i, rm):
    sig, ex = u.sigs[i], u.ss.exit_info(i)
    c, h, lo = u.close, u.high, u.low
    if u.sid == "S5" and rm["_rm_type"] == "pivot":
        x = u.ss.extra
        bd = L.s5r.precompute_pivot_base_distances(*(x[k][i] for k in ("P", "S1", "S2", "R1", "R2")))
        return L.v41._run_s5_bt(rm, sig, ex, c, h, lo, u.ind, bd, u.n)
    if u.sid == "S5":
        return L.v41._run_s5_bt(rm, sig, ex, c, h, lo, u.ind, None, u.n)
    if u.sid == "S6":
        x = u.ss.extra
        return L.v41._run_s6_bt(rm, sig, ex, c, h, lo, u.ind, x["vwap"][i], x["dev"][i], u.n)
    return L.sw.dispatch_bt_atr_v3(u.sid, sig, ex, c, h, lo, u.ind, rm, 200, u.n, 0)[0]


def _legacy_b(L, u, i, bp, sigma, med):
    sig, ex = u.sigs[i], u.ss.exit_info(i)
    eng = {k: v for k, v in bp.items() if k not in ("horizon", "ratio")}
    if "ratio" in bp:
        eng["k_tp"] = bp["k_sl"] * bp["ratio"]
    if u.sid in ("S5", "S6"):
        k_be = -1.0 if bp["k_be"] is None else bp["k_be"]
        gh = -1.0 if bp["gamma_hold"] is None else bp["gamma_hold"]
        return L.sw.bt_range_vpred_v3(sig, ex, u.close, u.high, u.low, sigma, u.adx, bp["k_sl"],
                                      eng["k_tp"], 30.0, 10, k_be, gh, med, 5, 200, u.n, 0)[0]
    return L.sw.dispatch_bt_vpred_v3(u.sid, sig, ex, u.close, u.high, u.low, u.ind, sigma, eng,
                                     200, u.n, False, med, 0)[0]


def _cmp(new, old, u, rec):
    rec["pos_eq"] = bool(np.array_equal(np.asarray(new) + 0.0, np.asarray(old) + 0.0))
    rec["ret_diff"] = float(np.max(np.abs(new * u.lr - old * u.lr)))
    return rec


def check(L, u, approach, rng, cfg):
    eg = grids.exec_grid()
    i, r, e = rng.integers(len(u.sigs)), rng.integers(len(u.rm)), rng.integers(len(eg))
    ep = eg[e]
    rec = {"ticker": u.ticker, "sid": u.sid, "approach": approach, "sp": i, "rm": r, "ep": e}
    te, tx, td = trades_one(u, i, u.rmP[r], u.trend)
    raw = pos_from_trades(te, tx, td, len(te), u.n)
    args = (ep["consec_entry"], ep["min_hold"], ep["cooldown_bars"], ep["min_strength"])
    pa = exec_layer(raw, u.strength[i], *args, u.n)
    if approach == "A":
        old = L.v4.execution_layer(_legacy_raw(L, u, i, u.rm[r]), u.strength[i], *args)
        new = pa
    elif approach == "B":
        grid, P = b_matrix(u.cat)
        q = rng.integers(len(grid))
        rec["bp"] = q
        sig3 = np.stack([u.td.sigma[h] for h in HZ])
        med3 = np.array([_median(s) for s in sig3])
        hs = HZ.index(grid[q]["horizon"])
        rh, rl = (u.rh[:1], u.rl[:1]) if u.trend else (u.rh, u.rl)
        from fastbt.approach_b import _ens_member
        ept = np.array(args, np.float64)
        bufs = (np.empty(u.n, np.int64), np.empty(u.n, np.int64), np.empty(u.n, np.int8))
        new = _ens_member(u.trend, i, P[q], u.sigs, u.exl, u.exs, u.nxt, u.close, u.high,
                          u.low, u.adx, u.atr, u.atr0, sig3, med3, u.sdl, u.sds, u.tdl, u.tds,
                          rh, rl, u.strength, ept, u.start, u.n, *bufs)
        old = L.v4.execution_layer(_legacy_b(L, u, i, grid[q], sig3[hs], med3[hs]),
                                   u.strength[i], *args)
    else:
        if approach == "C":
            rows, idx, params = c_masks(u.td, u.cat, u.sid in ("S1", "S2", "S5", "S6"))
            q = rng.integers(len(idx))
            p = params[q]
            m = L.sw._build_c_mask_v3(u.cat, u.td.sigma[p["horizon"]], p["lookback"], p["thr"],
                                      p["direction"], 5, u.td.sigma["h22"], u.td.sigma["h1"],
                                      p["term_filter"], p["hysteresis"], u.n)
            old = pa.copy()
            old[~m] = 0.0
        else:
            rows, idx, params = d_gates(u.td, False)
            q = rng.integers(len(idx))
            dp = dict(params[q])
            sc = L.sw._compute_d_scale_v3(u.td.sigma[dp["horizon"]], dp)
            old = pa * (sc >= dp["gate_threshold"]).astype(np.float64)
        rec["mask"] = int(q)
        new = pa * rows[idx[q]]
    _cmp(new, old, u, rec)
    vs, ve = year_bounds(u.dates, 2015)[1], year_bounds(u.dates, 2025)[1]
    vs = max(vs, 200)
    mask = np.zeros(u.n, bool)
    mask[vs:ve] = True
    so = L.v4.calc_sharpe_v4(old, u.lr, mask, np.sqrt(252), cfg.comm)
    sn = sharpe_window(np.asarray(new, np.float64), u.lr, vs, ve, cfg.comm)
    rec["sharpe_rel"] = float(abs(sn - so) / max(abs(so), 1e-12)) if so != 0 else abs(sn)
    return rec


def run(tickers, n, seed=0, sids=SIDS):
    L = load_legacy()
    cfg = Config.gate()
    rng = np.random.default_rng(seed)
    recs = []
    for tk in tickers:
        units = {s: make_unit(tk, s) for s in sids}
        for ap in "ABCD":
            for _ in range(n):
                recs.append(check(L, units[sids[rng.integers(len(sids))]], ap, rng, cfg))
    return pd.DataFrame(recs)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SBER,IRAO,MOEX,MTLR,AFLT")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    df = run(a.tickers.split(","), a.n, a.seed)
    if a.out:
        df.to_csv(a.out, index=False)
    bad = df[(~df.pos_eq) | (df.ret_diff > 1e-9) | (df.sharpe_rel > 1e-9)]
    print(df.groupby("approach").agg(n=("pos_eq", "size"), pos_eq=("pos_eq", "mean"),
                                     ret_max=("ret_diff", "max"),
                                     sh_rel_max=("sharpe_rel", "max")).to_string())
    print(f"расхождений: {len(bad)} из {len(df)}")
    if len(bad):
        print(bad.head(20).to_string(index=False))
    return 0 if len(bad) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
