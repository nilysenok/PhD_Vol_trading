"""Поставщик сигналов для ядра: семейство fastbt (если есть) или старый код c22a4b3.

get_signals(sid, td, ind) -> SignalSet с кэшем на диске. FASTBT_SIGNALS=legacy —
принудительно старый код (эталон для сквозной эквивалентности ядра).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

from fastbt import cache
from fastbt.data import daily_df
from fastbt.signals.interface import SignalSet, expand_grid

S5_SIG = dict(pivot_period=["weekly", "monthly"], pivot_type=["classic", "fibonacci"],
              filter_set=[2, 3, 4])
S6_SIG = dict(vwap_window=[5, 10, 20, 60], dev_mult=[0.5, 1.0, 1.5, 2.0],
              filter_set=[1, 2, 3, 4], dir_bias=["OFF", "ON"])
SIG_GRIDS = {"S1": dict(ma_window=[15, 20, 25], z_entry=[1.5, 2.0, 2.5], z_exit=[0.3, 0.5, 0.7]),
             "S2": dict(bb_window=[15, 20, 25], bb_std=[1.5, 2.0, 2.5]),
             "S3": dict(dc_window=[10, 15, 20, 25]),
             "S4": dict(atr_period=[10, 14, 20], multiplier=[2.0, 2.5, 3.0, 3.5]),
             "S5": S5_SIG, "S6": S6_SIG}


def _family_available(sid: str) -> bool:
    if os.environ.get("FASTBT_SIGNALS", "") == "legacy":
        return False
    return importlib.util.find_spec(f"fastbt.signals.{sid.lower()}") is not None


def _pack(rows, grid, split, extra_rows):
    sig = np.stack([r[0] for r in rows]).astype(np.int8)
    el = np.stack([r[1] for r in rows]).astype(bool)
    es = np.stack([r[2] for r in rows]).astype(bool)
    st = np.stack([r[3] for r in rows]).astype(np.float64)
    extra = {k: np.stack([e[k] for e in extra_rows]) for k in (extra_rows[0] if extra_rows else {})}
    return SignalSet(sig, el, es, st, grid, split, extra)


def legacy_signals(sid, td, ind):
    """Сигналы старым кодом (SW/V4 для S1–S4, V41 для S5/S6)."""
    from fastbt.legacy import load_legacy
    L = load_legacy()
    c, h, l, v = td.close, td.high, td.low, td.volume
    grid = expand_grid(SIG_GRIDS[sid])
    rows, extras = [], []
    if sid in ("S1", "S2", "S3", "S4"):
        sw = L.sw
        sma = sw.precompute_sma_cache(c, [15, 20, 25])
        dc = sw.precompute_donchian_cache(h, l, [10, 15, 20, 25])
        st = sw.precompute_supertrend_cache(h, l, c, [10, 14, 20], [2.0, 2.5, 3.0, 3.5])
        vw = sw.precompute_vwap_cache(c, h, l, v, [10, 15, 20])
        piv = {"classic": sw.calc_pivot_daily(h, l, c, "classic")}
        for sp in grid:
            s, ex = sw.dispatch_signals(sid, c, h, l, v, ind, sma, dc, st, vw, piv, None, sp)
            stg = L.v4.dispatch_strength(sid, c, ind, sma, dc, st, vw, piv, sp)
            el, es = ex if isinstance(ex, tuple) else (ex, ex)
            rows.append((s, el, es, stg))
        return _pack(rows, grid, sid in ("S3", "S4"), [])
    tdf = daily_df()
    tdf = tdf[tdf["ticker"] == td.ticker].sort_values("date").reset_index(drop=True)
    if sid == "S5":
        pc = L.v41.prepare_pivot_cache(daily_df(), td.ticker, None, False, tdf)
        for sp in grid:
            P, S1, S2, R1, R2 = pc[(sp["pivot_period"], sp["pivot_type"])]
            s, ex = L.s5r.gen_s5_signals_v2(c, ind, P, S1, R1, sp["filter_set"])
            span = np.abs(R1 - S1)
            dist = np.minimum(np.abs(c - S1), np.abs(c - R1))
            span_safe = np.where(np.isnan(span) | (span < 1e-12), 1e-12, span)
            dist_safe = np.where(np.isnan(dist), 0.0, dist)
            rows.append((s, ex, ex, np.minimum(dist_safe / span_safe, 1.0)))
            extras.append(dict(P=P, S1=S1, S2=S2, R1=R1, R2=R2))
        return _pack(rows, grid, False, extras)
    vc, _, dba = L.v41.prepare_vwap_cache_ext(c, h, l, v, False, tdf, daily_df(), td.ticker)
    for sp in grid:
        vwap, dev = vc[sp["vwap_window"]]
        db = dba if sp["dir_bias"] == "ON" else None
        s, ex = L.s6r.gen_s6_signals_v2(c, ind, vwap, dev, sp["dev_mult"], sp["filter_set"], db)
        dev_safe = np.maximum(dev, 1e-12)
        vwap_safe = np.where(np.isnan(vwap), c, vwap)
        rows.append((s, ex, ex, np.minimum(np.abs(c - vwap_safe) / (2.0 * dev_safe), 1.0)))
        extras.append(dict(vwap=vwap, dev=dev))
    return _pack(rows, grid, False, extras)


def get_signals(sid, td, ind) -> SignalSet:
    src = "fam" if _family_available(sid) else "legacy"
    k = cache.key(f"sig_{sid}_{td.ticker}_{src}", td.digest())
    z = cache.load(k)
    grid = expand_grid(SIG_GRIDS[sid])
    if z is not None:
        extra = {n[2:]: z[n] for n in z if n.startswith("x_")}
        return SignalSet(z["sig"], z["el"], z["es"], z["st"], grid, bool(z["split"]), extra)
    if src == "fam":
        from fastbt.signals.interface import get_family
        ss = get_family(sid).generate(td, ind, grid)
    else:
        ss = legacy_signals(sid, td, ind)
    arr = dict(sig=ss.sig, el=ss.exit_l, es=ss.exit_s, st=ss.strength, split=np.array(ss.split))
    arr.update({f"x_{n}": a for n, a in ss.extra.items()})
    cache.save(k, arr)
    return ss
