"""S5@v5 (Pivot Points) и S6@v5 (VWAP), daily — лист Л4 плана сетки v5 (Ж-127 п. 2).

S5: 5 типов пивота (classic, fibonacci, woodie, camarilla, demark; формулы Р
`strategies_walkforward.py:470-510` и `grid_scan_phase3.py:121-148` @6a7a4636) x период
den/nedelya/mesyac x уровень 1-3 x atr_filtr x fs (net, 2, 3, 4).
S6: okno x porog x dir_bias x fs (net, 1-4); ядро VWAP и фильтры fs те же, что у v4.

Пивот периода считается ТОЛЬКО по завершённому прошлому периоду (для den — по прошлому дню):
на баре текущего периода известна только прошлая группа. fs = "net" (Р): без наборов v4 и
без vf_pass; общие фильтры Р подключаются через `v5_filtr.primenit(df, tip, params)`
(ключ "filtr" в параметрах точки; нет ключа — фильтров нет).
Отличие от v4: сила сигнала S5 — по выбранному уровню (при уровне 1 равна v4).
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from fastbt.indicators import calc_sma
from fastbt.signals.interface import SignalSet, expand_grid, register_family
from fastbt.signals.s56 import _filters, _gid
from fastbt.signals.s56_kern import _entry, _s5_exit, _vwap

PERIODS = {"den": None, "nedelya": "weekly", "mesyac": "monthly"}
G5 = dict(pivot_type=["classic", "fibonacci", "woodie", "camarilla", "demark"],
          pivot_period=list(PERIODS), uroven=[1, 2, 3],
          atr_filtr=[0, 0.5, 1.0, 1.5, 2.0], filter_set=["net", 2, 3, 4])
G6 = dict(vwap_window=[3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100],
          dev_mult=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0],
          dir_bias=["OFF", "ON"], filter_set=["net", 1, 2, 3, 4])
TIP_FILTROV = "kontr"      # IS_TREND["S5"] = IS_TREND["S6"] = False (zh126_S5/S6.yaml)


def _group_ids(dates, period):
    """Группа периода; den — календарный день (на дневках = номер бара, на внутридневных ТФ — день, П5)."""
    if PERIODS[period] is None:
        return pd.DatetimeIndex(dates).normalize().asi8
    return _gid(dates, PERIODS[period])


def _levels(tip, h, l, c, o):
    """Уровни (P, S1, S2, S3, R1, R2, R3) из H/L/C/O завершённого периода (массивы по периодам)."""
    rng = h - l
    P = (h + l + c) / 3.0
    if tip == "classic":
        s1, r1, s2, r2 = 2 * P - h, 2 * P - l, P - rng, P + rng
        s3, r3 = l - 2 * (h - P), h + 2 * (P - l)
    elif tip == "fibonacci":
        s1, r1, s2, r2 = P - 0.382 * rng, P + 0.382 * rng, P - 0.618 * rng, P + 0.618 * rng
        s3, r3 = P - 1.000 * rng, P + 1.000 * rng
    elif tip == "woodie":
        P = (h + l + 2 * c) / 4.0
        s1, r1, s2, r2 = 2 * P - h, 2 * P - l, P - rng, P + rng
        s3, r3 = s2 - rng, r2 + rng
    elif tip == "camarilla":
        s1, r1, s2, r2 = c - 1.1 * rng / 12.0, c + 1.1 * rng / 12.0, c - 1.1 * rng / 6.0, c + 1.1 * rng / 6.0
        s3, r3 = c - 1.1 * rng / 4.0, c + 1.1 * rng / 4.0
    elif tip == "demark":
        x = np.where(c < o, h + 2 * l + c, np.where(c > o, 2 * h + l + c, h + l + 2 * c))
        P = x / 4.0
        s1, r1, s2, r2, s3, r3 = x / 2.0 - h, x / 2.0 - l, P - rng, P + rng, P - 2 * rng, P + 2 * rng
    else:
        raise ValueError(f"неизвестный тип пивота: {tip}")
    return P, s1, s2, s3, r1, r2, r3


def pivots(gid, o, h, l, c, tip):
    """Уровни по барам: бар периода g видит уровни периода g-1 (первый период — NaN)."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])
    ends = np.r_[starts[1:], len(gid)]
    G = np.maximum.reduceat(h, starts), np.minimum.reduceat(l, starts), c[ends - 1], o[starts]
    lev = _levels(tip, *G)
    out = []
    for a in lev:
        prev = np.r_[np.nan, a[:-1]]
        out.append(np.repeat(prev, ends - starts))
    return out        # P, S1, S2, S3, R1, R2, R3


def _ok_filtr(td, p, ok, lc, sc):
    """Общие фильтры Р (ключ "filtr") через v5_filtr.primenit(df, tip, params, sig); sig — сырой
    вход до фильтров (нужен RSI и mom). Нет ключа — без изменений."""
    if not p.get("filtr"):
        return ok
    F = importlib.import_module("fastbt.signals.v5_filtr")
    df = pd.DataFrame({"date": td.dates, "open": td.open, "high": td.high, "low": td.low,
                       "close": td.close, "volume": td.volume})
    sig0 = np.where(lc, 1, np.where(sc, -1, 0)).astype(np.int8)
    return ok & np.asarray(F.primenit(df, TIP_FILTROV, p["filtr"], sig=sig0), dtype=bool)


def _fs_mask(ind, fs, n):
    return np.ones(n, bool) if fs == "net" else _filters(ind, fs)


def _atr_ok(td, ind, S, R, f):
    """Расстояние до уровня > f*ATR (grid_scan_phase3.py:158-169); f = 0 — без фильтра."""
    if f <= 0:
        return np.ones(td.n, bool), np.ones(td.n, bool)
    atr = ind["atr14"]
    with np.errstate(invalid="ignore"):
        ok = ~np.isnan(atr) & (atr > 1e-12)
        return ok & (np.abs(td.close - S) > f * atr), ok & (np.abs(td.close - R) > f * atr)


def generate5(td, ind, grid):
    n, K = td.n, len(grid)
    sig = np.zeros((K, n), np.int8); ex = np.zeros((K, n), bool); st = np.zeros((K, n))
    extra = {k: np.zeros((K, n)) for k in ("P", "S1", "S2", "S3", "R1", "R2", "R3")}
    piv, flt = {}, {}
    for i, p in enumerate(grid):
        key = (p["pivot_period"], p["pivot_type"])
        if key not in piv:
            piv[key] = pivots(_group_ids(td.dates, key[0]), td.open, td.high, td.low, td.close, key[1])
        if p["filter_set"] not in flt:
            flt[p["filter_set"]] = _fs_mask(ind, p["filter_set"], n)
        P, S1, S2, S3, R1, R2, R3 = piv[key]
        S, R = {1: (S1, R1), 2: (S2, R2), 3: (S3, R3)}[p["uroven"]]
        with np.errstate(invalid="ignore"):
            s_ok, r_ok = _atr_ok(td, ind, S, R, p["atr_filtr"])
            lc, sc = (td.close < S) & s_ok, (td.close > R) & r_ok
            _entry(sig[i], lc, sc, _ok_filtr(td, p, flt[p["filter_set"]], lc, sc))
            span = np.abs(R - S)
            dist = np.minimum(np.abs(td.close - S), np.abs(td.close - R))
            st[i] = np.minimum(np.where(np.isnan(dist), 0.0, dist)
                               / np.where(np.isnan(span) | (span < 1e-12), 1e-12, span), 1.0)
        ex[i] = _s5_exit(td.close, P)
        for k, a in zip(("P", "S1", "S2", "S3", "R1", "R2", "R3"), piv[key]):
            extra[k][i] = a
    return SignalSet(sig, ex, ex.copy(), st, list(grid), False, extra)


def generate6(td, ind, grid, vwap_fn=_vwap):
    """vwap_fn(close, high, low, volume, w) -> (vwap, dev); на внутридневных ТФ — VWAP сессии (fastbt.tf_os_v5)."""
    n, K = td.n, len(grid)
    sig = np.zeros((K, n), np.int8); ex = np.zeros((K, n), bool); st = np.zeros((K, n))
    extra = {"vwap": np.zeros((K, n)), "dev": np.zeros((K, n))}
    vc, flt = {}, {}
    ma50, ma200 = calc_sma(td.close, 50), calc_sma(td.close, 200)   # dir_bias: build_daily_trend
    bull = ~np.isnan(ma50) & ~np.isnan(ma200) & (ma50 > ma200)
    bear = ~np.isnan(ma50) & ~np.isnan(ma200) & ~(ma50 > ma200)
    for i, p in enumerate(grid):
        w, m, fs = p["vwap_window"], p["dev_mult"], p["filter_set"]
        if w not in vc:
            vc[w] = vwap_fn(td.close, td.high, td.low, td.volume, w)
        if fs not in flt:
            flt[fs] = _fs_mask(ind, fs, n)
        vwap, dev = vc[w]
        with np.errstate(invalid="ignore"):
            lc = ~np.isnan(vwap) & (td.close < vwap - m * dev)
            sc = ~np.isnan(vwap) & (td.close > vwap + m * dev)
            if p["dir_bias"] == "ON":
                lc, sc = lc & bull, sc & bear
            _entry(sig[i], lc, sc, _ok_filtr(td, p, flt[fs], lc, sc))
            ex[i] = ~np.isnan(vwap) & (np.abs(td.close - vwap) < 0.5 * dev)
            st[i] = np.minimum(np.abs(td.close - np.where(np.isnan(vwap), td.close, vwap))
                               / (2.0 * np.maximum(dev, 1e-12)), 1.0)
        extra["vwap"][i], extra["dev"][i] = vwap, dev
    return SignalSet(sig, ex, ex.copy(), st, list(grid), False, extra)


def params5():
    return expand_grid(G5)


def params6():
    return expand_grid(G6)


register_family("S5", generate5, setka="v5")
register_family("S6", generate6, setka="v5")
