"""Обёртка сигналов v6 над семьями v5 (Ж-132 п. 3). Лист V6-2: ось dlit ∈ {1, 2, 3} (К2).

dlit = k: вход стороны s в день d разрешён, только если УСЛОВИЕ ВХОДА семьи (а не состояние позиции и не
счётчик consec фазы 3) выполнено для стороны s в каждый из дней d−k+1…d. k = 1 — сигнал v5 побайтно.
Условие — сырое: без фильтров v5_filtr, подтверждений podtv, dir_bias и fs (они остаются в sig семьи).
Условия: S1 z < −z_entry / z > z_entry; S2 close за полосой; S3 close за каналом Дончиана[t−1];
S4 close по сторону линии Supertrend; S5 close за уровнем пивота (с atr_filtr); S6 close за VWAP ± dev_mult·dev.
S3/S4 (тренд): при vhod_tip = stop_zayavka заявка стоит до пробоя, «k дней» не определено — dlit свёртывается
в канон (sig без изменений); действует только при vhod_tip = podtverzhdenie (И9; по умолчанию stop_zayavka).
S4: сигнал v5 — событие пересечения, а не состояние (при буквальном «условие k дней» после пересечения
сигнал при k ≥ 2 был бы пуст всегда). Поэтому при k ≥ 2 сигнал ставится в день, когда серия «close по сторону
линии» после пересечения достигла ровно k (ADX > 20 и vf_pass — как в v5, в этот день); k = 1 — пересечение v5.
Значение на d зависит только от данных <= d; исполнение >= d+1 (Д2) — не здесь.
"""
from __future__ import annotations

import numpy as np

from fastbt.signals.interface import WARMUP
from fastbt.signals.s34 import rolling_ext, supertrend
from fastbt.signals.s56_kern import _vwap
from fastbt.signals.v5_s12 import _dev, _ma_kesh
from fastbt.signals.v5_s56 import _atr_ok, _group_ids, pivots

DLIT = (1, 2, 3)
TREND = ("S3", "S4")


def _proverit(k):
    if k not in DLIT:
        raise ValueError(f"dlit вне {DLIT}: {k!r}")


def dlina_serii(cond, start=None):
    """run[t] = число подряд идущих дней …t с истинным cond; если задан start — серия начинается только в дне start."""
    run = np.zeros(len(cond), np.int32)
    for t in range(len(cond)):
        if not cond[t]:
            continue
        if t and run[t - 1] > 0:
            run[t] = run[t - 1] + 1
        elif start is None or start[t]:
            run[t] = 1
    return run


def maska_dlit(cond_l, cond_s, k):
    """(ok_l, ok_s): условие стороны держится k дней подряд до дня d включительно."""
    _proverit(k)
    return dlina_serii(cond_l) >= k, dlina_serii(cond_s) >= k


def _cmp(a, b, op):
    with np.errstate(invalid="ignore"):
        return op(a, b) & ~np.isnan(a) & ~np.isnan(b)


def _usl_s1(td, ind, p, cache):
    w = p["ma_window"]
    ma, dev = _ma_kesh(td, p.get("ma_tip", "sma"), w, cache), _dev(td, ind, p, w, cache)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (td.close - ma) / dev
    ze = np.full_like(z, p["z_entry"])
    return _cmp(z, -ze, np.less), _cmp(z, ze, np.greater)


def _usl_s2(td, ind, p, cache):
    w = p["bb_window"]
    ma, dev = _ma_kesh(td, p.get("ma_tip", "sma"), w, cache), _dev(td, ind, p, w, cache)
    return (_cmp(td.close, ma - p["bb_std"] * dev, np.less_equal),
            _cmp(td.close, ma + p["bb_std"] * dev, np.greater_equal))


def _usl_s3(td, ind, p, cache):
    w = int(p["okno"])
    ph = np.r_[np.nan, rolling_ext(td.high, w, True)[:-1]]
    pl = np.r_[np.nan, rolling_ext(td.low, w, False)[:-1]]
    return _cmp(td.close, ph, np.greater), _cmp(td.close, pl, np.less)


def _st(td, p):
    return supertrend(td.high, td.low, td.close, int(p["period"]), float(p["mnozh"]))


def _usl_s4(td, ind, p, cache):
    st = _st(td, p)
    return _cmp(td.close, st, np.greater), _cmp(td.close, st, np.less)


def _usl_s5(td, ind, p, cache):
    lev = pivots(_group_ids(td.dates, p["pivot_period"]), td.open, td.high, td.low, td.close, p["pivot_type"])
    S, R = {1: (lev[1], lev[4]), 2: (lev[2], lev[5]), 3: (lev[3], lev[6])}[p["uroven"]]
    s_ok, r_ok = _atr_ok(td, ind, S, R, p["atr_filtr"])
    return _cmp(td.close, S, np.less) & s_ok, _cmp(td.close, R, np.greater) & r_ok


def _usl_s6(td, ind, p, cache):
    vwap, dev = _vwap(td.close, td.high, td.low, td.volume, p["vwap_window"])
    m = p["dev_mult"]
    return _cmp(td.close, vwap - m * dev, np.less), _cmp(td.close, vwap + m * dev, np.greater)


_USL = {"S1": _usl_s1, "S2": _usl_s2, "S3": _usl_s3, "S4": _usl_s4, "S5": _usl_s5, "S6": _usl_s6}


def usloviya(sid, td, ind, p, cache=None):
    """Сырое условие входа семьи для точки p: (cond_long, cond_short), bool[T]; NaN → ложь."""
    return _USL[sid](td, ind, p, {} if cache is None else cache)


def _s4_sobytie(td, ind, p, k):
    """S4: сигнал в день, когда серия после пересечения достигла ровно k; фильтры v5 в этот день."""
    st = _st(td, p)
    valid = ~np.isnan(st)
    adx = ind["adx14"]
    ok = np.zeros(td.n, bool)
    ok[WARMUP:] = (~np.isnan(adx) & (adx > 20) & ind["vf_pass"])[WARMUP:]
    out = np.zeros(td.n, np.int8)
    for side, cond in ((1, _cmp(td.close, st, np.greater)), (-1, _cmp(td.close, st, np.less))):
        prev = np.r_[False, cond[:-1]]
        start = cond & np.r_[False, valid[:-1]] & ~prev    # пересечение: прошлый день с линией, но не по эту сторону
        out[(dlina_serii(cond, start) == k) & ok & (out == 0)] = side
    return out


def primenit_dlit(sid, td, ind, p, sig, k, cache=None):
    """sig (int8[T]) — сигнал точки v5; возвращает сигнал с осью dlit = k (k = 1 — v5 побайтно)."""
    _proverit(k)
    if sid in TREND and p.get("vhod_tip", "stop_zayavka") != "podtverzhdenie":
        return sig.copy()
    if sid == "S4":
        return _s4_sobytie(td, ind, p, k)
    ok_l, ok_s = maska_dlit(*usloviya(sid, td, ind, p, cache), k)
    out = sig.copy()
    out[(sig == 1) & ~ok_l] = 0
    out[(sig == -1) & ~ok_s] = 0
    return out
