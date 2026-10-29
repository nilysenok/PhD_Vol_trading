"""Семьи S1@v5 (MeanRev) и S2@v5 (Bollinger) сетки v5 (Ж-127, лист Л2).

S1: ma_tip (7, fastbt.ma_tipy) × ma_window × dev_tip (std | atr) × z_entry × z_exit (число | "net").
    z = (close − MA) / dev, dev = СКО окна (calc_std, ddof=1, вокруг SMA при любом ma_tip) или ATR14;
    z_exit = "net": сигнального выхода нет, выход только по РМ (семантика Р).
S2: ma_tip × bb_window × bb_std × podtv (none | rsi30 | rsi20 | volume | close_confirm).
    Формулы подтверждений — apply_confirmation @6a7a4636 (grid_scan_test.py:285-300), RSI — rsi_wilder.
    close_confirm в контртренде — тождество: сигнал уже по закрытию (в источнике close_confirm есть только у S3,
    grid_scan_phase3.py:78-82, и там совпадает с none).
Ядро сигналов — _s1/_s2 из s12.py (v4) без зашитых фильтров входа: фильтры v5 — через v5_filtr.primenit
(ключ "filtr" в точке; если его нет — выключены). Значение на t зависит от данных <= t; исполнение >= t+1 (Д2) — не здесь.
"""
from __future__ import annotations

import numpy as np

from fastbt.filtry_v5 import rsi_wilder
from fastbt.indicators import calc_std
from fastbt.ma_tipy import TIPY, ma as _ma_tipy
from fastbt.signals import interface as _iface
from fastbt.signals.interface import SignalSet
from fastbt.signals.s12 import _s1, _s2

PODTV = ("none", "rsi30", "rsi20", "volume", "close_confirm")
DEV_TIPY = ("std", "atr")
FILTR_TIP = "kontrtrend"   # S1, S2 — контртренд (IS_TREND = False)


def _ma(close, tip, n):
    return _ma_tipy(close, tip, n)


def podtverzhdenie(sig, tip, rsi, volume, vol_sma20):
    """apply_confirmation(is_trend=False) из 6a7a4636. Возвращает копию."""
    if tip not in PODTV:
        raise ValueError(f"неизвестное подтверждение: {tip!r}")
    out = sig.copy()
    if tip == "rsi30":
        out[(sig == 1) & (np.isnan(rsi) | (rsi > 30))] = 0
        out[(sig == -1) & (np.isnan(rsi) | (rsi < 70))] = 0
    elif tip == "rsi20":
        out[(sig == 1) & (np.isnan(rsi) | (rsi > 20))] = 0
        out[(sig == -1) & (np.isnan(rsi) | (rsi < 80))] = 0
    elif tip == "volume":
        vs = np.where(np.isnan(vol_sma20) | (vol_sma20 < 1), 1.0, vol_sma20)
        out[volume / vs < 1.0] = 0
    return out


def _filtr_maski(td, p, T):
    """(long, short) bool[T] из v5_filtr.primenit; без ключа "filtr" — всё разрешено."""
    ok = np.ones(T, bool)
    if not p.get("filtr"):
        return ok, ok
    import pandas as pd
    from fastbt.signals.v5_filtr import primenit
    df = pd.DataFrame({"open": td.open, "high": td.high, "low": td.low, "close": td.close,
                       "volume": td.volume}, index=pd.to_datetime(td.dates))
    m = primenit(df, FILTR_TIP, p["filtr"])
    fl, fs = m if isinstance(m, tuple) else (m, m)
    return np.asarray(fl, bool), np.asarray(fs, bool)


def _dev(td, ind, p, w, cache):
    if p.get("dev_tip", "std") == "atr":
        return ind["atr14"]
    if ("std", w) not in cache:
        cache[("std", w)] = calc_std(td.close, w)
    return cache[("std", w)]


def _ma_kesh(td, tip, w, cache):
    if (tip, w) not in cache:
        cache[(tip, w)] = _ma(td.close, tip, w)
    return cache[(tip, w)]


def _generate(kind):
    wkey = "ma_window" if kind == 1 else "bb_window"

    def generate(td, ind, grid):
        K, T = len(grid), td.n
        sig = np.zeros((K, T), np.int8)
        ex = np.zeros((K, T), np.bool_)
        st = np.zeros((K, T))
        cache = {}
        rsi = rsi_wilder(td.close) if kind == 2 else None
        for i, p in enumerate(grid):
            w = p[wkey]
            ma = _ma_kesh(td, p.get("ma_tip", "sma"), w, cache)
            dev = _dev(td, ind, p, w, cache)
            fl, fs = _filtr_maski(td, p, T)
            if kind == 1:
                zx = p["z_exit"]
                zx = -1.0 if zx == "net" else float(zx)   # |z| < −1 никогда: выхода по сигналу нет
                _s1(td.close, ma, dev, fl, fs, p["z_entry"], zx, sig[i], ex[i], st[i])
            else:
                _s2(td.close, ma, dev, fl, fs, p["bb_std"], sig[i], ex[i], st[i])
                sig[i] = podtverzhdenie(sig[i], p.get("podtv", "none"), rsi, td.volume, ind["vol_sma20"])
        return SignalSet(sig, ex, ex, st, list(grid), split=False)
    return generate


generate_s1 = _generate(1)
generate_s2 = _generate(2)

# Реестр с ключом setka — лист Л1 (interface.register_family). Пока его нет в базе, регистрация пропускается.
_reg = getattr(_iface, "register_family", None)
if _reg is not None:
    _reg("S1", generate_s1, setka="v5")
    _reg("S2", generate_s2, setka="v5")

__all__ = ["TIPY", "PODTV", "DEV_TIPY", "generate_s1", "generate_s2", "podtverzhdenie"]
