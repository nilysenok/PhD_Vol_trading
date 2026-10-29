"""Сетка v5, лист Л3: S3@v5 (okno × podtv) и S4@v5 (period × mnozh). Ж-127.

S3@v5: канал Дончиана[t-1] (rolling max high / min low за okno, сдвиг на 1 бар), вход по пробою close;
подтверждения podtv — формулы gen_s3_extended @6a7a4636 (grid_scan_phase3.py:62-117): none, close_confirm,
two_close, volume, adx_above_20, momentum. Зашитые в S3 v4 ADX > 20, объём > SMA20 и vf_pass здесь сняты:
фильтры по типу накладываются снаружи через v5_filtr.primenit(df, tip, params). Выходы и сила — как в v4.
S4@v5: те же сигнал, выходы и сила, что S4 v4 (s34.generate_s4); меняются только имена осей
(period = atr_period, mnozh = multiplier) и шире сетка (zh126_S4.yaml).
"""
from __future__ import annotations

import sys

import numpy as np
from numba import njit

from fastbt.signals.interface import WARMUP, Family, SignalSet, register_family
from fastbt.signals.s34 import generate_s4, rolling_ext

SETKA = "v5"
OKNO = [3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120]
PODTV = ["none", "close_confirm", "two_close", "volume", "adx_above_20", "momentum"]
PERIOD = [3, 5, 7, 10, 12, 14, 18, 20, 25, 30, 35, 40, 50, 60, 80]
MNOZH = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0]
S3_GRID_V5 = {"okno": OKNO, "podtv": PODTV}
S4_GRID_V5 = {"period": PERIOD, "mnozh": MNOZH}
_KOD = {n: i for i, n in enumerate(PODTV)}


@njit(cache=True)
def _s3_v5(high, low, close, adx, vs20, vol, w, kod, sig, exit_l, exit_s):
    """Канал по барам до t-1 включительно; подтверждение kod (индекс в PODTV)."""
    hc = rolling_ext(high, w, True)
    lc = rolling_ext(low, w, False)
    for i in range(1, len(close)):
        ph = hc[i - 1]; pl = lc[i - 1]
        exit_l[i] = (pl == pl) and close[i] < pl
        exit_s[i] = (ph == ph) and close[i] > ph
        if i < WARMUP or not (ph == ph):
            continue
        up = close[i] > ph
        dn = close[i] < pl
        if kod == 2:  # two_close: предыдущий бар тоже закрылся за каналом (канал t-2)
            if not (i >= 2 and hc[i - 2] == hc[i - 2]):
                continue
            up = up and close[i - 1] > hc[i - 2]
            dn = dn and close[i - 1] < lc[i - 2]
        elif kod == 3:  # volume
            vs = vs20[i] if (vs20[i] == vs20[i] and vs20[i] >= 1) else 1.0
            ok = vol[i] / vs > 1.0
            up = up and ok
            dn = dn and ok
        elif kod == 4:  # adx_above_20
            ok = (adx[i] == adx[i]) and adx[i] > 20
            up = up and ok
            dn = dn and ok
        elif kod == 5:  # momentum: close[t] - close[t-5]
            m = close[i] - close[i - 5]
            up = up and m > 0
            dn = dn and m < 0
        if up:
            sig[i] = 1
        elif dn:
            sig[i] = -1


def generate_s3_v5(td, ind, grid):
    k, t = len(grid), td.n
    sig = np.zeros((k, t), np.int8)
    el = np.zeros((k, t), np.bool_)
    es = np.zeros((k, t), np.bool_)
    adx = ind["adx14"]
    row = np.minimum(np.where(np.isnan(adx), 0.0, adx) / 35.0, 1.0)
    stg = np.empty((k, t))
    for j, p in enumerate(grid):
        _s3_v5(td.high, td.low, td.close, adx, ind["vol_sma20"], td.volume,
               int(p["okno"]), _KOD[p["podtv"]], sig[j], el[j], es[j])
        stg[j] = row
    return SignalSet(sig, el, es, stg, list(grid), split=True)


def generate_s4_v5(td, ind, grid):
    old = [{"atr_period": int(p["period"]), "multiplier": float(p["mnozh"])} for p in grid]
    ss = generate_s4(td, ind, old)
    return SignalSet(ss.sig, ss.exit_l, ss.exit_s, ss.strength, list(grid), split=True)


def _no_legacy(L, td, p):  # эталон v5 — формулы @6a7a4636, не запуск (Ж-127); сверка — tests/test_v5_s34.py
    raise NotImplementedError("legacy для v5 нет")


S3_V5 = Family("S3", "Trend", S3_GRID_V5, generate_s3_v5, _no_legacy, module=sys.modules[__name__])
S4_V5 = Family("S4", "Trend", S4_GRID_V5, generate_s4_v5, _no_legacy, module=sys.modules[__name__])
register_family("S3", generate_s3_v5, setka="v5")
register_family("S4", generate_s4_v5, setka="v5")
