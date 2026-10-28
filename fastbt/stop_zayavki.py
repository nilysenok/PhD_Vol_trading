"""Ж-091 пп. 1.2–1.3: уровни стоп-заявок S3 (Donchian) и S4 (Supertrend) по дням — ось `vhod_tip = stop_zayavka`.

Уровни дня d считаются только по барам ≤ d (окно включает d) и действуют на день d+1 (заявка активна с первого бара d+1).
 - up[d] — buy-stop входа, dn[d] — sell-stop входа (nan — заявки нет); xl[d] / xs[d] — стоп выхода лонга / шорта.
 - S3: up = max(high, w), dn = min(low, w); выход — обратная граница канала (лонг — dn, шорт — up).
 - S4: стоп-заявка на линии ST дня d: buy-stop, если ST выше close (нисходящий тренд), sell-stop, если ниже; выход — ST.
Те же функции канала и ST, те же фильтры (ADX14 > 20, vf_pass, у S3 ещё объём > SMA20, прогрев), что в signals/s34.
Подтверждения закрытием (close пробил канал / пересёк ST) нет — его заменяет касание заявки.
"""
from __future__ import annotations

import numpy as np

from fastbt.signals.interface import WARMUP
from fastbt.signals.s34 import rolling_ext, supertrend

NAN = float("nan")


def _gate_s3(td, ind) -> np.ndarray:
    vs, adx = np.asarray(ind["vol_sma20"], float), np.asarray(ind["adx14"], float)
    ok = np.isfinite(vs) & (vs > 0)
    f_vc = td.volume / np.where(ok, vs, 1.0) > 1.0
    return np.isfinite(adx) & (adx > 20) & f_vc & np.asarray(ind["vf_pass"], bool)


def urovni(sid: str, sp: dict, td, ind, storona: str = "obe") -> dict:
    """{up, dn, xl, xs}: массивы длины td.n. Фильтры сигнала — в up/dn (nan, если фильтр не пройден); выходы — всегда."""
    n, c = len(td.close), np.asarray(td.close, float)
    fin = np.arange(n) >= WARMUP
    if sid == "S3":
        hc, lc = rolling_ext(td.high, int(sp["dc_window"]), True), rolling_ext(td.low, int(sp["dc_window"]), False)
        up, dn, xl, xs = hc, lc, lc, hc
        gate = _gate_s3(td, ind)
    elif sid == "S4":
        st = supertrend(td.high, td.low, td.close, int(sp["atr_period"]), float(sp["multiplier"]))
        up, dn, xl, xs = np.where(st > c, st, NAN), np.where(st < c, st, NAN), st, st
        gate = np.isfinite(np.asarray(ind["adx14"], float)) & (np.asarray(ind["adx14"], float) > 20) & np.asarray(ind["vf_pass"], bool)
    else:
        raise ValueError(f"{sid}: ось vhod_tip только у S3 и S4")
    ok = fin & gate
    return dict(up=np.where(ok, up, NAN), dn=np.where(ok & (storona != "long"), dn, NAN),     # long — без sell-stop входа
                xl=np.asarray(xl, float), xs=np.asarray(xs, float))


def okno(u: dict, s0: int, b: int, allow) -> dict:
    """Строки [s0; s0+b] (как sig оценщика); allow — разрешение входа по close d (события, режим, фильтр): запрещено — заявки нет."""
    sl, al = slice(s0, s0 + b + 1), np.asarray(allow, bool)
    if len(al) != b + 1:
        raise ValueError("allow и окно разной длины")
    return dict(up=np.where(al, u["up"][sl], NAN), dn=np.where(al, u["dn"][sl], NAN), xl=u["xl"][sl], xs=u["xs"][sl])
