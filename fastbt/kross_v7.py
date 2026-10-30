"""Л1.1 (Ж-146): панель скоров сечения S7–S9 для интерфейса сигнала. Скоры и книги — fastbt/kross_seksiya (только импорт).

panel — скор [T, N] по sid; rebal — решение каждый день или в последний торговый день недели; storona — стороны ±1/0
с порогом и гистерезисом оборота (bufer=0 даёт тот же узор, что vesa); placebo_sechenie — перестановка тикеров внутри
дня; k2 — циклический сдвиг (kross_seksiya.k2_skor). Решение дня d использует данные ≤ d; исполнение d+1 — в оценщике.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import kross_seksiya as ks

REZHIMY = ("den", "nedelya")


def panel(c: pd.DataFrame, imoex: pd.Series, metki: np.ndarray, sid: str) -> np.ndarray:
    """Скор [T, N] ∈ [−1; 1] (NaN — входа нет). c — закрытия 18:50, imoex — ряд на тех же датах, metki — сектор [T, N]."""
    if sid == "S7":
        return ks.skor_s7(c, imoex).to_numpy()
    if sid == "S8":
        return ks.skor_s8(c).to_numpy()
    if sid == "S9":
        return ks.skor_s9(c, imoex, metki).to_numpy()
    raise ValueError(f"sid {sid!r}: панель есть у S7, S8, S9")


def rebal(skor: np.ndarray, daty, rezhim: str) -> np.ndarray:
    """den — как есть. nedelya — решение в последний торговый день недели (по данным ≤ него), держится до следующего;
    до первого такого дня решения нет (NaN)."""
    if rezhim not in REZHIMY:
        raise ValueError(f"rezhim {rezhim!r} не из {REZHIMY}")
    if len(daty) != len(skor):
        raise ValueError("daty и skor разной длины")
    if rezhim == "den" or len(skor) == 0:
        return np.array(skor, dtype=float)
    iso = pd.DatetimeIndex(daty).isocalendar()
    wk = (iso["year"].to_numpy().astype(np.int64) * 100 + iso["week"].to_numpy()).astype(np.int64)
    posl = np.append(wk[1:] != wk[:-1], True)                       # день, после которого неделя меняется (или лента кончилась)
    src = np.maximum.accumulate(np.where(posl, np.arange(len(wk)), -1))
    out = np.asarray(skor, float)[np.maximum(src, 0)]
    out[src < 0] = np.nan
    return out


def storona(skor, sigma, metki, ok, porog: float = ks.POROG, bufer: float = 0.0, cap: int = ks.V_SEKTORE) -> np.ndarray:
    """sig[T, N] ∈ {−1, 0, 1} int8. Вход при |скор| ≥ porog; позиция держится, пока знак скора тот же и |скор| ≥ porog − bufer.
    ≤ cap ставок одной стороны в секторе (по |скор|); нет ног с одной из сторон — строка нулевая (как kross_seksiya.vesa)."""
    if not 0 <= bufer < porog:
        raise ValueError("bufer должен быть в [0; porog)")
    sk = np.asarray(skor, float)
    sig = np.zeros(sk.shape, np.int8)
    prev = np.zeros(sk.shape[1], np.int8)
    for t in range(len(sk)):
        s = sk[t]
        good = np.isfinite(s) & np.isfinite(sigma[t]) & (sigma[t] > 0) & ok[t]
        row, side = np.zeros(sk.shape[1], np.int8), {}
        for sg in (1, -1):
            sgs = np.where(good, sg * np.where(good, s, 0.0), -np.inf)
            idx = [i for i in np.flatnonzero((sgs >= porog) | ((prev == sg) & (sgs >= porog - bufer)))]
            idx.sort(key=lambda i: -abs(s[i]))
            used, keep = {}, []
            for i in idx:
                if used.get(metki[t, i], 0) < cap:
                    used[metki[t, i]] = used.get(metki[t, i], 0) + 1
                    keep.append(i)
            side[sg] = keep
        if side[1] and side[-1]:
            row[side[1]], row[side[-1]] = 1, -1
        sig[t], prev = row, row
    return sig


def placebo_sechenie(skor: np.ndarray, seed: int) -> np.ndarray:
    """Плацебо: в каждый день значения скора переставлены между тикерами случайно (множество значений дня и число ног целы)."""
    sk = np.asarray(skor, float)
    rng = np.random.default_rng(seed)
    return np.stack([row[rng.permutation(len(row))] for row in sk]) if len(sk) else sk.copy()


def k2(skor: np.ndarray, lag: int) -> np.ndarray:
    """Контроль к2 панели: циклический сдвиг по времени; lag=0 — исходный ряд."""
    return ks.k2_skor(skor, lag)
