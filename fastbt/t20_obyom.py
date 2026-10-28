"""T20 — объём без движения (контракт #376 §2.6; толкование [Ж-065 п. 4]). Обе стороны.

U_d = vol_d / mean(vol_{d−20..d−1}) > v0 и |z_d| < r0. Сторона — sign(r_d), r_d = 0 — события нет. Вход close d+1 —
в `postroit_sig`. Объём в абсолютах. Сила = U_d (кратные объёма).

inp = {"dates", "tickers": {t: {"close", "vol", "sigma"}}}; osi = {"v0", "r0"}. Даты ≥ 2022 → PermissionError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt.t_obshchee import z_ryad

CUT = np.datetime64("2022-01-01")
OKNO = 20
SMA_SDVIG = 1                 # 0 — мутация: окно среднего включает день d


def vhod() -> dict:
    from fastbt import t_vhody
    return t_vhody.zagruzit()


def porog(osi: dict) -> float:
    """Порог веса в единицах силы (U_d): на пороге v0 множитель 1."""
    return float(osi["v0"])


def sobytiya(inp: dict, osi: dict) -> list:
    if len(inp["dates"]) and np.max(np.asarray(inp["dates"], "datetime64[D]")) >= CUT:
        raise PermissionError("T20: даты ≥ 2022")
    v0, r0 = float(osi["v0"]), float(osi["r0"])
    out = []
    for t in sorted(inp["tickers"]):
        x = inp["tickers"][t]
        c, vol = np.asarray(x["close"], float), np.asarray(x["vol"], float)
        z = z_ryad(c, x["sigma"])
        sma = pd.Series(vol).rolling(OKNO).mean().shift(SMA_SDVIG).to_numpy()
        r = np.r_[np.nan, np.diff(np.log(c))] if len(c) else c
        with np.errstate(invalid="ignore", divide="ignore"):
            U = np.where(sma > 0, vol / sma, np.nan)
        for d in np.flatnonzero((U > v0) & (np.abs(z) < r0) & (r != 0)):    # NaN → False
            out.append((t, int(d), 1 if r[d] > 0 else -1, float(U[d])))
    return out
