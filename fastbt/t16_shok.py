"""T16 — пост-шоковый возврат (контракт #376 §2.2; толкование [Ж-065 п. 4]). Лонг.

Событие d: z_d ≤ −z0 и |Z_d| ≤ m0 (шок бумаги без шока рынка). Вход close d+1 (это делает `postroit_sig`, lag = 1).
Сила = |z_d| (в σ). Только данные, известные к 18:50 d: z_d = r_d/σ̂_{d−1}, Z_d = R_d/sig_mkt_{d−1}.

inp = {"dates": datetime64[D], "tickers": {t: {"close", "sigma"}}, "imoex": {"close", "sigma"}} — ряды одной длины;
osi = {"z0", "m0"}. Событие — (тикер, индекс d, +1, сила). Даты ≥ 2022 → PermissionError. Модуль файлов не открывает.
"""
from __future__ import annotations

import numpy as np

from fastbt.t_obshchee import Z_ryad, z_ryad

CUT = np.datetime64("2022-01-01")
FILTR_IMOEX = True            # False — мутация: IMOEX вне ±m0 не отсекается


def sobytiya(inp: dict, osi: dict) -> list:
    if len(inp["dates"]) and np.max(np.asarray(inp["dates"], "datetime64[D]")) >= CUT:
        raise PermissionError("T16: даты ≥ 2022")
    z0, m0 = float(osi["z0"]), float(osi["m0"])
    Z = Z_ryad(inp["imoex"]["close"], inp["imoex"]["sigma"])
    ok_mkt = np.abs(Z) <= m0 if FILTR_IMOEX else np.ones(len(Z), bool)       # NaN → False: нет данных, нет события
    out = []
    for t in sorted(inp["tickers"]):
        x = inp["tickers"][t]
        z = z_ryad(x["close"], x["sigma"])
        for d in np.flatnonzero((z <= -z0) & ok_mkt):
            out.append((t, int(d), 1, float(-z[d])))
    return out
