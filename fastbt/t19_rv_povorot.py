"""T19 — переворот RV (контракт #376 §2.5; толкование [Ж-065 п. 4]). Лонг.

V_c = (σ_rv5,c / σ_rv66,c)² > q (отношение дисперсий). Вход — первый d ∈ (c, c+10], где σ_rv5,d < σ_rv5,d−1
(«первое снижение», только прошлое). Одно событие на день d: при нескольких c берётся ранний c. Вход close d+1 —
в `postroit_sig`. Сила = V_c (отношение дисперсий).

inp = {"dates", "tickers": {t: {"sigma5", "sigma66"}}}; osi = {"q"}. Даты ≥ 2022 → PermissionError.
"""
from __future__ import annotations

import numpy as np

CUT = np.datetime64("2022-01-01")
OKNO = 10
VPERED = 0                    # 1 — мутация: снижение по σ_rv5,d+1 < σ_rv5,d


def sobytiya(inp: dict, osi: dict) -> list:
    if len(inp["dates"]) and np.max(np.asarray(inp["dates"], "datetime64[D]")) >= CUT:
        raise PermissionError("T19: даты ≥ 2022")
    q = float(osi["q"])
    out = []
    for t in sorted(inp["tickers"]):
        s5, s66 = (np.asarray(inp["tickers"][t][k], float) for k in ("sigma5", "sigma66"))
        n = len(s5)
        with np.errstate(invalid="ignore", divide="ignore"):
            V = np.where(s66 > 0, (s5 / s66) ** 2, np.nan)
        used = set()
        for c in np.flatnonzero(V > q):                           # NaN → False
            for d in range(c + 1, min(c + OKNO, n - 1 - VPERED) + 1):
                if s5[d + VPERED] < s5[d - 1 + VPERED]:
                    if d not in used:
                        used.add(d)
                        out.append((t, int(d), 1, float(V[c])))
                    break
    return sorted(out, key=lambda e: (e[0], e[1]))
