"""T17 — сжатие диапазона и пробой (контракт #376 §2.3; толкование [Ж-065 п. 4]). Обе стороны.

Rg_d = (max h − min l за W дней по d включительно) / c_d. День сжатия c: Rg_c ≤ min Rg за L дней по c + 1e−12.
Пробой — первый d ∈ (c, c+W]: c_d > H_c = max h[c−W+1..c] (лонг) или c_d < L_c = min l[c−W+1..c] (шорт).
Один пробой на день d (эпизод): при нескольких c с одним d берётся ранний c. Вход close d+1 — в `postroit_sig`.
Сила = |ln(c_d / граница)| / σ̂_d (в σ; σ̂_d известна на 18:50 d).

inp = {"dates", "tickers": {t: {"close", "high", "low", "sigma"}}}; osi = {"W", "L"}. Даты ≥ 2022 → PermissionError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUT = np.datetime64("2022-01-01")
SDVIG = 0                     # 1 — мутация: окна Rg и H_c захватывают d+1


def _vpered(a: np.ndarray) -> np.ndarray:
    return a if SDVIG == 0 else np.r_[a[SDVIG:], np.full(SDVIG, np.nan)]


def vhod() -> dict:
    from fastbt import t_vhody
    return t_vhody.zagruzit()


def porog(osi: dict) -> float:
    """У пробоя порога в условии нет (↺): базой веса взята сила 1σ (пробой на 1σ̂_d даёт множитель 1)."""
    return 1.0


def sobytiya(inp: dict, osi: dict) -> list:
    if len(inp["dates"]) and np.max(np.asarray(inp["dates"], "datetime64[D]")) >= CUT:
        raise PermissionError("T17: даты ≥ 2022")
    W, L = int(osi["W"]), int(osi["L"])
    out = []
    for t in sorted(inp["tickers"]):
        x = inp["tickers"][t]
        c, h, lo, sg = (np.asarray(x[k], float) for k in ("close", "high", "low", "sigma"))
        hi_w = _vpered(pd.Series(h).rolling(W).max().to_numpy())
        lo_w = _vpered(pd.Series(lo).rolling(W).min().to_numpy())
        rg = (hi_w - lo_w) / c
        rmin = pd.Series(rg).rolling(L).min().to_numpy()
        used = set()
        for k in np.flatnonzero(rg <= rmin + 1e-12):               # NaN в сравнении → False
            for d in range(k + 1, min(k + W, len(c) - 1) + 1):
                side = 1 if c[d] > hi_w[k] else -1 if c[d] < lo_w[k] else 0
                if side:
                    if d not in used and sg[d] > 0:
                        used.add(d)
                        bound = hi_w[k] if side == 1 else lo_w[k]
                        out.append((t, int(d), side, float(abs(np.log(c[d] / bound)) / sg[d])))
                    break
    return sorted(out, key=lambda e: (e[0], e[1]))
