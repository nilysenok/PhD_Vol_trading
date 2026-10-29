"""Слой исполнения v4 (consec, min_hold, cooldown, min_sily) как фильтр сигнала ДО автомата v2 (План сетки v5, П2).

Не переносится поверх готовых позиций: уровни заявок, стопы и тейки строит автомат по уже отфильтрованному sig.
Правила — fastbt/execlayer.exec_layer (V4:211-275, побитово): сила → |sig|>0,5 → consec → min_hold → cooldown;
знак берётся из сырого sig ОДИН раз внутри exec_layer — после слоя знак не умножается повторно (мутация № 20).
Cooldown считается от дня выхода (первый день b=0 после b=1), не от дня входа. Все оси выключены → sig возвращается без изменений.
min_hold_rezhim (Ж-129 п. 7): "znak_vhoda" (по умолчанию) — в дни удержания при сыром sig = 0 позиция сохраняет знак входа;
разворот внутри удержания — как в v4 (знак из сырого sig, срок не перезапускается). "v4" — прежнее обнуление (знак 0),
путь через execlayer.exec_layer, только для воспроизведения печати.
"""
from __future__ import annotations

import numpy as np

from fastbt import execlayer
from fastbt.exec_e import run
from fastbt.exec_e_types import Params


def _sloj_znak_vhoda(raw, st, consec, min_hold, cool, min_str):
    """Копия execlayer.exec_layer с одним отличием: в днях удержания при raw = 0 знак = знак входа."""
    n = len(raw)
    b = (np.abs(raw) > 0.5).astype(float)
    if min_str > 0.0:
        b[st < min_str] = 0.0
    if consec > 1:
        cnt, inp = 0, False
        for i in range(n):
            if b[i] > 0:
                if not inp:
                    cnt += 1
                    if cnt >= consec:
                        inp = True
                    else:
                        b[i] = 0.0
            else:
                cnt, inp = 0, False
    fill = np.zeros(n)
    rem, hs = 0, 0.0
    for i in range(n):
        if b[i] > 0 and rem == 0:
            rem, hs = min_hold, float(np.sign(raw[i]))
        if rem > 0:
            b[i], fill[i] = 1.0, hs
            rem -= 1
        else:
            b[i] = 0.0
    if cool > 0:
        res, cl = np.zeros(n), 0
        for i in range(n):
            if cl > 0:
                cl -= 1
                continue
            if b[i] > 0:
                res[i] = 1.0
            elif i > 0 and res[i - 1] > 0:
                cl = cool
        b = res
    sg = np.sign(raw)
    return b * np.where(sg == 0, fill, sg)


def filtr_signala(sig, strength=None, consec: int = 1, min_hold: int = 0, cooldown: int = 0, min_sily: float = 0.0,
                  min_hold_rezhim: str = "znak_vhoda") -> np.ndarray:
    """sig[d] ∈ {−1, 0, +1} (известен на close d) → отфильтрованный sig той же длины; strength[d] — сила на close d."""
    sig = np.asarray(sig)
    if min_hold_rezhim not in ("znak_vhoda", "v4"):
        raise ValueError("min_hold_rezhim ∈ {'znak_vhoda', 'v4'}")
    if consec < 1 or min_hold < 0 or cooldown < 0 or min_sily < 0:
        raise ValueError("consec ≥ 1, min_hold ≥ 0, cooldown ≥ 0, min_sily ≥ 0")
    if execlayer.is_trivial(consec, min_hold, cooldown, min_sily):
        return sig
    if min_sily > 0:
        if strength is None or len(strength) != len(sig):
            raise ValueError("min_sily > 0 требует strength длины sig")
        st = np.asarray(strength, float)
    else:
        st = np.zeros(len(sig))
    if min_hold > 0 and min_hold_rezhim == "znak_vhoda":
        return _sloj_znak_vhoda(sig.astype(float), st, int(consec), int(min_hold), int(cooldown), float(min_sily))
    return execlayer.exec_layer(sig.astype(float), st, int(consec), int(min_hold), int(cooldown), float(min_sily), len(sig))


def run_v4(ticker, days, sig, sigma, p: Params, strength=None, consec: int = 1, min_hold: int = 0, cooldown: int = 0,
           min_sily: float = 0.0, min_hold_rezhim: str = "znak_vhoda", **kw):
    """exec_e.run по сигналу, отфильтрованному слоем v4; остальные аргументы — как у run. Знак не трогается."""
    f = filtr_signala(sig, strength, consec, min_hold, cooldown, min_sily, min_hold_rezhim)
    return run(ticker, days, f, sigma, p, **kw)
