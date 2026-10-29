"""Слой исполнения v4 (consec, min_hold, cooldown, min_sily) как фильтр сигнала ДО автомата v2 (План сетки v5, П2).

Не переносится поверх готовых позиций: уровни заявок, стопы и тейки строит автомат по уже отфильтрованному sig.
Правила — fastbt/execlayer.exec_layer (V4:211-275, побитово): сила → |sig|>0,5 → consec → min_hold → cooldown;
знак берётся из сырого sig ОДИН раз внутри exec_layer — после слоя знак не умножается повторно (мутация № 20).
Cooldown считается от дня выхода (первый день b=0 после b=1), не от дня входа. Все оси выключены → sig возвращается без изменений.
Особенность v4 (сохранена): в дни удержания min_hold, где сырой sig = 0, знак 0 → позиция 0.
"""
from __future__ import annotations

import numpy as np

from fastbt import execlayer
from fastbt.exec_e import run
from fastbt.exec_e_types import Params


def filtr_signala(sig, strength=None, consec: int = 1, min_hold: int = 0, cooldown: int = 0, min_sily: float = 0.0) -> np.ndarray:
    """sig[d] ∈ {−1, 0, +1} (известен на close d) → отфильтрованный sig той же длины; strength[d] — сила на close d."""
    sig = np.asarray(sig)
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
    return execlayer.exec_layer(sig.astype(float), st, int(consec), int(min_hold), int(cooldown), float(min_sily), len(sig))


def run_v4(ticker, days, sig, sigma, p: Params, strength=None, consec: int = 1, min_hold: int = 0, cooldown: int = 0,
           min_sily: float = 0.0, **kw):
    """exec_e.run по сигналу, отфильтрованному слоем v4; остальные аргументы — как у run. Знак не трогается."""
    f = filtr_signala(sig, strength, consec, min_hold, cooldown, min_sily)
    return run(ticker, days, f, sigma, p, **kw)
