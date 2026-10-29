"""П3b (Ж-127): параболический трейлинг тренда как значение оси трейлинга автомата v2 (Params.parab = (ps, pm)).

Правила — fastbt/vyhod_parabolic.py (#589): стоп бара u строится по состоянию на конец бара u−1; экстремум и шаг
обновляются ПОСЛЕ бара u. Состояние [sl, ext, step]: sl — стоп, действовавший на предыдущем баре (храповик), ext — экстремум
с входа (на входе — high/low бара входа), step — доля подтяжки (ps, +ps за новый экстремум, не выше pm).
Стоп бара: sl' = sl ∨ (sl + step·(ext − sl)) (лонг; шорт зеркально). В баре входа действует только начальный стоп (sl).
Начальный стоп — P(1 − sd·b·σ), как у стопа σ; безубытка в оси нет (вторичные трейлинги несовместимы). Медленный путь.
"""
from __future__ import annotations

from fastbt.exec_e_types import Params


def parab_new(p: Params) -> bool:
    return p.parab is not None


def parab_check(p: Params) -> None:
    if p.parab is None:
        return
    ps, pm = p.parab if len(p.parab) == 2 else (0, 0)
    if not 0 < ps <= pm:
        raise ValueError("parab = (ps, pm), 0 < ps ≤ pm")
    if p.b is None or p.check != "bary" or p.s != "fiks":
        raise ValueError("parab требует b (начальный стоп), check='bary', s='fiks'")
    if p.k_tr is not None or p.be or p.atr_m is not None or p.chand is not None or p.b0 is not None or p.b0_c is not None \
            or p.sp is not None or p.step_s is not None or p.pir:
        raise ValueError("parab несовместим с другими трейлингами и безубытком")


def pb_start(sd: int, px: float, sg: float, p: Params, hi: float, lo: float) -> list:
    """Состояние на входе: [начальный стоп, экстремум бара входа, ps]."""
    return [px * (1 - sd * p.b * sg), hi if sd > 0 else lo, p.parab[0]]


def pb_level(sd: int, st: list) -> float:
    """Стоп следующего бара по состоянию на конец предыдущего (чистая функция: храповик, без записи)."""
    sl, ext, step = st
    t = sl + step * (ext - sl)
    return max(sl, t) if sd > 0 else min(sl, t)


def pb_after(sd: int, st: list, p: Params, hi: float, lo: float) -> None:
    """После бара: стоп бара становится sl, затем новый экстремум → ext и step += ps (не выше pm)."""
    st[0] = pb_level(sd, st)
    if (hi > st[1]) if sd > 0 else (lo < st[1]):
        st[1] = hi if sd > 0 else lo
        st[2] = min(st[2] + p.parab[0], p.parab[1])
