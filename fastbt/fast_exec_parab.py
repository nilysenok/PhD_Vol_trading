"""П3b в быстром ядре (Ф-parab): параболический трейлинг тренда, Params.parab = (ps, pm). Построчно как `exec_e_parab` (pb_start,
pb_level, pb_after): состояние [стоп, экстремум, шаг] лежит в fs[PBS..PBP], флаг «бар входа» — st[PBN]. Результат побайтно равен эталону.
Модуль не импортирует fast_exec (индексы осей — здесь, fast_exec их берёт)."""
from __future__ import annotations

import numpy as np
from numba import njit

PBS, PBE, PBP = 18, 19, 20                  # fs: стоп бара (храповик), экстремум с входа, шаг
PBN = 32                                    # st: 1 — бар входа ещё не пройден (действует только начальный стоп)
F_PS, F_PM = 22, 23                         # pf: (ps, pm)
P_PARAB = 32                                # pi: ось включена


def params_parab(p) -> tuple:
    ps, pm = p.parab if p.parab is not None else (0.0, 0.0)
    return np.array([ps, pm], float), np.array([p.parab is not None], np.int64)


@njit(cache=True)
def start(fs, st, sd, px, b, sg, hi, lo, ps):
    """pb_start: [P(1 − sd·b·σ), high | low бара входа, ps]; бар входа не пройден."""
    fs[PBS], fs[PBE], fs[PBP] = px * (1 - sd * b * sg), hi if sd > 0 else lo, ps
    st[PBN] = 1


@njit(cache=True)
def level(sd, sl, ext, step):
    """pb_level: sl' = sl ∨ (sl + step·(ext − sl)) (лонг; шорт зеркально); чистая функция."""
    t = sl + step * (ext - sl)
    if sd > 0:
        return t if t > sl else sl
    return t if t < sl else sl


@njit(cache=True)
def stop(fs, st, sd):
    """Стоп текущего бара: в баре входа — начальный, далее по состоянию на конец прошлого бара."""
    return fs[PBS] if st[PBN] else level(sd, fs[PBS], fs[PBE], fs[PBP])


@njit(cache=True)
def after(fs, sd, hi, lo, ps, pm):
    """pb_after: стоп бара становится sl, затем новый экстремум → ext и step += ps (не выше pm)."""
    fs[PBS] = level(sd, fs[PBS], fs[PBE], fs[PBP])
    if (hi > fs[PBE]) if sd > 0 else (lo < fs[PBE]):
        fs[PBE] = hi if sd > 0 else lo
        s = fs[PBP] + ps
        fs[PBP] = pm if pm < s else s
