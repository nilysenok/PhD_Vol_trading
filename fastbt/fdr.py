"""Занятость капитала на 18:50 по тикеру для учёта ФДР (контракт #376, §1.3 А).

Строка t — интервал [close t, close следующего торгового дня). noch[t] — доля слота тикера,
занятая в этой ночи. Чистые функции, никуда не подключены (подключение — К-Ф2).
"""
import numpy as np

REZHIMY = ("t0", "t1")


def _check(rezhim):
    if rezhim not in REZHIMY:
        raise ValueError(f"rezhim должен быть из {REZHIMY}, получено {rezhim!r}")


def noch_iz_sdelok(trades, w_tr: dict, n: int, rezhim) -> np.ndarray:
    """noch0[t] = Σ w·1[entry_day ≤ t < exit_day]; T+1 добавляет w в строке exit_day, min(1, ·).

    Внутридневная сделка (entry_day = exit_day) не занимает ни одной ночи при T+0.
    exit_day None — открыта на конце окна: занята до строки n−1 включительно.
    Вес сделки — w_tr[tr.d_sig]; отсутствие веса — ошибка (KeyError), не молчаливая единица.
    """
    _check(rezhim)
    noch = np.zeros(n)
    extra = np.zeros(n)
    for tr in trades:
        w = float(w_tr[tr.d_sig])
        a = max(0, int(tr.entry_day))
        x = n if tr.exit_day is None else min(n, int(tr.exit_day))
        if a < x:
            noch[a:x] += w
        if rezhim == "t1" and tr.exit_day is not None and 0 <= tr.exit_day < n:
            extra[int(tr.exit_day)] += w
    if rezhim == "t1":
        return np.minimum(1.0, noch + extra)
    return noch


def noch_iz_pos(pos, first, rezhim) -> np.ndarray:
    """По позициям A (дробная pos): noch0 = |pos|; T+1: min(1, |pos| + zakryto).

    zakryto[t] = |pos[t−1]| при смене знака, иначе max(0, |pos[t−1]| − |pos[t]|); на первой строке пары — 0.
    """
    _check(rezhim)
    pos = np.asarray(pos, dtype=float)
    first = np.asarray(first, dtype=bool)
    ap = np.abs(pos)
    if rezhim == "t0":
        return ap
    prev = np.r_[0.0, pos[:-1]]
    aprev = np.abs(prev)
    flip = np.sign(pos) != np.sign(prev)
    zak = np.where(flip, aprev, np.maximum(0.0, aprev - ap))
    zak[first] = 0.0
    return np.minimum(1.0, ap + zak)
