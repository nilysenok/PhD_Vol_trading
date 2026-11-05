"""B2: причинный CUSUM по ошибкам HAR-J(б) (GATES_B2 §4). Параметры не подбираются: k=0,5; h=5 или 8; окно стандартизации 252 (>=126)."""
from __future__ import annotations

import numpy as np

K, WIN, MINW, MIN_PAIRS = 0.5, 252, 126, 252


def arrivals(y: np.ndarray) -> np.ndarray:
    """y_s = ln RV_{s+1} − lb_s известна на строке s+1: e[u] = y[u−1]."""
    e = np.full(len(y), np.nan)
    e[1:] = y[:-1]
    return e


def har_arrivals(F) -> np.ndarray:
    """Ошибки PIT HAR-J(б) по строкам прихода (e[u] — цель u)."""
    from models_v2 import har
    from models_v2.s3 import resid
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    return arrivals(resid.target_minus(F, resid.harj_ab(pan)[1]))


def run(e: np.ndarray, h: float, k: float = K, win: int = WIN, minw: int = MINW) -> tuple:
    """-> (S+, S−, alarm). Стандартизация e[u] — по закрытым ошибкам e[u−win:u] (строго до u). NaN состояние не меняет."""
    n = len(e)
    sp, sm, al = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    a = b = 0.0
    for u in range(n):
        past = e[max(0, u - win):u]
        past = past[np.isfinite(past)]
        if np.isfinite(e[u]) and len(past) >= minw:
            sd = past.std(ddof=1)
            if sd > 0:
                z = (e[u] - past.mean()) / sd
                a, b = max(0.0, a + z - k), max(0.0, b - z - k)
                if max(a, b) > h:
                    al[u], a, b = True, 0.0, 0.0
        sp[u], sm[u] = a, b
    return sp, sm, al


def since(alarm: np.ndarray, cap: int) -> np.ndarray:
    """Строк с последнего срабатывания <= i (потолок cap; срабатываний не было — cap)."""
    out, last = np.full(len(alarm), float(cap)), None
    for i, f in enumerate(alarm):
        last = i if f else last
        if last is not None:
            out[i] = min(cap, i - last)
    return out


def restart_row(alarm: np.ndarray, T: int):
    """Последняя строка срабатывания <= T или None."""
    w = np.flatnonzero(alarm[:T + 1])
    return int(w[-1]) if len(w) else None


def window_pairs(idx: np.ndarray, alarm: np.ndarray, T: int, floor: int = MIN_PAIRS) -> np.ndarray:
    """Пары окна: t >= строки срабатывания, но не меньше `floor` последних пар; без срабатываний — все (expanding)."""
    r = restart_row(alarm, T)
    if r is None:
        return idx
    sel = idx[idx >= r]
    return sel if len(sel) >= floor else idx[-floor:]
