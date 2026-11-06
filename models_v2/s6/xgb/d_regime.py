"""Диагностика переключения R4 (GATES_D_xgb Q6): число переключений, задержка реакции, потери в переходах. Метки — только функции прошлого; «шок» — диагностика (известен в день t)."""
from __future__ import annotations

import numpy as np

SHOCK_X, SHOCK_WIN, LOOKAHEAD, AFTER = 3.0, 252, 22, 5


def n_switches(state_grid: np.ndarray) -> int:
    s = np.asarray(state_grid, bool)
    return int((s[1:] != s[:-1]).sum())


def shock_rows(rv: np.ndarray) -> np.ndarray:
    """Строки t, где дневной RV > 3 × медиана предыдущих 252 (без t); известно в t."""
    out = []
    for t in range(SHOCK_WIN, len(rv)):
        past = rv[t - SHOCK_WIN:t]
        past = past[np.isfinite(past)]
        if len(past) >= SHOCK_WIN - 10 and np.isfinite(rv[t]) and rv[t] > SHOCK_X * np.median(past):
            out.append(t)
    return np.array(out, int)


def reaction_delay(state: np.ndarray, shocks: np.ndarray) -> dict:
    """Задержка (строк) от шока t до первого флага состояния на/после t (в пределах 22 строк). state — по строкам F."""
    st, lag = np.asarray(state, bool), []
    for t in shocks:
        hit = np.flatnonzero(st[t:t + LOOKAHEAD + 1])
        lag.append(int(hit[0]) if len(hit) else np.nan)
    lag = np.array(lag, float)
    ok = np.isfinite(lag)
    return {"n_shocks": int(len(lag)), "share_flagged": float(ok.mean()) if len(lag) else float("nan"),
            "median_delay": float(np.median(lag[ok])) if ok.any() else float("nan"), "share_same_day": float((lag == 0).mean()) if len(lag) else float("nan")}


def transition_mask(state_grid: np.ndarray) -> np.ndarray:
    """Даты сетки в день смены состояния и в 5 следующих за ней."""
    s = np.asarray(state_grid, bool)
    ch = np.concatenate([[False], s[1:] != s[:-1]])
    m = ch.copy()
    for k in range(1, AFTER + 1):
        m[k:] |= ch[:-k]
    return m
