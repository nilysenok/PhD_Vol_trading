"""Пауза при скользящем Шарпе тени — фильтр входа этапа 1, итерация 23 (Ж-051, контракт §2Б.5).

Ряд — дневной excess ТЕНИ Sk (та же точка без паузы; по собственному ряду пауза не кончилась бы).
Строка с индексом d — доход интервала [close d, close d+1); известна на close d+1.
  Шарп_d = mean/std (ddof=1) строк d−W…d−1 · √252;  Шарп_d < porog → новые входы Sk в день d запрещены.
Открытые позиции не закрываются. Нет полного окна или NaN в окне → Шарп NaN → пауза не включается
(вход разрешён: нет оснований запрещать). Нулевой std → Шарп NaN (деления на 0 нет).
Константы для тестов-мутаций (подмена в памяти): ROW_LAG=0 — как в контракте; −1 = в окно входит строка d.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROW_LAG = 0       # окно кончается строкой d−1; −1 = строкой d (мутация, мутации № 18 по смыслу)
DDOF = 1          # одно определение (№ 12)
ANN = 252


def rolling_sharpe(excess: pd.Series, w: int) -> pd.Series:
    """Шарп_d по строкам d−W…d−1 (годовая шкала); индекс — дни ряда."""
    sh = 1 + ROW_LAG                      # сдвиг: строки d−W…d−1 известны на close d
    r = excess.astype(float).shift(sh)
    sd = r.rolling(w, min_periods=w).std(ddof=DDOF)
    return (r.rolling(w, min_periods=w).mean() / sd.where(sd > 0)) * np.sqrt(ANN)


def pauza_mask(excess_ten: pd.Series, w: int = 63, porog: float = 0.0) -> pd.Series:
    """True = пауза в день d (новые входы запрещены)."""
    return (rolling_sharpe(excess_ten, w) < porog).fillna(False)


def allow(pt, t: str, dates, ctx) -> tuple:
    """(long, short) — оба False в дни паузы; `ctx.ten` — {тикер: excess тени}; оси pt: pauza_okno, pauza_porog."""
    g = (lambda k, d: pt.get(k, d)) if isinstance(pt, dict) else (lambda k, d: getattr(pt, k, d))
    dt = pd.DatetimeIndex(dates)
    w, p = g("pauza_okno", None), g("pauza_porog", None)
    if w is None or p is None:                       # ось не задана — пути P1 побайтно
        ok = np.ones(len(dt), bool)
        return ok, ok.copy()
    m = pauza_mask(ctx.ten[t], int(w), float(p)).reindex(dt).fillna(False).to_numpy(bool)
    return ~m, ~m
