"""Строгое исполнение дневной позиции для C/D/E (лист CE-1, формат strogoe v 1.0; Ж-086 пп. 1–2, 4; Ж-088 п. 2; Ж-089).

Позиция пары pos[t] решена на close t и исполняется по open 5-мин бара 10:00 дня t+1. Доход строки t (close t → close t+1):
  ночь  n[t]  = ln((o10[t+1] + D·0,87)/c[t])  держит прежняя позиция pos[t−1];
  день  dn[t] = ln(c[t+1]/o10[t+1])           держит новая позиция pos[t].
g[t] = pos[t−1]·n[t] + pos[t]·dn[t]; стороны |pos[t] − pos[t−1]|. Последняя строка окна пары: pos[T] не исполняется
(исполнение было бы за окном), ночная нога = pos[T−1], комиссия выхода |pos[T−1]|. При o10 ≡ c (n = 0, dn = lr) —
побайтно `Engine.port`. Плата за шорт (Ж-089): s·max(0, −pos[t−1])·cal[t]/365 — только перенос через ночь.
`s9.py` не правится: `patched` подменяет `Engine.port` на экземпляре и возвращает как было.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from fastbt import dividendy
from fastbt.exec_e_vremya import HHMM, bar_at

V = "1.0"
ISPOLNENIE = {"vhod": "open1000_d1", "vyhod": "open1000_x1", "v": V}
COMM = 0.0004


def _o10_dnya(days: list, i: int, sch: list) -> float:
    """open 10:00 дня i; нет бара — первый бар позже, иначе open первого бара следующего дня (Ж-086 п. 4); счёт подмен."""
    for j in (i, i + 1):
        if j >= len(days):
            return np.nan
        k = bar_at(days[j], HHMM["open2"])
        if k is not None and k < len(days[j].t):
            sch[0] += int(days[j].t[k] != HHMM["open2"] or j != i)
            return float(days[j].o[k])
    return np.nan


def o10_ryad(eng, days_by_ticker: list) -> tuple[np.ndarray, dict]:
    """[тикер, бар] open 10:00 дня бар+1 (последний бар — nan) и счётчик подмен запасным правилом."""
    nt = len(days_by_ticker)
    nb = max(len(d) for d in days_by_ticker)
    out = np.full((nt, nb), np.nan)
    sch = [0]
    for ti, days in enumerate(days_by_ticker):
        for b in range(len(days) - 1):
            out[ti, b] = _o10_dnya(days, b + 1, sch)
    return out, {"podmen": sch[0]}


MAX_DYR = 0.01                                    # доля дыр на тикер, выше — отказ


def nogi(eng, o10: np.ndarray, div: np.ndarray, close: np.ndarray, max_dyr: float = MAX_DYR) -> tuple:
    """(n_row, dn_row, {"n_dyr": …}) по строкам Engine. div[тикер, бар] — D дня гэпа (`dividendy.gap_divs`), close[тикер, бар] — c[t].

    Дыра (нет o10 или close в строке, кроме последней строки пары) — как у `exec_e` для дня pusto: позиция переносится,
    доход дня 0; каждая такая подстановка считается в n_dyr. Доля дыр тикера > max_dyr — ValueError (Д4)."""
    ti, b = eng.r["tick"], eng.r["bar"]
    o, c = o10[ti, b], close[ti, b]
    c1 = close[ti, np.minimum(b + 1, close.shape[1] - 1)]
    d = div[ti, np.minimum(b + 1, div.shape[1] - 1)] * dividendy.DIV_NALOG
    last = np.r_[eng.first[1:], True]
    with np.errstate(invalid="ignore", divide="ignore"):
        n_row = np.where(last, 0.0, np.log((o + d) / c))
        dn_row = np.where(last, 0.0, np.log(c1 / o))
    dyra = ~last & ~(np.isfinite(n_row) & np.isfinite(dn_row))
    for t in np.unique(ti):
        m = ti == t
        if dyra[m].mean() > max_dyr:
            raise ValueError(f"тикер {t}: дыр {dyra[m].sum()} из {m.sum()} строк > {max_dyr:.0%}")
    n_row, dn_row = np.where(dyra, 0.0, n_row), np.where(dyra, 0.0, dn_row)
    return n_row, dn_row, {"n_dyr": int(dyra.sum())}


def port_strogiy(eng, pos_row, comm=None, zaem: float = 0.05, nogi=None, cap: float = 1.0):
    """Колонки `Engine._finish` + zaem (доля капитала за строку) и noch (занятость на 18:50). Плата входит в r."""
    if nogi is None:
        raise ValueError("port_strogiy: нужны ноги (n_row, dn_row)")
    comm = COMM if comm is None else comm
    n_row, dn_row = nogi[:2]
    pos = np.asarray(pos_row, float)
    first = eng.first
    last = np.r_[first[1:], True]
    prev = np.r_[0.0, pos[:-1]]
    prev[first] = 0.0
    g = prev * n_row + pos * dn_row
    sides = np.abs(pos - prev)
    g = np.where(last, prev * n_row, g)
    sides = np.where(last, np.abs(prev), sides)
    di, nd = eng.r["date"], len(eng.dates)
    z_row = zaem * np.maximum(0.0, -prev) * eng.cal[di] / 365.0
    bc = lambda x: np.bincount(di, x, nd) / eng.cnt             # noqa: E731
    noch = bc(np.abs(prev))
    p = eng._finish(bc(g - sides * comm) - bc(z_row), bc(g), noch, bc(sides), cap, noch)
    p["zaem"], p["noch"] = bc(z_row), noch
    return p


@contextmanager
def patched(eng, nogi, zaem: float = 0.05):
    """Engine.port(pos, comm, cap) → port_strogiy; вызов с wgt — ValueError, не молчаливый close."""
    def port(pos, comm=None, cap=1.0, wgt=None):
        if wgt is not None:
            raise ValueError("strogoe.patched: wgt (P1) не поддержан")
        return port_strogiy(eng, pos, comm, zaem, nogi, cap)
    eng.port = port
    try:
        yield eng
    finally:
        del eng.port


def ispolnenie_tochki(semya: str, tochka: dict) -> dict:
    """Поле `ispolnenie` строки C/D/E по параметрам точки (функция только параметров, не исходов)."""
    if semya in ("C", "D") or tochka.get("family") in ("E0", "E2"):
        return dict(ISPOLNENIE)
    if semya != "E" or tochka.get("family") != "E1":
        raise ValueError(f"неизвестная точка {semya} {tochka.get('family')}")
    if tochka.get("f") == "zakrytie_d1":                      # вход по close d+1 — справочная строка
        return {"vhod": "close_d1", "vyhod": ISPOLNENIE["vyhod"], "v": V}
    return {"vhod": "limit_d1", "vyhod": ISPOLNENIE["vyhod"], "v": V}
