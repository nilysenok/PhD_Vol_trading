"""E, длинная сторона, строгое исполнение (лист CE-6; Ж-088 п. 4, Ж-086 п. 1, 4).

Вход E0/E2 (рынок) — по open 10:00 дня d+1; E1 (лимит) — по барам d+1; выход по сигналу и по τ — по open 10:00 дня x+1
(`t_vyhod = open2`); close — только справочно (`spravka=True`, f = zakrytie_d1). Движок — `exec_e2.Run2` (оси `vhod_open`,
`t_vyhod` там уже есть), `sem_e.py` и `exec_e*.py` не правятся. Короткая сторона и портфель по парам — CE-7.

↺ Конец окна: открытая лонг-позиция платит выход по close последней строки (Н5, как в `exec_e.run`), а не по o10
следующего дня — вопрос владельца ❓ 11 плана CE (строгая формула конца окна — в `strogoe.port_strogiy`, CE-7).
"""
from __future__ import annotations

import numpy as np

from fastbt import sem_e
from fastbt.exec_e import Params, run
from fastbt.strogoe import _o10_dnya

PRICHINY = ("signal", "tau", "vremya")
_KEYS = ("k", "f", "a", "b", "tau", "s")


def _o10(days: list, i: int) -> float:
    if not 0 <= i < len(days):
        raise ValueError(f"день {i} вне окна (дней {len(days)}): строгая цена недоступна")
    px = _o10_dnya(days, i, [0])
    if not np.isfinite(px):
        raise ValueError(f"день {i}: нет open 10:00 и запасного бара")
    return px


def cena_vhoda(days: list, d: int, tochka: dict) -> float:
    """Цена входа рынком E0/E2 по сигналу на close d: open бара 10:00 дня d+1. E1 (лимит) — цена по барам, не точка."""
    if tochka.get("k") is not None or tochka.get("family") == "E1":
        raise ValueError("E1: вход лимитом по барам d+1 (Run2), единой цены нет")
    return _o10(days, d + 1)


def cena_vyhoda(days: list, x: int, tochka: dict, prichina: str) -> float:
    """Цена выхода по сигналу / τ с решением на close x: open бара 10:00 дня x+1."""
    if prichina not in PRICHINY:
        raise ValueError(f"причина выхода {prichina!r}: допустимо {PRICHINY}")
    return _o10(days, x + 1)


def params_strogie(par: dict, comm_pct: float, spravka: bool = False) -> Params:
    """Params строгого E: k = None → вход по open 10:00 d+1; выходы по signal/vremya — open2. f = zakrytie_d1 (вход по
    close d+1) — только справочно."""
    if par.get("f") == "zakrytie_d1" and not spravka:
        raise ValueError("f = zakrytie_d1 — вход по close, только справочная строка (spravka=True; Ж-088 п. 4)")
    kw = {k: par[k] for k in _KEYS if k in par}
    return Params(**kw, comm=comm_pct / 100, t_vyhod="open2", vhod_open=kw.get("k") is None)


def run_dlinnaya(p, inp, obj: dict, par: dict, comm_pct: float, end, spravka: bool = False):
    """Прогон длинной стороны пары на строгом исполнении → (Result, n, даты, rf). Сигнал, σ̂ и ставка — как в `sem_e.pair_e`."""
    pr = params_strogie(par, comm_pct, spravka)
    n = int(p.dates.searchsorted(end, side="right"))
    d, rf = p.dates[:n], sem_e._rf_cal(p.dates[:n], inp.rate)
    sig = sem_e._sig_e(p, inp, obj, par.get("q", 0.0))[:n]
    r = run(p.ticker, p.days[:n], sig, inp.sigma[obj["sigma"]][p.ticker][:n], pr, rf_cal=rf)
    return r, n, d, rf


def pair_e(p, inp, obj: dict, par: dict, comm_pct: float, end, t0=None, spravka: bool = False):
    """Длинная сторона одной пары: → (избыточная серия лонга, позиция ≥ 0 по дням, число сторон в окне [t0; end])."""
    import pandas as pd
    r, n, d, rf = run_dlinnaya(p, inp, obj, par, comm_pct, end, spravka)
    ex = r.pnl + r.fdr - rf
    m = d >= (d[0] if t0 is None else t0)
    cnt = sum((d[t.entry_day] >= d[m][0]) + (t.exit_day is None or d[t.exit_day] >= d[m][0]) for t in r.trades) if m.any() else 0.0
    return pd.Series(ex, d), r.pos, float(cnt)
