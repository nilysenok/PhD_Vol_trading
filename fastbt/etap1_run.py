"""Адаптер Sk → exec_e (К4, Ж-047): строки портфеля g, sides, expo по одному тикеру с весом на сделку.

`run_e1` гоняет `exec_e.run` и раскладывает результат в строки окна [a; b) для `Engine.port_rows`:
 - g[t]    = вес сделки × доходность строки (строка t — интервал [close t; close след. дня));
 - expo[t] = вес × доля баров следующего дня в рынке (у A — |pos|);
 - sides[t]= вес × число сторон. Вход — в строке row0 (по close — день, внутри дня — начало интервала),
   выход — в строке комиссии выхода. Границы окна — по правилу `Engine.port` (Н5, FA2a): сделка, живая в строке a,
   платит вход в строке a; сделка, открытая в последней строке окна (или выходящая позже), платит выход в строке b−1;
   сделка, закончившаяся по close a (последняя строка дохода a−1), и вход в строке ≥ b в окно не входят.
Вес фиксирован на сделку: w[день сигнала] (ежедневной перебалансировки нет, контракт §1.1, тест 2.17).
Якорь: Params без σ-выходов, k=None → g = pos·lr, sides = |Δpos|, expo = |pos| (те же числа, что внутри `Engine.port`).
Строка b−1 берёт доход из дня b (первый день после окна, Н5) — вызывающий отдаёт ему один бар-close (не 5-мин).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fastbt.exec_e_types import Params


@dataclass
class E1Rows:
    g: np.ndarray
    sides: np.ndarray
    expo: np.ndarray
    cnt: np.ndarray                 # стороны без веса (счёт сделок)
    trades: list = field(default_factory=list)
    n_orders: int = 0
    n_fills: int = 0
    noch: np.ndarray | None = None  # занятость на 18:50 в окне [a; b) при fdr ≠ None (fastbt/fdr.py), иначе None
    x_orders: int = 0               # лимитных заявок выхода и из них исполнено (Ж-065 п. 3)
    x_fills: int = 0


def _spans(res, ix, p: Params, n: int):
    """(сделка, строка входа, последняя строка дохода, строка комиссии выхода) по сделкам прогона."""
    jx = {i: j for j, i in enumerate(ix)}
    for tr in res.trades:
        if tr.exit_day is None:
            yield tr, tr.row0, ix[-1], n
            continue
        j = jx[tr.exit_day]
        close_exit = p.check == "zakrytie" or tr.reason in ("vremya", "signal") or j == 0
        yield tr, tr.row0, ix[max(j - 1, 0)], (tr.exit_day if close_exit else ix[j - 1])


def run_e1(ticker, days, sig, sigma, p: Params, close=None, w=None, win=None, div=None, nalog=None,
           fdr=None) -> E1Rows:
    """fdr: None — как раньше; "t0" | "t1" — E1Rows.noch по fastbt/fdr.py (ставка на свободный капитал, Ж-065 п. 3)."""
    from fastbt import exec_e                       # ленивый импорт: exec_e.run_e1 — тот же объект
    n = len(days)
    a, b = (0, n) if win is None else win
    w = np.ones(n) if w is None else np.asarray(w, float)
    res = exec_e.run(ticker, days, sig, sigma, p, close=close, div=div, nalog=nalog)
    ix = [i for i, d in enumerate(days) if not d.pusto]
    wr, sides, cnt = np.zeros(n), np.zeros(b - a), np.zeros(b - a)
    for tr, r0, last_g, er in _spans(res, ix, p, n):
        wt = float(w[tr.d_sig])
        wr[r0:last_g + 1] = wt
        if r0 < b and last_g >= a:
            for r in (max(r0, a), min(er, b - 1)):
                sides[r - a] += wt
                cnt[r - a] += 1.0
    g, expo = res.gross * wr, (1.0 - res.free) * wr
    noch = None
    if fdr is not None:
        from fastbt import fdr as fdr_mod
        noch = fdr_mod.noch_iz_sdelok(res.trades, {tr.d_sig: float(w[tr.d_sig]) for tr in res.trades}, n, fdr)[a:b]
    return E1Rows(g[a:b], sides, expo[a:b], cnt, res.trades, res.n_orders, res.n_fills, noch,
                  res.x_orders, res.x_fills)
