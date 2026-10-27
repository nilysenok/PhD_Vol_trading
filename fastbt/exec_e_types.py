"""Типы движка E (fastbt/exec_e.py): бары дня, параметры точки, сделка, результат."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Day:
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    pusto: bool = False


@dataclass(frozen=True)
class Params:
    k: float | None = None        # None — вход по close d; иначе лимит L = P_d·(1 − k·σ̂₁)
    f: str = "propusk"            # propusk | zakrytie_d1 | povtor
    a: float | None = None        # тейк T = P_вх·(1 + a·σ̂)
    b: float | None = None        # стоп S = P_вх·(1 − b·σ̂)
    tau: int | None = None        # временной стоп, торговых дней
    s: str = "fiks"               # fiks | ezhedn
    comm: float = 0.0004          # за сторону, доля
    exit_signal: bool = True      # выход по close первого дня с sig == 0


@dataclass
class Trade:
    ticker: str
    d_sig: int
    entry_day: int
    entry_bar: int
    entry_px: float
    exit_day: int | None = None
    exit_bar: int | None = None
    exit_px: float | None = None
    reason: str | None = None     # stop | take | vremya | signal; None — открыта на конце
    fee_in: float = 0.0
    fee_out: float = 0.0


@dataclass
class Result:
    trades: list
    gross: np.ndarray
    cost: np.ndarray
    pos: np.ndarray
    free: np.ndarray
    fdr: np.ndarray

    @property
    def pnl(self):
        return self.gross - self.cost
