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
    t: np.ndarray | None = None   # начало бара, минуты от полуночи (заполняет Л-3); None — времени нет


@dataclass(frozen=True)
class Params:
    k: float | None = None        # None — вход по close d; иначе лимит L = P_d·(1 − k·σ̂₁)
    f: str = "propusk"            # propusk | zakrytie_d1 | povtor
    a: float | None = None        # тейк T = P_вх·(1 + a·σ̂)
    b: float | None = None        # стоп S = P_вх·(1 − b·σ̂)
    tau: int | None = None        # временной стоп, торговых дней
    s: str = "fiks"               # fiks | ezhedn
    comm: float = 0.0004          # за сторону, доля
    exit_signal: bool = True      # выход по close первого дня с sig != стороне сделки
    # --- Ж-047 (контракт §1.1); всё выключено по умолчанию, E не меняется ни на байт ---
    side: int = 1                 # +1 / −1: множитель знака sig (−1 — шорт по сигналу 0/1)
    tick: float | None = None    # шаг цены: тейк и лимит — на 1 тик сквозь уровень; None — строго (E §2)
    stop_touch: bool = False      # стоп при касании (l <= S), иначе l < S
    R: float | None = None        # цель P(1 + R·b·σ√h); требует b; несовместимо с a
    h: float | None = None        # h в √h; None — τ (❓1)
    k_tr: float | None = None     # трейлинг: экстремум с входа ∓ k_tr·σ (нужен b)
    be: bool = False              # безубыток после +1σ, со следующего бара
    check: str = "bary"           # bary | zakrytie (выходы только по close Н6)
    reentry: bool = True          # False — блок повторного входа до sig == 0 или смены знака
    vyhod_lim: bool = False       # Ж-065 п. 3: выход по signal/vremya — лимитом на d+1 (иначе close дня решения); нужен k, check="bary"
    # --- прогон этапа 1, Л-2 (контракт §1.1); умолчания = прежнее поведение, Run2 включается только при отличиях ---
    zhizn: int | None = None      # исполнение лимита только на барах b < N дня d+1 (None — весь день)
    gap_propusk: bool = False     # open бара 0 уже за лимитом → заявка снята, blocked = сторона
    t_vyhod: str = "close"        # close | 1500 | 1700 | open2 — исполнение выхода по signal/vremya (exec_e_vremya)
    bar_plus: int = 0             # исполнение (лимит, стоп, тейк) — по open бара b+n, если хуже уровня
    stop_tik: int = 0             # цена стопа хуже на n тиков (нужен tick)
    tejk: str = "net"             # Ж-084: net | k_srednemu | pol_puti — тейк от уровня сигнала M (fastbt/tejk_signal.py); нужен tejk_M в run


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
    side: int = 1                 # +1 лонг, −1 шорт
    fee_in: float = 0.0
    fee_out: float = 0.0
    row0: int = -1                # строка комиссии входа (день по close или начало интервала); для весов на сделку (К4)


@dataclass
class Result:
    trades: list
    gross: np.ndarray
    cost: np.ndarray
    pos: np.ndarray
    free: np.ndarray
    fdr: np.ndarray
    n_orders: int = 0             # лимитных заявок поставлено
    n_fills: int = 0              # из них исполнено
    x_orders: int = 0             # лимитных заявок выхода поставлено (vyhod_lim); n_orders/n_fills — только входы
    x_fills: int = 0              # из них исполнено (причина "lim")
    x_t: int = 0                  # выходов исполнено не по close (t_vyhod ≠ close)
    n_bez_tejka: int = 0          # Ж-084: входов без тейка (M не в сторону прибыли или нечисловой)

    @property
    def pnl(self):
        return self.gross - self.cost

    @property
    def fill_rate(self):
        return self.n_fills / self.n_orders if self.n_orders else float("nan")
