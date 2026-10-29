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
    vhod_tip: str = "podtverzhdenie"  # Ж-091: podtverzhdenie | stop_zayavka (S3/S4: вход и выход стоп-заявками, нужен urovni в run)
    vhod_open: bool = False       # Ж-086 п. 1: вход рынком (k = None) по open бара 10:00 дня d+1, а не по close d (exec_e_vremya.bar_at)
    atr_m: float | None = None    # Ж-092 S3/S4 п. 4: трейлинг экстремум с входа ∓ m·ATR(14, дни до d−1), стоп не ослабляется; None — выкл
    chast_1r: bool = False        # Ж-092: на +1R (R = расстояние входа до начального стопа, нужен b) закрыть ½ лимитом, со следующего бара
    T: int = 1                    # V6-a (Ж-111): срок жизни заявки входа, дней; пересчёт каждый вечер, снятие при уходе/смене стороны сигнала
    reentry_den: bool = False     # V6-a (Ж-111): повторный вход в тот же день после внутридневного выхода (уровень вечера d−1, не более 1 за день)
    # --- V6-b (Ж-111 пп. 7, 8б, 10; fastbt/exec_e_v6.py): всё выключено, нужен b, check="bary"; медленный путь ---
    b0: float | None = None       # безубыток: после +b0·σ стоп не хуже цены входа (взаимоисключающе с be и b0_c)
    b0_c: float | None = None     # то же, порог b0 = b0_c·ранг силы (ранг на день сигнала — аргумент run(rang=…), [0; 1])
    sp: tuple | None = None       # «стоп в плюс» (p, q): после +p·σ стоп на +q·σ от цены входа (q < p)
    step_s: float | None = None   # ступенчатый трейлинг: каждые +s·σ стоп подтягивается на +s·σ
    pir: float = 0.0              # пирамидинг: размер добавки (доля базовой ноги) на +1σ, +2σ; ≤ 2 добавок, риск ≤ 2ρ
    ed_s: str = "sigma"           # Л5 (Ж-127): единица стопа и безубытка: sigma | atr (ATR(14) до дня входа / close) | pct (1: b = доля цены)
    ed_t: str = "sigma"           # Л5: единица тейка a (a = 0,03 при pct — 3 %); те же значения
    chand: float | None = None    # Л5: Chandelier k — экстремум 11 торговых дней до дня ∓ k·ATR(14); сверх начального стопа b, храповик
    tejk: str = "net"         # Ж-084: net | k_srednemu | pol_puti — тейк от уровня сигнала M (fastbt/tejk_signal.py); нужен tejk_M в run
    razryv: int | None = None     # Д5-2 (умолчание дирижёра 29.10): отложенная заявка снимается, если до следующего торгового дня тикера > razryv календарных
    #                               дней (остановка торгов); нужен dates в run; None — выкл (прежнее); открытая позиция держится; быстрый слой не знает (ne_znaet)
    # --- Л6 (Ж-127), медленный путь; выключено по умолчанию — путь прежний побайтно ---
    k_e: float = 0.0              # stop_zayavka: уровень входа сдвинут на k_e·ATR(14, close d) от цены (лонг вверх, шорт вниз; < 0 — ближе)
    a_rang: tuple | None = None   # тейк от силы: a = a0 + a1·ранг(d_сигнала), (a0 > 0, a1 ≥ 0); нужен rang в run, несовместим с a и R
    parab: tuple | None = None    # П3b: параболический трейлинг тренда (ps, pm) — fastbt/exec_e_parab.py; нужен b, check="bary"; медленный путь
    chast: tuple | None = None    # частичный выход (доля, r): закрыть долю лимитом на +r·R; (0,5; 1) ≡ chast_1r; нужен b, check="bary"
    time_decay: bool = False      # V6-9 (v4 engine.py:113): после tau//2 дней в позиции тейк «net» — половина дистанции; нужен tau; быстрый слой не знает (ne_znaet)
    # --- V6-10 (И11, И12; fastbt/exec_e_piv.py): стоп/тейк по уровням пивота S5, стоп по VWAP S6; уровни — PivUrovni в tejk_M; быстрый слой не знает (ne_znaet) ---
    piv_sl: str | None = None     # стоп: next (|S1−S2| / |R2−R1|) | half (его половина) | vwap (за полосой VWAP ± vwap_m·dev); расстояние от цены входа
    piv_buf: float = 0.0          # буфер стопа piv_sl next/half: расстояние ×(1 + piv_buf); 0 — без буфера
    piv_tp: str | None = None     # тейк: P (до пивота) | R (до следующего уровня |R1−S1|) | half (его половина)
    vwap_m: float | None = None   # множитель dev для piv_sl = vwap (v4 sl_dev 1,5…3)


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
    row0: int = -1               # строка комиссии входа (день по close или начало интервала); для весов на сделку (К4)
    frac: float = 1.0             # Ж-092: доля позиции сделки (частичный выход ½: у дочерней 0,5, у родителя после него 0,5)
    parent: int | None = None     # Ж-092: индекс родительской сделки в Result.trades (частичный выход); иначе None


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
