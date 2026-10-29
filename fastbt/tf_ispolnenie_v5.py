"""П5b (Ж-125 п. 3, Ж-111): внутридневное исполнение для tf ∈ {10m, 30m, 1h, 2h} поверх автомата v2 (fastbt/exec_e*.py).

Отображение: каждый бар ТФ — «день» автомата из одного бара, сессия (календарный день) — отдельный прогон Run/Run2.
Решение — на close бара t; лимит-вход, стоп, тейк, лимит-выход, Chandelier, частичный выход, пирамидинг исполняются по
high/low баров ТФ t+1 и позже тем же кодом автомата (порядок событий в баре — mutations.md № 3, гэп — № 4, касание лимита — № 2).
Первый бар сессии без позиции; решение на последнем баре не исполняется; позиция, открытая на последнем баре сессии,
закрывается по его close и платит выход (№ 5); через ночь ничего не переносится.
Д2: вход по close бара сигнала (k = None) и выход по сигналу по close бара сигнала (vyhod_lim = False) запрещены (ValueError).
σ̂ бара — СКО (ddof=1) внутрисессионных логдоходностей за 20 баров ≤ t (мин. 10); ATR — среднее истинного диапазона за 14 баров ≤ t.
  ↺ — Chandelier: экстремум до 11 баров ТФ внутри сессии (нет окна — кандидата нет); τ, T — в барах ТФ · откат: считать по дням (29.10).
  ↺ — доход сделки логарифмический (как автомат), у П5 простой · откат: tf_os_v5.dohod (29.10).
  ↺ — лимит: касание (low = L) заявку не исполняет, нужен пробой (tick = None, № 2) · откат: Params.tick = 0 (29.10).
tf = den здесь не считается: дневной движок не меняется.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fastbt import tf_os_v5 as T
from fastbt.exec_e import _check_params, _new_axes, unit_series
from fastbt.exec_e2 import Run2
from fastbt.exec_e_rules import atr_series
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day, Params

# Точка исполнения для замера: лимит-вход 0,25σ, стоп 2σ (при касании), тейк 2σ, Chandelier 3·ATR, выход по сигналу — лимитом на t+1.
PARAMS = Params(k=0.25, b=2.0, a=2.0, chand=3.0, vyhod_lim=True, stop_touch=True, comm=T.KOMISSIYA)
OKNO_SIGMA, MIN_SIGMA, OKNO_ATR = 20, 10, 14


@dataclass
class Sessiya:
    gross: float = 0.0
    cost: float = 0.0
    trades: list = field(default_factory=list)
    n_orders: int = 0
    n_fills: int = 0


def sigma_bar(td, den: np.ndarray) -> np.ndarray:
    """σ̂ бара t: СКО внутрисессионных логдоходностей close за 20 баров ≤ t (ночные разрывы выброшены); нет окна — nan."""
    c = td.close
    r = np.r_[np.nan, np.where(den[1:] == den[:-1], np.log(c[1:] / c[:-1]), np.nan)]
    return pd.Series(r).rolling(OKNO_SIGMA, min_periods=MIN_SIGMA).std().to_numpy()


def atr_bar(td) -> np.ndarray:
    """ATR(14) бара t в единицах цены: среднее истинного диапазона (с close прошлого бара) за 14 баров ≤ t."""
    pc = np.r_[td.close[0], td.close[:-1]]
    tr = np.maximum(td.high - td.low, np.maximum(abs(td.high - pc), abs(td.low - pc)))
    return pd.Series(tr).rolling(OKNO_ATR, min_periods=OKNO_ATR).mean().to_numpy()


def sostoyanie(sig: np.ndarray, ex_l: np.ndarray, ex_s: np.ndarray) -> np.ndarray:
    """Сторона по сигналу на close бара t в пределах сессии (старт 0): sig ≠ 0 — сторона sig; иначе сигнальный выход — 0; иначе держим."""
    cur, out = 0, np.zeros(len(sig), np.int8)
    for t in range(len(sig)):
        s = int(sig[t])
        if s:
            cur = s
        elif (cur == 1 and ex_l[t]) or (cur == -1 and ex_s[t]):
            cur = 0
        out[t] = cur
    return out


def proverka_d2(trades: list) -> None:
    """Д2: вход строго после бара сигнала (entry_day > d_sig). Нарушение — ValueError."""
    for tr in trades:
        if tr.parent is None and tr.entry_day <= tr.d_sig:
            raise ValueError("Д2: вход на баре сигнала")


def _proverka_params(p: Params) -> None:
    if p.k is None:
        raise ValueError("Д2: k = None — вход по close бара сигнала; нужен лимит-вход (k)")
    if not p.vyhod_lim:
        raise ValueError("Д2: vyhod_lim = False — выход по сигналу по close бара сигнала; нужен лимит на t+1")
    _check_params(p)


def sessiya(ticker: str, o, h, l, c, sost, sigma, p: Params = PARAMS, atr=None) -> Sessiya:
    """Один торговый день из баров ТФ (o,h,l,c — массивы), sost — сторона по сигналу на close каждого бара.
    Возвращает сумму логдоходов (брутто), комиссию, сделки (Trade) и счётчики заявок. Сессия закрывается по close последнего бара."""
    _proverka_params(p)
    n = len(c)
    days = [Day(o[i:i + 1], h[i:i + 1], l[i:i + 1], c[i:i + 1]) for i in range(n)]
    sg = np.asarray(sigma, float)
    need_atr = p.atr_m is not None or p.chand is not None or p.k_e != 0 or "atr" in (p.ed_s, p.ed_t)
    a = None
    if need_atr:
        a = np.asarray(atr, float) if atr is not None else np.asarray(atr_series(days), float)
    un = dict(us=unit_series(days, None, p.ed_s, sg, a), ut=unit_series(days, None, p.ed_t, sg, a))
    cls = Run2 if _new_axes(p, None) else Run
    r = cls(ticker, days, np.asarray(sost), sg, p, None, None, 1.0, atr=a, **un)
    for j, i in enumerate(r.ix):
        r.step(j, i)
    if r.tr is not None:                            # конец сессии: close последнего бара, выход платит комиссию (№ 5)
        tr = r.tr
        r._exit(r.ix[-1], len(r.ix) - 1, 0, float(c[-1]), "vremya")
        tr.reason = "sessiya"
    proverka_d2(r.trades)
    if r.vne:                                       # инвариант: цена каждого исполнения — в [low; high] своего бара (БЛОКЕР #596)
        raise ValueError(f"исполнение вне [low; high] бара: {r.vne[:3]}")
    return Sessiya(float(r.gross.sum()), float(r.cost.sum()), r.trades, r.n_orders, r.n_fills)


def granicy(den: np.ndarray) -> list:
    """[(начало, конец)) баров каждой сессии."""
    s = np.r_[0, np.flatnonzero(den[1:] != den[:-1]) + 1, len(den)]
    return list(zip(s[:-1], s[1:]))


def tochka(name: str, td, ss, i: int, den, sigma, atr, m, p: Params) -> tuple:
    """(брутто, издержки, сделок, заявок, исполнено) точки i по тикеру за сессии с маской m (по началу сессии)."""
    br = iz = 0.0
    ns = no = nf = 0
    for a, b in granicy(den):
        if not m[a] or b - a < 2:
            continue
        st = sostoyanie(ss.sig[i][a:b], ss.exit_l[i][a:b], ss.exit_s[i][a:b])
        if not st.any():
            continue
        x = sessiya(name, td.open[a:b], td.high[a:b], td.low[a:b], td.close[a:b], st, sigma[a:b], p, atr[a:b])
        br, iz, ns, no, nf = br + x.gross, iz + x.cost, ns + sum(t.parent is None for t in x.trades), no + x.n_orders, nf + x.n_fills
    return br, iz, ns, no, nf


def progon(sid: str, tf: str, tds: dict, grid: list, a: str, b: str, p: Params = PARAMS, komissiya: float = T.KOMISSIYA) -> list:
    """Строки издержек точек grid семьи sid на ТФ tf (10m–2h) с исполнением Ж-111 (автомат v2); счёт — сессии в [a, b)."""
    if tf == "den":
        raise ValueError("tf = den исполняет дневной движок")
    p = dataclasses.replace(p, comm=komissiya)
    a64, b64 = np.datetime64(pd.Timestamp(a), "ns"), np.datetime64(pd.Timestamp(b), "ns")
    years = (b64 - a64) / np.timedelta64(1, "D") / 365.25
    chasti = [[] for _ in grid]
    for name, td in tds.items():
        ss, den = T.signaly(sid, td, grid, tf), T.den_id(td)
        sigma, atr = sigma_bar(td, den), atr_bar(td)
        m = (td.dates >= a64) & (td.dates < b64)
        n_d = len(np.unique(den[m]))
        for i in range(len(grid)):
            br, iz, ns, no, nf = tochka(name, td, ss, i, den, sigma, atr, m, p)
            chasti[i].append((br, iz, ns, n_d, no, nf))
    rows = []
    for g, ch in zip(grid, chasti):
        row = T.stroka_izderzhek(sid, tf, g, [x[:4] for x in ch], years, komissiya)
        no, nf = sum(x[4] for x in ch), sum(x[5] for x in ch)
        rows.append(dict(row, zayavok=no, zapolneno_pct=100 * nf / no if no else float("nan")))
    return rows
