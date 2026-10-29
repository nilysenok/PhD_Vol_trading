"""Ось таймфрейма tf для семей S1–S6 сетки v5 (Ж-125 п. 3, лист П5): tf ∈ {den, 10m, 30m, 1h, 2h}.

Бары — fastbt.bars_agg поверх 10-минуток (окно [10:00; 18:50), бар ТФ не выходит за день; проверка
`proverka_sessii` — до сигналов). Сигнал — та же функция семьи v5 (`get_family(sid, "v5")`) на барах ТФ:
окна и WARMUP — в барах ТФ; значение на баре t зависит только от закрытых баров ≤ t.
  tf = den — бары bars_agg.daily и семья без оси побайтно; исполнение — дневной движок (здесь не меняется).
  S5 на 10m–2h: пивот прошлого дня / недели / месяца (группа «den» = календарный день, v5_s56._group_ids).
  S6 на 10m–2h: VWAP с начала сессии (накопленный по барам дня), dev = СКО(close − VWAP) за vwap_window баров.
Исполнение внутри дня (Д2): решение на close бара t исполняется по open бара t+1 того же дня; первый бар дня
без позиции; решение на последнем баре дня не исполняется (следующий бар — завтра); позиция закрывается по close
последнего бара дня (до 18:50), через ночь не переносится. Доход бара j: pos[j]·(конец_j / open_j − 1), конец_j —
open бара j+1 того же дня или close последнего бара. Издержки: комиссия × |Δpos| на каждом исполнении плюс
закрытие в конце дня — каждая сторона каждой сделки. ↺ — вход и выход рыночные по open следующего бара, без
заявок по уровню Ж-111 (лист исполнения Ж-111 — отдельно) · откат: заменить `pozicii`/`dohod` слоем Ж-111 (29.10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import bars_agg as ba
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12, v5_s34, v5_s56  # noqa: F401 — регистрация семей v5
from fastbt.signals.interface import TickerData, get_family
from fastbt.signals.s56_kern import _roll_std

TFS = ("den", "10m", "30m", "1h", "2h")
KOMISSIYA = 0.0004          # доля на сторону (0,04 %, решение владельца по комиссии)
STATUS = "разработка train4 2016-2019, до аудита; потолок, не результат"


def bary(b10: pd.DataFrame, tf: str) -> pd.DataFrame:
    """10-минутки (date, slot, OHLCV; bars_agg.prep_10m) → бары ТФ с колонкой ts (начало бара; den — дата)."""
    if tf == "den":
        f = ba.daily(b10)
        return f.assign(ts=f["date"])
    k = ba.K[tf]
    f = ba.resample(b10, k)
    f = f.assign(ts=f["date"] + pd.to_timedelta(ba.OPEN_MIN + f["slot"] * ba.STEP * k, unit="m"))
    proverka_sessii(b10, f)
    return f


def proverka_sessii(b10: pd.DataFrame, f: pd.DataFrame) -> None:
    """Бар ТФ не склеивает дни: по каждому дню open/high/low/close/volume баров ТФ = дневной бар из 10-минуток,
    время начала бара — внутри [10:00; 18:50) своего дня. Нарушение — ValueError."""
    d = ba.daily(b10).set_index("date")
    g = f.sort_values("ts").groupby("date").agg(ba._AGG)
    mod = (f["ts"] - f["date"]).dt.total_seconds() / 60
    ok = g.index.equals(d.index) and np.allclose(g.values, d[g.columns].values, rtol=0, atol=1e-9)
    if not ok or bool(((mod < ba.OPEN_MIN) | (mod >= ba.CLOSE_MIN)).any()):
        raise ValueError("бар ТФ склеивает дни или выходит за [10:00; 18:50)")


def ticker_data(f: pd.DataFrame, ticker: str) -> TickerData:
    """Бары ТФ → TickerData (dates = начало бара)."""
    f = f.sort_values("ts")
    c = f["close"].to_numpy(float)
    lr = np.r_[np.log(c[1:] / np.maximum(c[:-1], 1e-12)), 0.0]
    return TickerData(ticker, f["ts"].to_numpy("datetime64[ns]"), f["open"].to_numpy(float),
                      f["high"].to_numpy(float), f["low"].to_numpy(float), c,
                      f["volume"].to_numpy(float), lr)


def den_id(td: TickerData) -> np.ndarray:
    """Номер календарного дня бара (int64, по возрастанию)."""
    return pd.DatetimeIndex(td.dates).normalize().asi8


def vwap_sessii(den: np.ndarray):
    """vwap_fn для v5_s56.generate6: VWAP с начала сессии (типичная цена × объём, накоплено по барам дня ≤ t);
    dev = СКО(close − VWAP) за w баров (ddof=1, как у дневного S6), зажат снизу 1e-12."""
    def fn(close, high, low, volume, w):
        tpv = (high + low + close) / 3.0 * volume
        cv = pd.Series(volume).groupby(den).cumsum().to_numpy()
        vwap = pd.Series(tpv).groupby(den).cumsum().to_numpy() / np.where(cv < 1e-12, 1e-12, cv)
        dev = _roll_std(close - vwap, w)
        return vwap, np.where(np.isnan(dev) | (dev < 1e-12), 1e-12, dev)
    return fn


def signaly(sid: str, td: TickerData, grid: list, tf: str):
    """SignalSet семьи sid@v5 на барах ТФ td. tf = den — семья без оси; S6 на 10m–2h — VWAP сессии."""
    if tf not in TFS:
        raise KeyError(f"tf: {tf!r}")
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    if sid == "S6" and tf != "den":
        return v5_s56.generate6(td, ind, grid, vwap_fn=vwap_sessii(den_id(td)))
    return get_family(sid, "v5")(td, ind, grid)


def pozicii(sig: np.ndarray, ex_l: np.ndarray, ex_s: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Позиция, которую держат на баре j (с open j): решение по close бара j−1 того же дня (Д2).
    Решение: sig ≠ 0 — сторона sig; иначе сигнальный выход своей стороны — 0; иначе держим."""
    n = len(sig)
    pos = np.zeros(n, np.int8)
    cur = 0
    for t in range(n - 1):
        if den[t + 1] != den[t]:
            cur = 0                     # ночь: позиция закрыта по close t, новое решение не исполняется
            continue
        s = int(sig[t])
        if s != 0:
            cur = s
        elif (cur == 1 and ex_l[t]) or (cur == -1 and ex_s[t]):
            cur = 0
        pos[t + 1] = cur
    return pos


def dohod(pos: np.ndarray, o: np.ndarray, c: np.ndarray, den: np.ndarray, komissiya: float = KOMISSIYA):
    """(брутто, издержки, оборот) по барам. Оборот бара: |pos_j − pos_{j−1}| (на первом баре дня pos_{j−1} = 0)
    плюс |pos_j| на последнем баре дня (закрытие по close)."""
    first = np.r_[True, den[1:] != den[:-1]]
    last = np.r_[den[1:] != den[:-1], True]
    p = pos.astype(float)
    end = np.where(last, c, np.r_[o[1:], np.nan])
    gross = p * (end / o - 1.0)
    prev = np.where(first, 0.0, np.r_[0.0, p[:-1]])
    turn = np.abs(p - prev) + np.where(last, np.abs(p), 0.0)
    return gross, turn * komissiya, turn


def sdelki(pos: np.ndarray, den: np.ndarray) -> int:
    """Число сделок (входов): бар, где позиция ≠ 0 и отличается от предыдущей того же дня (разворот — новая сделка)."""
    first = np.r_[True, den[1:] != den[:-1]]
    prev = np.where(first, 0, np.r_[0, pos[:-1]])
    return int(((pos != 0) & (pos != prev)).sum())


def stroka_izderzhek(sid: str, tf: str, tochka: dict, chasti: list, years: float,
                     komissiya: float = KOMISSIYA) -> dict:
    """Строка издержек точки по тикерам: chasti = [(брутто, издержки, сделки, дней), ...]; доходы — % в год на тикер."""
    br, iz = sum(x[0] for x in chasti), sum(x[1] for x in chasti)
    n_s, n_d = sum(x[2] for x in chasti), sum(x[3] for x in chasti)
    norm = 100.0 / (len(chasti) * years)
    return dict(sid=sid, tf=tf, tochka=str(tochka), komissiya_storona_pct=100 * komissiya, sdelok=n_s,
                dnej=n_d, sdelok_v_den=n_s / max(n_d, 1), brutto_pct_god=br * norm,
                izderzhki_pct_god=iz * norm, netto_pct_god=(br - iz) * norm, status=STATUS)


def progon(sid: str, tf: str, tds: dict, grid: list, a: str, b: str, komissiya: float = KOMISSIYA) -> list:
    """Строки издержек для точек grid семьи sid на ТФ tf (10m–2h); tds = {тикер: TickerData}; счёт — дни в [a, b)."""
    if tf == "den":
        raise ValueError("tf = den исполняет дневной движок; здесь — только внутридневные ТФ")
    a, b = np.datetime64(pd.Timestamp(a), "ns"), np.datetime64(pd.Timestamp(b), "ns")
    years = (b - a) / np.timedelta64(1, "D") / 365.25
    chasti = [[] for _ in grid]
    for td in tds.values():
        ss, den = signaly(sid, td, grid, tf), den_id(td)
        m = (td.dates >= a) & (td.dates < b)
        n_d = len(np.unique(den[m]))
        for i in range(len(grid)):
            pos = pozicii(ss.sig[i], ss.exit_l[i], ss.exit_s[i], den)
            g, cst, _ = dohod(pos, td.open, td.close, den, komissiya)
            chasti[i].append((float(g[m].sum()), float(cst[m].sum()), sdelki(pos[m], den[m]), n_d))
    return [stroka_izderzhek(sid, tf, p, ch, years, komissiya) for p, ch in zip(grid, chasti)]
