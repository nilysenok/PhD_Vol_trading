"""Л2.1 (Ж-146 п. 2): ось otnositelno ∈ {net, sektor, imoex} для S1–S4 на дневном ряду.

r_rel[d] = r_i[d] − r_b[d], r = close[d]/close[d−1] − 1; b — равновзвешенный сектор из НАШИХ тикеров без самого тикера
(`kross_seksiya.sektor_ryad`; нет коллег в день — IMOEX, с флагом `otn_imoex_vmesto_sektora` в extra, молча — нет) или IMOEX.
td_rel: close_rel = close[0]·Π(1 + r_rel); O/H/L дня умножены на тот же множитель дня close_rel/close; volume — тикера.
Отличие от `rel_ryad` (ось rel_signal куба v6): там 10-минутные бары, сектор — индекс MOEX, разность лог-цен; здесь дневные
доходности и сектор из тикеров панели. Д1 — `rel_ryad.proverka_d1` (не дублируется). net — исходные td и сигналы побайтно.
Решение дня d видит ряды ≤ d (Д2), исполнение d+1 — в оценщике. Файлов данных не читает (кроме карты секторов).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import kross_seksiya as ks
from fastbt.indicators import compute_base
from fastbt.signals.interface import SignalSet, TickerData

OTNOSITELNO = ("net", "sektor", "imoex")
FLAG = "otn_imoex_vmesto_sektora"
KARTA_CSV = Path(__file__).resolve().parents[1] / "docs" / "issledovaniya" / "2026-10-27_sektory.csv"


def _proverit(rezhim: str) -> None:
    if rezhim not in OTNOSITELNO:
        raise ValueError(f"otnositelno вне {OTNOSITELNO}: {rezhim!r}")


def metki_iz_csv(path, dates, tickers) -> np.ndarray:
    """Метки сектора [T, N]; единый состав на весь train4 (❓ Д-4, как в golaya_s7_s9)."""
    k = pd.read_csv(path, usecols=["ticker", "indeks"])
    k["s"] = pd.Timestamp("2000-01-01")
    return ks.sektor_na_d(k, dates, list(tickers))


def fon_ryad(c: pd.DataFrame, imoex: pd.Series, metki: np.ndarray, rezhim: str):
    """(fon, flag): доходность фона [T, N] и флаг «вместо сектора взят IMOEX» (строка 0 — без доходности, флаг False)."""
    _proverit(rezhim)
    if rezhim == "net":
        raise ValueError("net: фона нет")
    r = c / c.shift(1) - 1
    rm = (imoex / imoex.shift(1) - 1).reindex(c.index)
    if rezhim == "imoex":
        fon = pd.DataFrame(np.repeat(rm.to_numpy()[:, None], c.shape[1], axis=1), index=c.index, columns=c.columns)
        return fon, pd.DataFrame(False, index=c.index, columns=c.columns)
    fon = pd.DataFrame(ks.sektor_ryad(r, metki, rm), index=c.index, columns=c.columns)
    flag = np.ones(r.shape, bool)
    R = r.to_numpy()
    for lab in {x for x in metki.ravel() if x}:
        M = metki == lab
        ok = M & ~np.isnan(R)
        flag &= ~(M & (ok.sum(axis=1, keepdims=True) - ok >= 1))
    flag[0] = False
    return fon, pd.DataFrame(flag, index=c.index, columns=c.columns)


def td_rel(td: TickerData, fon: np.ndarray) -> TickerData:
    """Относительный дневной ряд тикера против фона (fon — доходности на датах td; fon[0] не используется)."""
    fon = np.asarray(fon, float)
    if len(fon) != td.n or not np.isfinite(fon[1:]).all():
        raise ValueError("фон: длина не равна td или есть NaN после первого дня (без подстановок, Д4)")
    f = 1 + (td.close[1:] / td.close[:-1] - 1) - fon[1:]
    if not (f > 0).all():
        raise ValueError("1 + r_rel <= 0: относительный ряд не определён")
    close = td.close[0] * np.cumprod(np.r_[1.0, f])
    m = close / td.close
    lr = np.r_[np.log(close[1:] / close[:-1]), 0.0]
    return replace(td, open=td.open * m, high=td.high * m, low=td.low * m, close=close, log_ret=lr)


def primenit(td: TickerData, rezhim: str, fon_i):
    _proverit(rezhim)
    return td if rezhim == "net" else td_rel(td, fon_i)


def semya(fn, rezhim: str, fon: pd.DataFrame, flag: pd.DataFrame = None):
    """Семья S1–S4 с осью: net — сама fn (побайтно); иначе сигнал считается на td_rel, ind пересчитан на нём."""
    _proverit(rezhim)
    if rezhim == "net":
        return fn

    def generate(td, ind, grid):
        idx = pd.DatetimeIndex(td.dates)
        if not idx.isin(fon.index).all():
            raise ValueError("фон не покрывает даты td")
        rel = td_rel(td, fon[td.ticker].reindex(idx).to_numpy())
        base = compute_base(rel.close, rel.high, rel.low, rel.open, rel.volume)
        ss = fn(rel, base, grid)
        fl = np.zeros(td.n, bool) if flag is None else flag[td.ticker].reindex(idx).to_numpy(bool)
        return SignalSet(ss.sig, ss.exit_l, ss.exit_s, ss.strength, ss.params, ss.split,
                         {**ss.extra, FLAG: fl})
    return generate
