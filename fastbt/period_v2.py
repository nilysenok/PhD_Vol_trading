"""Период v2 (Ж-113 п. 1, 6; Ж-114 п. 1–2): train 2015–2019, прогрев с 26.08.2014; путь A без σ-источников.

Одна точка подключения правила качества (`vselennaya_kachestvo`, #497): невалидный тикер-день — нет сигнала,
позиция держится (`hold_invalid`; вызывается из `s9._job` только при PERIOD == "v2").
Ранг с окном < 252 (C/D/E) в пути A не участвует — не проверялся здесь.
Дивиденды 2014 не подключены: таблица дивидендов начинается с 2015 (NOTE_DIV_2014); нулём не подменяется —
ряд 2014 идёт без поправки на отсечки и в метрики не входит (только прогрев).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

PERIOD = "v2"
YEARS = (2015, 2016, 2017, 2018, 2019)
VAL_START = 2014                                   # год прогрева (как 2015 у train4)
START = pd.Timestamp("2014-08-26")                 # начало прогрева: покрытие 10-минуток
NOTE_DIV_2014 = "дивиденды 2014 не подключены"


def trim_td(td):
    """TickerData с датами >= START (прогрев v2 начинается 26.08.2014, а не с начала дневного ряда)."""
    k = int(np.searchsorted(td.dates, np.datetime64(START)))
    cut = lambda a: a[k:]                          # noqa: E731
    return dataclasses.replace(td, dates=cut(td.dates), open=cut(td.open), high=cut(td.high), low=cut(td.low),
                               close=cut(td.close), volume=cut(td.volume), log_ret=cut(td.log_ret),
                               sigma={h: cut(a) for h, a in td.sigma.items()})


def valid_vec(ticker: str, dates, kach=None) -> np.ndarray:
    """Валидность тикер-дня по правилу качества; даты вне календаря 10-минуток — невалидны."""
    if kach is None:
        from fastbt import vselennaya_kachestvo as vk
        kach = vk.default()
    return kach.valid[ticker].reindex(pd.DatetimeIndex(dates), fill_value=False).to_numpy(bool)


def hold_invalid(pos: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """В невалидный день позиция = позиции предыдущего дня (нет сигнала — держится). Первый день — как есть."""
    idx = np.where(valid, np.arange(len(pos)), 0)
    idx[0] = 0
    return pos[np.maximum.accumulate(idx)]


def hold_years(a: dict, ticker: str, dates, kach=None) -> dict:
    """{год: позиции на весь ряд} → то же с удержанием в невалидные дни."""
    ok = valid_vec(ticker, dates, kach)
    return {Y: hold_invalid(v, ok) for Y, v in a.items()}
