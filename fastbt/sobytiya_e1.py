"""События этапа 1 (Ж-047 п. 3, контракт К3): фильтры входа point-in-time.

Массив bool по торговым датам: True = вход по close d запрещён. Действуют только на новый вход
(открытую позицию не снимают, контракт §1.1).
- cbr: заседание ЦБ приходится на следующий торговый день d+1 и known_from ≤ d.
- div: last_cum_date = d (последний день с правом на дивиденд) и announce_date ≤ d, по тикеру.
Ось дивидендов НЕ включается, пока даты не сверены с эмитентом (❓5): без div_verified=True
sobytiya() отдаёт только net и cbr. Даты ≥ 2022 → PermissionError.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

END = pd.Timestamp("2022-01-01")
CBR_CSV = Path("data/calendar/cbr.csv")
DIV_CSV = Path("docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv")


def _dates(dates) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(dates)
    from fastbt import zona
    if len(d) and d.max() >= zona.end():
        raise PermissionError(f"события: даты ≥ {zona.end().date()}")
    return d


def cbr_block(dates, cbr: pd.DataFrame) -> np.ndarray:
    """cbr: колонки date, known_from. Блок на d, если заседание — на следующий торговый день."""
    d = _dates(dates)
    out = np.zeros(len(d), bool)
    if len(d) < 2:
        return out
    for m, kf in zip(pd.to_datetime(cbr["date"]), pd.to_datetime(cbr["known_from"])):
        j = int(d.searchsorted(m, side="left"))     # первый торговый день ≥ даты заседания
        if j == 0 or j >= len(d):
            continue
        if kf <= d[j - 1]:                          # решение о дате известно в день d = d[j−1]
            out[j - 1] = True
    return out


def div_block(dates, div: pd.DataFrame, ticker: str) -> np.ndarray:
    """div: колонки ticker, last_cum_date, announce_date. Блок на d = last_cum_date, если announce ≤ d."""
    d = _dates(dates)
    out = np.zeros(len(d), bool)
    x = div[div["ticker"] == ticker]
    for lc, an in zip(pd.to_datetime(x["last_cum_date"]), pd.to_datetime(x["announce_date"])):
        i = int(d.searchsorted(lc, side="right")) - 1   # last_cum_date не торговый → предыдущий торговый
        if i >= 0 and an <= d[i] and d[i] <= lc:
            out[i] = True
    return out


def load_cbr(path=CBR_CSV) -> pd.DataFrame:
    c = pd.read_csv(path, parse_dates=["date", "known_from"])
    from fastbt import zona
    return c[c["date"] < zona.end()]


def load_div(path=DIV_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["record_date", "last_cum_date", "announce_date"])


def sobytiya(dates, tickers, cbr: pd.DataFrame, div: pd.DataFrame | None = None,
             div_verified: bool = False) -> dict:
    """{имя: {тикер: bool}}; net — ничего не запрещает. div/cbr_div — только при div_verified (❓5)."""
    d = _dates(dates)
    c = cbr_block(d, cbr)
    out = {"net": {t: np.zeros(len(d), bool) for t in tickers},
           "cbr": {t: c.copy() for t in tickers}}
    if div_verified:
        if div is None:
            raise ValueError("div_verified=True требует таблицу дивидендов")
        dv = {t: div_block(d, div, t) for t in tickers}
        out["div"] = dv
        out["cbr_div"] = {t: dv[t] | c for t in tickers}
    return out
