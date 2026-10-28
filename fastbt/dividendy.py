"""Дивиденды в PnL, сигналах и стопах (Ж-057 п. 1, Ж-059 п. 2, Ж-060; расследование — PR #362).

Цены ряда 18:50 сырые: в день отсечки закрытие падает на D, и позиция через отсечку теряет (лонг) или
получает (шорт) D, которого на самом деле не было. Три места:

1. PnL. Для бара b = g−1 (последний бар <= last_cum_date; g — первый торговый день после него, «день гэпа»)
   lr[b] = log((c[g] + D_нал) / c[b]),  D_нал = D·DIV_NALOG, DIV_NALOG = 0,87 (НДФЛ 13 %; чувствительность — 1,0).
   Знак даёт позиция (pos·lr): лонг получает D, шорт платит D. Дни без отсечки не меняются побайтно. Сплиты не трогаются.
2. Сигналы и индикаторы — на ряду, обратно скорректированном: цены баров до g умножаются на (1 − D/c[g−1]) (полный D:
   цена падает на весь дивиденд). Исполнение (5-минутки, close Н6) — на реальных ценах.
3. Стоп открытой позиции через отсечку сдвигается вниз на полный D (лонг и шорт) — fastbt/exec_e_run.py.

Таблица — dohod.ru, `docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv` (реестр и размер сверены на 3 тикерах).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TABLE = Path(__file__).resolve().parent.parent / "docs" / "issledovaniya" / "2026-10-27_dividendy_2017_2021.csv"
DIV_NALOG = 0.87          # доля дивиденда, попадающая в PnL (после НДФЛ 13 %); div_nalog = 1.0 — полный D


def digest() -> str:
    """Д5: хэш таблицы дивидендов и налогового множителя — входит в ключи кэшей."""
    return hashlib.sha1(Path(TABLE).read_bytes() + repr(float(DIV_NALOG)).encode()).hexdigest()


def events(ticker: str) -> pd.DataFrame:
    """События тикера: last_cum_date (datetime64), dps_rub > 0; по возрастанию даты."""
    t = pd.read_csv(TABLE, parse_dates=["last_cum_date"])
    t = t[(t["ticker"] == ticker) & (t["dps_rub"] > 0)]
    return t[["last_cum_date", "dps_rub"]].sort_values("last_cum_date").reset_index(drop=True)


def gap_divs(ticker: str, dates: np.ndarray) -> np.ndarray:
    """D[i] — полный дивиденд (руб./акция), «падающий» на день i: i — первый торговый день после last_cum_date."""
    out = np.zeros(len(dates))
    for cum, dps in events(ticker).itertuples(index=False):
        g = int(np.searchsorted(dates, np.datetime64(cum), side="right"))     # первый бар > last_cum_date
        if 0 < g < len(dates):                                                # вне ряда: гэпа нет
            out[g] += float(dps)
    return out


def adjust_lr(ticker: str, dates: np.ndarray, close: np.ndarray, lr: np.ndarray, nalog: float | None = None) -> np.ndarray:
    """Копия lr: lr[g-1] = log((c[g] + D·nalog) / c[g-1]) в день гэпа g; вне отсечек значения те же побайтно."""
    k = DIV_NALOG if nalog is None else nalog
    out = np.array(lr, dtype=np.float64, copy=True)
    div = gap_divs(ticker, dates)
    for g in np.flatnonzero(div):
        out[g - 1] = np.log((close[g] + div[g] * k) / np.maximum(close[g - 1], 1e-12))
    return out


def adjust_ohlc(ticker: str, dates: np.ndarray, close: np.ndarray, *bars: np.ndarray) -> tuple:
    """Обратная корректировка для сигналов: цены баров до дня гэпа g × (1 − D/c[g−1]); close — реальный (для c[g−1]).
    Возвращает копии bars в том же порядке; объём не корректируется (его здесь нет)."""
    div = gap_divs(ticker, dates)
    f = np.ones(len(dates))
    for g in np.flatnonzero(div):
        m = 1.0 - div[g] / close[g - 1]
        if not m > 0:
            raise ValueError(f"{ticker}: дивиденд {div[g]} не меньше цены {close[g - 1]} на {dates[g - 1]}")
        f[:g] *= m
    return tuple(np.asarray(b, dtype=np.float64) * f for b in bars)
