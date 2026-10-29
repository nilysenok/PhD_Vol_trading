"""Дивиденды в PnL, сигналах и стопах (Ж-057 п. 1, Ж-059 п. 2, Ж-060; расследование — PR #362).

Цены ряда 18:50 сырые: в день отсечки закрытие падает на D, и позиция через отсечку теряет (лонг) или
получает (шорт) D, которого на самом деле не было. Три места:

1. PnL. Для бара b = g−1 (последний бар <= last_cum_date; g — первый торговый день после него, «день гэпа»)
   lr[b] = log((c[g] + D_нал) / c[b]),  D_нал = D·DIV_NALOG, DIV_NALOG = 0,87 (НДФЛ 13 %; чувствительность — 1,0).
   Знак даёт позиция (pos·lr): лонг получает D, шорт платит D. Дни без отсечки не меняются побайтно. Сплиты не трогаются.
2. Сигналы и индикаторы — на ряду, ПРЯМО скорректированном (Ж-137 п. 7): цены баров с g делятся на (1 − D/c[g−1]) (полный D:
   цена падает на весь дивиденд); до g ряд сырой, будущие выплаты прошлое не меняют. Исполнение (5-минутки, close Н6) — на реальных ценах.
3. Стоп открытой позиции через отсечку сдвигается вниз на полный D (лонг и шорт) — fastbt/exec_e_run.py.

Таблица — dohod.ru, `docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv` (реестр и размер сверены на 3 тикерах).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_DOCS = Path(__file__).resolve().parent.parent / "docs" / "issledovaniya"
TABLE = _DOCS / "2026-10-27_dividendy_2017_2021.csv"
# 2015–2016 (PR #385): потолок — только dohod.ru, первичные не сверены (primary_checked=no); last_cum_date в файле нет,
# считается здесь тем же правилом T+2, что и для 2017–2021 (docs/issledovaniya/2026-10-27_dividendy.md), по календарю дневных баров.
TABLE_2015 = _DOCS / "2026-10-28_dividendy_2015_2016.csv"
RULE = "last_cum=T+2bars:all-years"   # Д5: правило даты входит в ключ кэша
# MOEX: расчёты по акциям T+1 с 31.07.2023 (источник уточняет исследователь отдельно; PR #535 — режим NET).
T1_OT = "2023-07-31"
RULE_T1 = f"last_cum=T+1bars:record>={T1_OT}"   # в ключ — только вне dev: dev-digest побайтно прежний
DIV_NALOG = 0.87        # доля дивиденда, попадающая в PnL (после НДФЛ 13 %); div_nalog = 1.0 — полный D


def _tables() -> list[Path]:
    from fastbt import zona                              # noqa: PLC0415
    if zona.div_net():                                   # NET: без таблицы 2022–2024, явно (не пустой файл)
        z2 = []
    else:
        z2 = [] if zona.name() == "dev" else [zona.path("div")]      # зона Z2: + таблица 2022–2024 (FASTBT_Z2_DIV)
    return [Path(TABLE), Path(TABLE_2015), *z2]


def bez_dividendov() -> bool:
    """True — тест-1 считается без дивидендов 2022–2024 (FASTBT_Z2_DIV=NET); режим явный, шапки csv помечаются."""
    from fastbt import zona                              # noqa: PLC0415
    return zona.div_net()


def digest() -> str:
    """Д5: хэш обеих таблиц дивидендов и налогового множителя — входит в ключи кэшей."""
    h = hashlib.sha1()
    if bez_dividendov():
        h.update(b"bez_dividendov:2022-2024")            # режим в ключе: кэш без дивидендов ≠ кэш с таблицей
    for p in _tables():
        h.update(p.read_bytes())
    from fastbt import zona                              # noqa: PLC0415
    rule = RULE if zona.name() == "dev" else RULE + ";" + RULE_T1   # все реестры dev <= 2021: правило T+1 их не касается
    return hashlib.sha1(h.digest() + repr(float(DIV_NALOG)).encode() + rule.encode()).hexdigest()


@lru_cache(maxsize=1)
def _calendar() -> np.ndarray:
    """Торговые дни: объединение дат дневных баров 17 тикеров (ряд 18:50, 2014-06…2021-12)."""
    from fastbt import zona                                                       # noqa: PLC0415
    from fastbt.daily_1850 import out_path                                        # noqa: PLC0415
    d = pd.read_parquet(zona.path("daily", out_path()), columns=["date"])
    return np.sort(pd.to_datetime(d["date"]).dt.normalize().unique().astype("datetime64[ns]"))


from fastbt import zona as _zona                                                  # noqa: E402

_zona.on_change(_calendar.cache_clear)


def last_cum(record: pd.Series) -> pd.Series:
    """T+2: торговый день на два раньше последнего торгового дня <= даты реестра (реестр в выходной — предыдущий день).
    Календарь баров верен, а не ручной календарь таблицы 2017–2021: по ценам день гэпа 2017–2019 совпадает с T+2 по барам
    (MOEX 13.05.2019, TATN 08.01.2019), а не с табличной датой (см. docs/otchety/2026-10-28_v1_dannye.md)."""
    cal = _calendar()
    rec = record.to_numpy("datetime64[ns]")
    i = np.searchsorted(cal, rec, side="right") - 1
    lag = np.where(rec >= np.datetime64(T1_OT), 1, 2)     # T+1 для реестров >= T1_OT (расчёты T+1 по акциям)
    if (i < lag).any():
        raise ValueError("дата реестра раньше начала календаря торгов")
    return pd.Series(cal[i - lag], index=record.index)


def _read(path: Path, ticker: str) -> pd.DataFrame:
    """Одна таблица в общий вид: last_cum_date, dps_rub тикера (строки без выплат — «выплат не было» — пропуск)."""
    t = pd.read_csv(path)
    t = t[(t["ticker"] == ticker) & (t["dps_rub"] > 0)].copy()
    if len(t):                                  # PR div-daty-t2: T+2 по барам для всех лет; колонка last_cum_date — только сверка
        t["last_cum_date"] = last_cum(pd.to_datetime(t["record_date"]))
    else:
        t["last_cum_date"] = pd.Series(dtype="datetime64[ns]")
    return t[["last_cum_date", "dps_rub"]]


def events(ticker: str) -> pd.DataFrame:
    """События тикера: last_cum_date (datetime64), dps_rub > 0; по возрастанию даты."""
    t = pd.concat([_read(p, ticker) for p in _tables()], ignore_index=True)
    return t.sort_values("last_cum_date").reset_index(drop=True)


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
    """ПРЯМАЯ корректировка для сигналов (Ж-137 п. 7, Д3): цены баров с дня гэпа g ÷ (1 − D/c[g−1]); до g — сырые.
    На день d ряд зависит только от выплат с last_cum <= d: прогон до t = префикс полного (обратная умножала прошлое
    на коэффициенты будущих выплат). Отношения соседних цен те же, что у обратной (доходность через отсечку — полная);
    меняются только уровни, поэтому исчезают/появляются ничьи `close == SMA`. close — реальный (для c[g−1]).
    Возвращает копии bars в том же порядке; объём не корректируется (его здесь нет)."""
    div = gap_divs(ticker, dates)
    f = np.ones(len(dates))
    for g in np.flatnonzero(div):
        m = 1.0 - div[g] / close[g - 1]
        if not m > 0:
            raise ValueError(f"{ticker}: дивиденд {div[g]} не меньше цены {close[g - 1]} на {dates[g - 1]}")
        f[g:] /= m
    return tuple(np.asarray(b, dtype=np.float64) * f for b in bars)
