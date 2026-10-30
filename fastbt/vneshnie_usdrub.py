"""Ось `usdrub` после обрыва USDRUB_TOM 11.06.2024: склейка ПО ДОХОДНОСТЯМ (решение дирижёра 30.10, Ж-140).

До 11.06.2024 включительно доходность — из USDRUB_TOM, с 12.06.2024 — из CNYRUB_TOM. Каждая доходность считается
внутри СВОЕГО ряда (по соседним наблюдениям), уровни двух рядов не склеиваются: через стык нет ни одной доходности.
То же для rv (из 10-минуток своего ряда). Признак, требующий УРОВНЯ USDRUB, использует только USD-ряд
(`uroven_usd`: после 11.06.2024 — нет значения, без подстановки CNY, Д4); в v6 такого признака нет — оси К25–К26 берут dx и σ.
Подключение: `istochniki_v6.Fajly.faktor("usdrub")` вызывает `faktor_usdrub` (вызывающий модуль правит другой лист).
Константа SPLICE_BY_LEVEL — для теста-мутации (подмена в памяти): True = доходность по склеенному уровню.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import vneshnie_1850 as v18

USD_LAST = pd.Timestamp("2024-06-11")   # последний день USDRUB_TOM; CNYRUB_TOM — с USD_LAST + 1 день
SPLICE_BY_LEVEL = False


def _chasti(usd: pd.Series, cny: pd.Series) -> pd.Series:
    if SPLICE_BY_LEVEL:                                   # мутация: уровни CNY приставлены к USD
        return np.log(pd.concat([usd[usd.index <= USD_LAST], cny[cny.index > USD_LAST]])).diff()
    return pd.concat([np.log(usd).diff().loc[:USD_LAST], np.log(cny).diff().loc[USD_LAST + pd.Timedelta(days=1):]])


def dx_usdrub(usd_close: pd.Series, cny_close: pd.Series) -> pd.Series:
    """Дневная лог-доходность оси usdrub: USD до USD_LAST включительно, CNY после; ряд отсортирован по дате."""
    return _chasti(usd_close, cny_close).sort_index()


def rv_usdrub(usd_bars: pd.DataFrame, cny_bars: pd.DataFrame) -> pd.Series:
    """rv дня по 10-минуткам своего ряда (USD до USD_LAST, CNY после)."""
    from fastbt import sigma_real
    u = sigma_real.rv_1850(usd_bars).loc[:USD_LAST]
    if not len(cny_bars):
        return u
    return pd.concat([u, sigma_real.rv_1850(cny_bars).loc[USD_LAST + pd.Timedelta(days=1):]]).sort_index()


def faktor_usdrub(usd_bars: pd.DataFrame, cny_bars: pd.DataFrame | None) -> tuple:
    """(dx, rv22) оси usdrub. Нет баров CNY (нет файла) — только USD-часть: после 11.06.2024 значений нет, не ffill."""
    from fastbt import sigma_real
    cny_bars = cny_bars if cny_bars is not None else usd_bars.iloc[0:0]
    nul = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    dx = dx_usdrub(v18.close_1850(usd_bars, "USDRUB"), v18.close_1850(cny_bars, "CNYRUB") if len(cny_bars) else nul)
    return dx, sigma_real.sigma_rv(rv_usdrub(usd_bars, cny_bars), 22)


def uroven_usd(usd_close: pd.Series) -> pd.Series:
    """Уровень USDRUB: только настоящий USD-ряд, после USD_LAST значений нет (CNY-уровень в USD не переводится)."""
    return usd_close.loc[:USD_LAST]
