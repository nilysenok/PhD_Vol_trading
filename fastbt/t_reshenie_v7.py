"""T_reshenie (Ж-146 п. 6, лист Л6.1): состояние дня на момент T и цена исполнения в T из 10-минуток.

Состояние дня на T — `daily_1850.aggregate(..., cut_min=T)`: бар входит, если его начало < T (бар, начинающийся в T, —
нет). T = 18:50 даёт прежнее состояние дня побайтно. Цена исполнения в T: open 10-минутного бара, начинающегося в T;
для 18:40 (аукцион закрытия) — close этого бара (умолчание плана §7). День без бара T — NaN с флагом, без ffill (Д4).
Модуль данных не читает: бары передаёт вызывающий (чтение < 2020-01-01 и период_v2 — на нём)."""
from __future__ import annotations

import re

import pandas as pd

from fastbt import daily_1850 as D

AUKCION = 18 * 60 + 40          # бар 18:40 покрывает аукцион закрытия: цена — его close


def _minuty(T) -> int:
    """'HH:MM' → минуты от полуночи; иначе или вне (09:50; 24:00) — ValueError."""
    m = re.fullmatch(r"(\d{2}):(\d{2})", T) if isinstance(T, str) else None
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"T={T!r}: нужен вид 'ЧЧ:ММ'")
    t = int(m.group(1)) * 60 + int(m.group(2))
    if t <= D.OPEN_MIN:
        raise ValueError(f"T={T}: не позже начала дня 09:50")
    return t


def td_T(bars10: pd.DataFrame, ticker: str, T: str) -> pd.DataFrame:
    """Дневные OHLCV только из баров с началом < T (колонки как у daily_1850.aggregate)."""
    return D.aggregate(bars10, ticker, cut_min=_minuty(T))


def cena_T(bars10: pd.DataFrame, T: str) -> pd.DataFrame:
    """Цена исполнения в T по дням состояния td_T: колонки date, cena, net_bara_T (True и NaN, если бара T нет)."""
    t = _minuty(T)
    ts = bars10["timestamp"]
    mod = ts.dt.hour * 60 + ts.dt.minute
    dni = pd.Index(D.aggregate(bars10, "_", cut_min=t)["date"])
    b = bars10[mod == t]
    pole = "close" if t == AUKCION else "open"
    c = pd.Series(b[pole].to_numpy(), index=b["timestamp"].dt.normalize().to_numpy())
    c = c[~c.index.duplicated(keep="last")].reindex(dni)
    return pd.DataFrame({"date": dni, "cena": c.to_numpy(dtype=float), "net_bara_T": c.isna().to_numpy()})
