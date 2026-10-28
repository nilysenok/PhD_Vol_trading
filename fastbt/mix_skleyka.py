"""Склейка фьючерса MIX по экспирациям (Ф4, Ж-093 п. 3): ряд OPEN и доходность open→open без скачка ролла.

Контракты `data/iss_mix_2015_2019/MX{H,M,U,Z}{цифра}*.csv`. Поля `LSTDELDATE` в выгрузке нет, экспирация = последний
торговый день контракта с ценой (сверено 28.10: MXH6 — 15.03.2016 (до 2016 экспирация 15-го), MXH7 — 16.03.2017,
с 2017 — третий четверг; третий четверг для 2015–2016 дал бы NaN в доходности за 2 дня до конца данных).
Ролл — за 2 торговых дня до экспирации: до ролла (включительно) активен ближний контракт, доходность дня t считается
ВНУТРИ одного контракта, open[t+1]/open[t] − 1, поэтому скачка цены между контрактами в ряду нет. После последнего
контракта в каталоге (с 18.12.2019; контрактов 2020 нет, данные ≥ 2020 не читаются) доходность nan — хедж 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KATALOG = Path(__file__).resolve().parent.parent / "data" / "iss_mix_2015_2019"
DNEY_DO_ROLLA = 2
GRANICA = pd.Timestamp("2020-01-01")


def zagruzit_kontrakty(katalog: Path = KATALOG) -> dict:
    """{secid: DataFrame(index=TRADEDATE, OPEN)}; строки с пустым OPEN и даты ≥ 2020 отброшены."""
    out = {}
    for f in sorted(Path(katalog).glob("MX*.csv")):
        df = pd.read_csv(f, usecols=["TRADEDATE", "SECID", "OPEN"], parse_dates=["TRADEDATE"])
        df = df[(df["TRADEDATE"] < GRANICA) & df["OPEN"].notna()]
        if len(df):
            out[df["SECID"].iloc[0]] = df.set_index("TRADEDATE")[["OPEN"]]
    return out


def ekspiraciya(kontrakt: pd.DataFrame) -> pd.Timestamp:
    """Последний торговый день контракта с ценой."""
    return kontrakt.index.max()


def data_rolla(kontrakt: pd.DataFrame, daty: pd.DatetimeIndex, dney: int = DNEY_DO_ROLLA) -> pd.Timestamp:
    """Последний день, когда контракт ещё активен: торговый день за `dney` дней до экспирации."""
    k = int(daty.searchsorted(ekspiraciya(kontrakt), side="right")) - 1   # последний торговый день ≤ экспирации
    return daty[max(k - dney, 0)]


def skleit(kontrakty: dict, daty, dney: int = DNEY_DO_ROLLA) -> pd.DataFrame:
    """Колонки: secid (активный на дату), open, ret_oo (open t → open t+1 внутри secid); nan — нет цены."""
    daty = pd.DatetimeIndex(daty)
    rolly = sorted((data_rolla(v, daty, dney), s) for s, v in kontrakty.items())
    sec, op, ret = [], [], []
    for i, t in enumerate(daty):
        s = next((s for r, s in rolly if r >= t), None)
        o = kontrakty[s]["OPEN"] if s else None
        o0 = float(o.get(t, np.nan)) if s else np.nan
        o1 = float(o.get(daty[i + 1], np.nan)) if s and i + 1 < len(daty) else np.nan
        if o0 <= 0 or o1 <= 0:                             # nan <= 0 ложно: дыра — nan, ноль — ошибка данных (Д4)
            raise ValueError(f"OPEN <= 0 у {s} вокруг {t.date()}: ноль в данных не подменяется")
        sec.append(s)
        op.append(o0)
        ret.append(o1 / o0 - 1.0)                              # nan, если цены нет на t или t+1
    return pd.DataFrame({"secid": sec, "open": op, "ret_oo": ret}, index=daty)


if __name__ == "__main__":
    k = zagruzit_kontrakty()
    print(len(k), "контрактов; последняя дата", max(v.index.max() for v in k.values()).date())
