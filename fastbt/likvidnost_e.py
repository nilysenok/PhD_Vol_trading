"""Семья E: подвыборка «топ-10 из 17 по рублёвому обороту за предыдущий год» (ex-ante, состав фиксирован на весь год Y).

Источник оборота: data/regen/D25/bars1m/{T}_{Y}.csv, столбец value (руб., 1-минутки основной сессии), сумма за календарный год.
Нужны только годы 2017–2020 (для Y = 2018–2021); данные ≥ 2022 не читаются (assert).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from fastbt.sigma_e import MAIN

BARS1M = MAIN / "data" / "regen" / "D25" / "bars1m"
OUT = Path(__file__).resolve().parent.parent / "data" / "fastbt" / "sigma_e" / "oborot_god.csv"
TICKERS = sorted(["AFLT", "ALRS", "HYDR", "IRAO", "LKOH", "LSRG", "MGNT", "MOEX",
                  "MTLR", "MTSS", "NVTK", "OGKB", "PHOR", "RTKM", "SBER", "TATN", "VTBR"])
YEARS_OBOROT = (2017, 2018, 2019, 2020)
N_TOP = 10


def oborot_po_godam(years=YEARS_OBOROT, cache: bool = True) -> pd.DataFrame:
    """Таблица (ticker, year, oborot_rub): Σ value за год по 1-минуткам. Дыра (нет файла/тикера/нулевой оборот) — ошибка."""
    assert max(years) <= 2021, "данные ≥ 2022 не читаются"
    if cache and OUT.exists():
        t = pd.read_csv(OUT)
        if set(years) <= set(t["year"]) and set(TICKERS) <= set(t["ticker"]):
            return t[t["year"].isin(years)].reset_index(drop=True)
    rows = []
    for tk in TICKERS:
        for y in years:
            d = pd.read_csv(BARS1M / f"{tk}_{y}.csv", usecols=["begin", "value"])
            assert (pd.to_datetime(d["begin"]).dt.year == y).all(), f"{tk}_{y}: даты вне года"
            rows.append({"ticker": tk, "year": y, "oborot_rub": float(d["value"].sum())})
    t = pd.DataFrame(rows)
    assert (t["oborot_rub"] > 0).all(), "нулевой оборот тикера за год"
    if cache:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(OUT, index=False)
    return t


def top10(year: int, oborot: pd.DataFrame | None = None) -> list[str]:
    """Топ-10 по обороту за год year−1; ничьи — по тикеру. Состав один на весь год year."""
    t = oborot_po_godam() if oborot is None else oborot
    p = t[t["year"] == year - 1]
    assert set(p["ticker"]) == set(TICKERS), f"нет оборота {year - 1} по всем 17 тикерам"
    p = p.sort_values(["oborot_rub", "ticker"], ascending=[False, True])
    return sorted(p["ticker"].head(N_TOP).tolist())


def sostav_na_datu(date, oborot: pd.DataFrame | None = None) -> list[str]:
    """Состав на дату: зависит только от года даты (внутри года не меняется)."""
    return top10(pd.Timestamp(date).year, oborot)


def tablica(years=(2018, 2019, 2020, 2021), oborot: pd.DataFrame | None = None) -> pd.DataFrame:
    return pd.DataFrame({"year": list(years), "top10": [" ".join(top10(y, oborot)) for y in years]})


if __name__ == "__main__":
    print(tablica().to_string(index=False))
