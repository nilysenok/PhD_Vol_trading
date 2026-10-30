"""Дописать key_rate_cbr.parquet до 2026-02-28 по решениям ЦБ (wf, Ж-136), point-in-time (Д3).

Значение дня d — ставка, действующая в d (data_vstupleniya <= d), а не объявленная: решение 13.02.2026
(15,5 %) вступает 16.02, на 13.02 ставка 16,0. Решения — docs/issledovaniya/2026-10-30_kluchevaya_stavka_2025_2026.csv
(первичное раскрытие cbr.ru, пресс-релизы). Строки до последней даты исходного файла не меняются.
Запуск: python3 scripts/dannye/stavka_wf_dopisat.py <исходный.parquet> <новый.parquet>; без аргументов — самопроверка.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prodlit_stavku as P  # noqa: E402


def reshenia() -> pd.DataFrame:
    r = pd.read_csv(P.CSV, usecols=["data_vstupleniya", "stavka_pct", "data_resheniya"],
                    parse_dates=["data_vstupleniya", "data_resheniya"])
    if (r["data_vstupleniya"] <= r["data_resheniya"]).any():
        raise ValueError("вступление не позже решения: нарушен point-in-time (Д3)")
    return r


def proverit(ryad: pd.DataFrame, r: pd.DataFrame, s: pd.Timestamp) -> None:
    """Новые дни (> s): значение = ставка, вступившая к дню; константа через дату вступления — ошибка (Д4)."""
    n = ryad[ryad["date"] > s]
    dni = pd.DatetimeIndex(n["date"])
    ozhid = P.stavka_na(dni, r[["data_vstupleniya", "stavka_pct"]])
    if list(n["value"]) != ozhid:
        bad = n[[a != b for a, b in zip(n["value"], ozhid)]]["date"].iloc[0]
        raise ValueError(f"{bad.date()}: значение не равно действующей ставке")


def dopisat(staryj: pd.DataFrame) -> pd.DataFrame:
    r = reshenia()
    novyj = P.prodlit(staryj, r[["data_vstupleniya", "stavka_pct"]])
    s = staryj["date"].max()
    proverit(novyj, r, s)
    if not novyj.iloc[:len(staryj)].equals(staryj):
        raise ValueError("префикс изменён")
    return novyj


def main(vhod: str, vyhod: str) -> None:
    s = pd.read_parquet(vhod)
    n = dopisat(s)
    n.to_parquet(vyhod, index=False)
    print(len(s), "->", len(n), n.tail(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        s = pd.DataFrame({"date": pd.to_datetime(["2026-02-05", "2026-02-06"]), "value": 16.0, "ticker": "K"})
        n = dopisat(s)
        v = dict(zip(n["date"].dt.strftime("%Y-%m-%d"), n["value"]))
        assert v["2026-02-13"] == 16.0 and v["2026-02-16"] == 15.5
        print("ok")
