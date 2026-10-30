"""Продление key_rate_cbr.parquet до 2026-02-28 по решениям ЦБ (wf, Ж-136; тол. [30.10 15:30] п. 3).

Вход: прежний parquet (date, value, ticker; до 2026-02-06) и docs/issledovaniya/2026-10-30_kluchevaya_stavka_2025_2026.csv.
Новые дни: рабочие дни после последней даты файла по 2026-02-28, кроме 2026-02-23 (выходной по таблице hd_base ЦБ,
в таблице 23.02 нет). Значение дня = ставка последнего решения, вступившего в силу к этому дню (фактическая ставка,
не подстановка — Д4): 16,0 до 13.02, 15,5 с 16.02 (решение 13.02.2026). Прежние строки — без изменений.
Запуск: python3 scripts/dannye/prodlit_stavku.py <старый.parquet> <новый.parquet>; без аргументов — самопроверка.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "docs" / "issledovaniya" / "2026-10-30_kluchevaya_stavka_2025_2026.csv"
KONEC = pd.Timestamp("2026-02-28")
NET_V_TABLICE = {pd.Timestamp("2026-02-23")}      # hd_base ЦБ: строки за 23.02.2026 нет (выходной)


def stavka_na(dni: pd.DatetimeIndex, reshenia: pd.DataFrame) -> list:
    """Ставка на каждый день: последняя строка с data_vstupleniya <= день. Раньше первой строки — ошибка (не ffill назад)."""
    r = reshenia.sort_values("data_vstupleniya")
    out = []
    for d in dni:
        k = r[r["data_vstupleniya"] <= d]
        if k.empty:
            raise ValueError(f"{d.date()}: нет решения, вступившего в силу")
        out.append(float(k["stavka_pct"].iloc[-1]))
    return out


def prodlit(staryj: pd.DataFrame, reshenia: pd.DataFrame) -> pd.DataFrame:
    posl = staryj["date"].max()
    if posl >= KONEC:
        raise ValueError("ряд уже продлён")
    dni = pd.bdate_range(posl + pd.Timedelta(days=1), KONEC)
    dni = pd.DatetimeIndex([d for d in dni if d not in NET_V_TABLICE])
    novye = pd.DataFrame({"date": dni, "value": stavka_na(dni, reshenia), "ticker": staryj["ticker"].iloc[-1]})
    return pd.concat([staryj, novye], ignore_index=True)


def chitat_reshenia() -> pd.DataFrame:
    r = pd.read_csv(CSV, usecols=["data_vstupleniya", "stavka_pct"], parse_dates=["data_vstupleniya"])
    return r


def main(vhod: str, vyhod: str) -> None:
    s = pd.read_parquet(vhod)
    n = prodlit(s, chitat_reshenia())
    n.to_parquet(vyhod, index=False)
    print(len(s), "->", len(n), n.tail(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        s = pd.DataFrame({"date": pd.to_datetime(["2026-02-05", "2026-02-06"]), "value": 16.0, "ticker": "K"})
        n = prodlit(s, chitat_reshenia())
        v = dict(zip(n["date"].dt.strftime("%Y-%m-%d"), n["value"]))
        assert v["2026-02-13"] == 16.0 and v["2026-02-16"] == 15.5 and v["2026-02-20"] == 15.5
        assert "2026-02-23" not in v and n["date"].max() == pd.Timestamp("2026-02-27")
        print("ok")
