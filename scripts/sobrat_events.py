"""Сборка data/events.csv (Ж-070 п. 2): ЦБ, IMOEX, FORTS (#392) + дивиденды (через fastbt.dividendy).

Колонки: ticker, tip, data_obyavleniya, data_sobytiya, istochnik. Только data_sobytiya < 2022.
Дивиденды: tip = div_last_cum, data_sobytiya = last_cum_date, data_obyavleniya = announce_date.
Руками csv не писать. Запуск: python3 scripts/sobrat_events.py  (печатает строки по типам и SHA1 файла).
Самопроверка без аргументов: сборка дважды в память -> одинаковые байты.
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastbt import dividendy  # noqa: E402

VHOD = ROOT / "docs" / "issledovaniya" / "2026-10-28_sobytiya_2016_2021.csv"
VYHOD = ROOT / "data" / "events.csv"
COLS = ["ticker", "tip", "data_obyavleniya", "data_sobytiya", "istochnik"]
END = "2022-01-01"


def _dividendy() -> pd.DataFrame:
    # events() отдаёт только last_cum_date и dps; announce_date берём из TABLE того же модуля
    t = pd.read_csv(dividendy.TABLE)
    t = t[t["dps_rub"] > 0]
    if t["announce_date"].isna().any():
        raise ValueError("дивиденды: пустая announce_date — point-in-time невозможен")
    if (t["announce_date"] > t["last_cum_date"]).any():
        raise ValueError("дивиденды: announce_date позже last_cum_date")
    return pd.DataFrame({"ticker": t["ticker"], "tip": "div_last_cum", "data_obyavleniya": t["announce_date"],
                         "data_sobytiya": t["last_cum_date"], "istochnik": t["source_url"]})


def sobrat() -> str:
    s = pd.read_csv(VHOD)[COLS]
    d = pd.concat([s, _dividendy()], ignore_index=True)
    if d[COLS].isna().any().any():
        raise ValueError("пустые значения во входах")
    if (d["data_obyavleniya"] > d["data_sobytiya"]).any():
        raise ValueError("объявление позже события")
    d = d[d["data_sobytiya"] < END]
    d = d.sort_values(["data_sobytiya", "ticker", "tip", "data_obyavleniya"], kind="mergesort")
    return d.to_csv(index=False, lineterminator="\n")


def main() -> None:
    text = sobrat()
    VYHOD.write_text(text, encoding="utf-8")
    d = pd.read_csv(VYHOD)
    print(dict(Counter(d["tip"])), len(d))
    print("sha1", hashlib.sha1(VYHOD.read_bytes()).hexdigest())


if __name__ == "__main__":
    if len(sys.argv) == 1:
        a, b = sobrat(), sobrat()
        assert a == b and a.startswith(",".join(COLS)) and "2022-" not in "".join(
            ln.split(",")[3] for ln in a.splitlines()[1:])
        main()
