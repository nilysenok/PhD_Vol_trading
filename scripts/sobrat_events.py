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


VHOD_WF = ROOT / "docs" / "issledovaniya" / "2026-10-30_sobytiya_2022_2026.csv"
END_WF = "2026-03-01"


def sobrat_wf() -> str:
    """wf (Ж-136): строки до 2022 — побайтно как sobrat(); за ними 2022-01…2026-02-28 из новой таблицы (без accessed/note).

    Дивиденды 2022+ сюда не входят (другая таблица, отдельный лист). Point-in-time: data_obyavleniya <= data_sobytiya."""
    old = sobrat()
    n = pd.read_csv(VHOD_WF)[COLS]
    if n[COLS].isna().any().any():
        raise ValueError("пустые значения во входе 2022–2026")
    if (n["data_obyavleniya"] > n["data_sobytiya"]).any():
        raise ValueError("объявление позже события")
    if (n["data_sobytiya"] < END).any() or (n["data_sobytiya"] >= END_WF).any():
        raise ValueError("событие вне 2022-01-01 … 2026-02-28")
    n = n.sort_values(["data_sobytiya", "ticker", "tip", "data_obyavleniya"], kind="mergesort")
    return old + n.to_csv(index=False, lineterminator="\n", header=False)


def main() -> None:
    text = sobrat()
    VYHOD.write_text(text, encoding="utf-8")
    d = pd.read_csv(VYHOD)
    print(dict(Counter(d["tip"])), len(d))
    print("sha1", hashlib.sha1(VYHOD.read_bytes()).hexdigest())


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "wf":      # python3 scripts/sobrat_events.py wf <куда писать>
        Path(sys.argv[2]).write_text(sobrat_wf(), encoding="utf-8")
    elif len(sys.argv) == 1:
        a, b = sobrat(), sobrat()
        assert a == b and a.startswith(",".join(COLS)) and "2022-" not in "".join(
            ln.split(",")[3] for ln in a.splitlines()[1:])
        main()
