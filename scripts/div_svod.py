"""Свод сверки дивидендов по Д8 (tolkovaniya.md, 30.10 02:40): `docs/issledovaniya/*div_sverka*_*.csv` -> `div_sverka_svod.csv`.

Строка годна для оси дивидендов (godna=1), если verified=yes либо verified=rashozhdenie при совпавших реестре и сумме
(|dps - dps_2| <= 0,01*max(1,|dps|)). announce_pit = более поздняя из дат объявления (консервативно для point-in-time).
Круг 2 (`div_sverka2_X`) перекрывает круг 1 (`div_sverka_X`) по ключу (ticker, record_date, src_tablica).
Строки «выплат нет» (пустой record_date) в свод не идут. Колонки: ticker, record_date, dps_rub, announce_pit, godna, prichina.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs" / "issledovaniya"
SVOD = DIR / "div_sverka_svod.csv"
COLS = ["ticker", "record_date", "dps_rub", "announce_pit", "godna", "prichina"]


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _pozdnyaya(daty: list) -> str:
    """Более поздняя из непустых дат ГГГГ-ММ-ДД (пусто, если дат нет)."""
    return max([d for d in daty if d], default="")


def chitat(papka: Path = DIR) -> list[dict]:
    """Все файлы сверки; круг 2 (имя содержит `sverka2`) читается после круга 1 и перекрывает его по ключу."""
    files = sorted((p for p in glob.glob(str(Path(papka) / "*div_sverka*_*.csv")) if Path(p).name != SVOD.name),
                   key=lambda p: ("sverka2" in Path(p).name, p))
    by_key: dict = {}
    for p in files:
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["record_date"]:
                    by_key[(r["ticker"], r["record_date"], r["src_tablica"])] = r
    return list(by_key.values())


def stroka(r: dict) -> dict:
    """Одна строка свода из строки сверки."""
    ann = _pozdnyaya([r.get("announce_date"), r.get("announce_date_2")])
    v, prichina = r["verified"], ""
    if v == "yes":
        pass
    elif v == "rashozhdenie":
        a, b = _num(r["dps_rub"]), _num(r["dps_rub_2"])
        if r["record_date"] != r.get("record_date_2"):
            prichina = "reestr_ne_sovpal"
        elif a is None or b is None or abs(a - b) > 0.01 * max(1.0, abs(a)):
            prichina = "summa_ne_sovpala"
    else:
        prichina = v or "net_verified"
    if not prichina and not ann:
        prichina = "net_daty_obyavleniya"
    return {"ticker": r["ticker"], "record_date": r["record_date"], "dps_rub": r["dps_rub"], "announce_pit": ann,
            "godna": 0 if prichina else 1, "prichina": prichina}


def svod(papka: Path = DIR) -> list[dict]:
    return sorted((stroka(r) for r in chitat(papka)), key=lambda x: (x["ticker"], x["record_date"]))


def zapisat(rows: list[dict], out: Path = SVOD) -> None:
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rows = svod()
    zapisat(rows)
    print(f"выплат {len(rows)}, годных {sum(r['godna'] for r in rows)} -> {SVOD}")
