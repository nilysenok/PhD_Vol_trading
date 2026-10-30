"""Свод сверки дивидендов по Д8 (tolkovaniya.md, 30.10 02:40): `docs/issledovaniya/*div_sverka*_*.csv` -> `div_sverka_svod.csv`.

Строка годна для оси дивидендов (godna=1), если verified=yes либо verified=rashozhdenie при совпавших реестре и сумме
(|dps - dps_2| <= 0,01*max(1,|dps|)). announce_pit = более поздняя из дат объявления (консервативно для point-in-time).
Круг 2 (`div_sverka2_X`) перекрывает круг 1 (`div_sverka_X`) по ключу (ticker, record_date, src_tablica).
Строки «выплат нет» (пустой record_date) в свод не идут. Колонки: ticker, record_date, dps_rub, announce_pit, godna, prichina.

Периоды 2022 – 2026-02 (W2, `*_dividendy_2022_2024.csv`, `*_dividendy_2025_2026.csv`): колонки verified нет, годность — по хостам
колонки source_url (только то, что записано в таблице): godna=1, если есть первичный хост (PERVICHNYE: эмитент/биржа) либо
не-агрегатор вместе с агрегатором, либо два не-агрегатора. Только агрегаторы (AGREGATORY) независимыми друг другу не считаются.
"""
from __future__ import annotations

import csv
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs" / "issledovaniya"
SVOD = DIR / "div_sverka_svod.csv"
AGREGATORY = {"dohod.ru", "investmint.ru", "investfuture.ru", "smart-lab.ru"}
PERVICHNYE = {"moex.com", "phosagro.ru", "novatek.ru", "company.rt.ru", "lukoil.ru", "ogk2.ru", "interrao.ru", "ir.mts.ru",
              "ir.aeroflot.ru"}
FILES_WF = ("*_dividendy_2022_2024.csv", "*_dividendy_2025_2026.csv")
COLS = ["ticker", "record_date", "dps_rub", "announce_pit", "godna", "prichina", "godna_korr"]


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
                    wf = r["src_tablica"] != "2017_2021"     # 2022+: два платежа в один реестр различаются суммой
                    by_key[(r["ticker"], r["record_date"], r["src_tablica"], _num(r["dps_rub"]) if wf else None)] = r
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
            "godna": 0 if prichina else 1, "prichina": prichina, "godna_korr": _korr(r)}


def _korr(r: dict) -> int:
    """godna_korr (корректировка цен #678, tolkovaniya 30.10 14:55): подтверждены реестр и сумма; дата объявления не нужна."""
    a, b = _num(r["dps_rub"]), _num(r.get("dps_rub_2"))
    if a is None or b is None or not r.get("record_date_2") or r["record_date"] != r["record_date_2"]:
        return 0
    return int(abs(a - b) <= 0.01 * max(1.0, abs(a)))


def hosty(url: str) -> set:
    """Хосты из ссылок колонки source_url (без www.)."""
    return {h.lower().removeprefix("www.") for h in re.findall(r"https?://([^/\s;)]+)", url or "")}


def godna_wf(url: str) -> str:
    """'' — годна по Д8; иначе причина. Подтверждение только по ссылкам таблицы."""
    h = hosty(url)
    agr, drugie = h & AGREGATORY, h - AGREGATORY
    if (h & PERVICHNYE) or (drugie and agr) or len(drugie) >= 2:
        return ""
    return "tolko_agregatory" if h else "net_istochnika"


def chitat_wf(papka: Path = DIR) -> list[dict]:
    """Строки 2022 – 2026-02 в виде строк свода; ключ (ticker, record_date, dps): два платежа в один реестр не склеиваются."""
    sverka = {(r["ticker"], r["record_date"], _num(r["dps_rub"])) for r in chitat(papka) if r["src_tablica"] in ("2022_2024", "2025_2026")}
    by_key: dict = {}
    for pat in FILES_WF:
        for p in sorted(glob.glob(str(Path(papka) / pat))):
            with open(p, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    k = (r["ticker"], r["record_date"], _num(r["dps_rub"]))
                    if r["record_date"] and (k[2] or 0) > 0 and k not in sverka:   # сверенные строки идут из chitat
                        by_key[k] = dict(r, wf2224="2022_2024" in Path(p).name)
    out = []
    for r in by_key.values():
        ann, prichina = r.get("announce_date") or "", godna_wf(r.get("source_url"))
        if r["wf2224"]:
            prichina = "net_sverki_wf"       # 2022–2024: одного dohod/ссылки в колонке мало, нужен файл сверки div_sverka_wf_*
        if not prichina and not ann:
            prichina = "net_daty_obyavleniya"
        out.append({"ticker": r["ticker"], "record_date": r["record_date"], "dps_rub": r["dps_rub"], "announce_pit": ann,
                    "godna": 0 if prichina else 1, "prichina": prichina,
                    "godna_korr": int(prichina in ("", "net_daty_obyavleniya"))})
    return out


def svod(papka: Path = DIR) -> list[dict]:
    rows = [stroka(r) for r in chitat(papka)] + chitat_wf(papka)
    return sorted(rows, key=lambda x: (x["ticker"], x["record_date"]))


def zapisat(rows: list[dict], out: Path = SVOD) -> None:
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rows = svod()
    zapisat(rows)
    print(f"выплат {len(rows)}, годных {sum(r['godna'] for r in rows)} -> {SVOD}")
