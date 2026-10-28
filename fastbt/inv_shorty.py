"""Инвентаризация Л-С, п. 8 № 3 и № 4: арифметика по готовым журналам, прогонов нет.

№ 3: плата за шорт, на каждые 10 % годовых: Σ календарных дней шорта × вес × 0,10 / 365.
Вес сделки в журнале равен 1 (pnl — доля от входа, сумма pnl = Σ PnL сделок),
результат в п. п. Σ PnL (×100).
№ 4: добавка при полном D у шортов: 0,13/0,87 × Σ колонки
`dividend_D087_dolya_vkhoda` по сделкам short (×100).
Журналы: docs/tables/diag/S{k}_sdelki_posle370.csv (коммит кода b8881ec).
"""
import csv
import sys
from datetime import date
from pathlib import Path

STAVKA = 0.10
ADD = 0.13 / 0.87
DIAG = Path(__file__).resolve().parent.parent / "docs" / "tables" / "diag"


def chitat(path):
    """Строки журнала (без комментариев «#») как словари."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def dni(row):
    """Календарные дни шорта: дата выхода минус дата входа."""
    return (date.fromisoformat(row["data_vykhoda"]) - date.fromisoformat(row["data_vhoda"])).days


def plata_za_short(rows):
    """№ 3: −Σ дней × 0,10 / 365 в п. п. на каждые 10 % годовых."""
    return -100 * sum(dni(r) * STAVKA / 365 for r in rows if r["napravlenie"] == "short")


def dobavka_D(rows):
    """№ 4: −0,13/0,87 × Σ дивидендной доли шортов в п. п."""
    return 0.0 - 100 * ADD * sum(float(r["dividend_D087_dolya_vkhoda"])
                                 for r in rows if r["napravlenie"] == "short")


def main():
    for k in range(1, 7):
        rows = chitat(DIAG / f"S{k}_sdelki_posle370.csv")
        print(f"S{k}: plata_za_short={plata_za_short(rows):.2f} dobavka_D={dobavka_D(rows):.2f}")


if __name__ == "__main__":
    sys.exit(main())
