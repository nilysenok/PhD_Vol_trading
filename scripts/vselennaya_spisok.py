"""К-а2: U_d на каждую дату смены состава 2015–2021 и список тикеров для шага 3 выгрузки (1-мин, 10-мин).

Только чтение data/regen/V40 и состава IMOEX; выгрузку не запускает. Пишет docs/tables/diag/vselennaya_U_2015_2021.csv.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastbt import vselennaya as V  # noqa: E402
from fastbt.likvidnost_e import TICKERS as NOW17  # noqa: E402

OUT = V.REPO / "docs" / "tables" / "diag" / "vselennaya_U_2015_2021.csv"
YEARS = range(2015, 2022)


def dates_for(year: int) -> list[pd.Timestamp]:
    ds = {pd.Timestamp(f"{year}-01-01")} | {d for d in V.sostav_dates() if d.year == year}
    return sorted(d for d in ds if V.sostav(d))  # до первой записи состава (1 янв 2015) U не определено


def build() -> tuple[list[dict], set[str], dict[str, set[str]]]:
    rows, ever, no_ob = [], set(), {}
    for y in YEARS:
        for d in dates_for(y):
            u = V.U(d)
            ever |= u
            miss = V.bez_oborota(d)
            if miss:
                no_ob[str(d.date())] = set(miss)
            rows.append({"date": d.date(), "year": y, "n_U": len(u), "bez_oborota": " ".join(sorted(miss)),
                         "U": " ".join(sorted(u))})
    return rows, ever, no_ob


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=V.REPO, capture_output=True, text=True)
    return r.stdout.strip() or "?"


def main() -> None:
    rows, ever, no_ob = build()
    assert all(r["n_U"] <= 40 for r in rows), "|U_d| > 40"
    new = sorted(ever - set(NOW17))
    head = [
        "# статус: разработка, данные < 2022 (оборот 2014–2021); до аудита не применять к тесту",
        f"# коммит кода: {git_head()} (рабочее дерево на момент запуска)",
        f"# хэш входов (состав + daily_value): {V.digest()}",
        f"# итого тикеров хоть раз в U_d 2015–2021: {len(ever)}; не входят в нынешние 17: {len(new)}",
        f"# весь список: {' '.join(sorted(ever))}",
        f"# вне нынешних 17: {' '.join(new)}",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(head) + "\n" + pd.DataFrame(rows).to_csv(index=False))
    print("\n".join(head))
    for r in rows:
        print(r["date"], r["n_U"], r["U"])
    print("состав без оборота в прошлом году (в U не входят):")
    for d, s in no_ob.items():
        print(" ", d, " ".join(sorted(s)))


if __name__ == "__main__":
    main()
