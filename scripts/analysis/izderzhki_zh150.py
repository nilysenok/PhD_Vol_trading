"""Таблица издержек Ж-150 (train4, справочно). Запуск: python3 scripts/analysis/izderzhki_zh150.py <sdelki_train4.csv> <кэш v6> <выход-каталог> [razlozhenie_sdelki.csv A_strogaya].
Пишет tablica.csv и tablica.md. Данных ≥ 2020 не читает (строки отбрасываются при загрузке)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import izderzhki_zh150 as iz  # noqa: E402


def a_strogaya(path) -> list:
    """A_strogaya: журнал есть только как pnl_pct по сделке (без выхода, издержек, ранга) — разложение невозможно; строка-оговорка."""
    n = len(pd.read_csv(path)) if path and Path(path).exists() else 0
    txt = ("нет журнала сделок с издержками: razlozhenie_sdelki.csv — только pnl_pct после издержек (%d строк), без удержания, комиссии и ранга"
           % n if n else "нет журнала сделок")
    return [{"razdel": "6_ogranicheniya", "src": "A_strogaya", "srez": txt}]


def md(t: pd.DataFrame) -> str:
    t = t.fillna("")
    return "\n".join(["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)])


def main(a: list) -> int:
    if len(a) < 3:
        print(__doc__)
        return 2
    df = pd.concat([iz.chitat_golaya(a[0]), iz.chitat_v6(a[1])], ignore_index=True)
    t = pd.concat([iz.tablica(df), pd.DataFrame(a_strogaya(a[3] if len(a) > 3 else None))], ignore_index=True)
    t["otbrosheno_za_2016_2019"] = ""
    t.loc[t.razdel == "6_ogranicheniya", "otbrosheno_za_2016_2019"] = str(iz.OTBROSHENO)
    out = Path(a[2])
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "tablica.csv", index=False)
    cut = t.drop(columns=["primechanie","otbrosheno_za_2016_2019"]).round(3)
    (out / "tablica.md").write_text(
        f"# Издержки Ж-150, train4 2016–2019 (справочно, потолок; 2022–2024 не считались)\n\n**{t.iloc[0].srez}**\n\n"
        "Издержки = комиссия + плата за шорт; доля — от валового PnL. Sharpe дн. приближённо (PnL на дату выхода). v6: ранг/сила/тип входа "
        "и плата за шорт по сделке не измерены. Блок 5 (k_окуп) — справочно, не отбор; корзина удержания известна лишь ex post.\n\n"
        + md(cut.iloc[1:]) + f"\n\nОтброшено строк вне 2016–2019: {iz.OTBROSHENO}\n", encoding="utf-8")
    print(t.iloc[0].srez)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
