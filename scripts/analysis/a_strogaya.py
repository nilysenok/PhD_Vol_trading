"""Ряды A_strogaya (S1–S6, A) и паспорт по годам train4 (Ж-086 п. 2). Запуск: FASTBT_MAIN=<копия> FASTBT_BARS1M=<D25/bars1m> python3 scripts/analysis/a_strogaya.py
Пишет docs/tables/a_strogaya/ryad_{S1..S6,A}.parquet, pasport.csv и docs/otchety/2026-10-28_a_strogaya_pasport.md. Данные ≥ 2020 не читает."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import a_strogaya as AS  # noqa: E402

REPORT = ROOT / "docs" / "otchety" / "2026-10-28_a_strogaya_pasport.md"
SHAPKA = """# A_strogaya: паспорт по годам train4 (Ж-086 п. 2)

**Статус: разработка train4 (2016–2019), после аудита. Не результат теста.** Прогрев 2015. Комиссия 0,04 % за сторону.
Код: коммит `{kod}`; ветка `a-strogaya`. Ряды: `docs/tables/a_strogaya/ryad_*.parquet`, паспорт: `docs/tables/a_strogaya/pasport.csv`.

- **A_strogaya**: сигналы S1–S6 и точка A — как у якоря (позиции s9); вход рынком по open 5-мин бара 10:00 дня d+1, выход по сигналу
  и по τ — по open бара 10:00 дня x+1 (`t_vyhod = open2`); close дня решения в расчёте не участвует. Портфель A: равный вес по парам (тикер × Sk), без таргета.
- **Якорь A = s9** (справа) — эталон кода: вход и выход по close дня решения; сверка с s9 ≤ 1e-12 (`tests/test_yakor_v1.py`).
- Плата за шорт 5 % годовых — только за перенос через ночь (`fastbt/zaem_v2.py`); колонки «с платой 5 %».
- Метрики: одно определение на проект (`fastbt/metriki_e.py`): mSR (Фавр–Галеано, CF), Шарп (ddof = 1, √252), доходность = сумма дневного
  excess за год, MDD по накопленному капиталу. Доли — в процентах. Время в позиции — доля 5-мин баров дня в рынке, среднее по паре и дню.
  Сделок в год — все пары (для Sk — 17 тикеров, для A — 17 × 6); в строке «все» — среднее за год.
- Хэши входов (SHA-256 первых 16 символов): {hashes}.
"""


def fmt(r: dict, c: str, pct: bool = False) -> str:
    v = r[c] * (100 if pct else 1)
    return "—" if pd.isna(v) else f"{v:.1f} %" if pct else f"{v:.2f}"


def tablica(df: pd.DataFrame, plata: float) -> str:
    """Строки: ряд × год; колонки: строгая | якорь (mSR, Шарп, доходность, MDD; без платы — ещё время и сделки)."""
    cols = ["msr", "sharpe", "dohodnost", "mdd"] + (["dolya_vremeni", "sdelok"] if plata == 0 else [])
    head = ["mSR", "Шарп", "доходность", "MDD"] + (["время в позиции", "сделок/год"] if plata == 0 else [])
    out = ["| ряд | год | " + " | ".join(f"A_strogaya {h}" for h in head) + " | " + " | ".join(f"якорь {h}" for h in head) + " |",
           "|" + "---|" * (2 + 2 * len(cols))]
    d = df[df.plata_zaem == plata]
    for (ryad, god), g in d.groupby(["ryad", "god"], sort=False):
        s, y = g[g.variant == "strogaya"].iloc[0], g[g.variant == "yakor"].iloc[0]
        cell = lambda r: [fmt(r, c, c in ("dohodnost", "mdd", "dolya_vremeni")) if c != "sdelok" else f"{r[c]:.0f}" for c in cols]  # noqa: E731
        out.append(f"| {ryad} | {god} | " + " | ".join(cell(s) + cell(y)) + " |")
    return "\n".join(out)


def main() -> None:
    res, rows = {}, []
    for sid in AS.SIDS:
        anc, st = AS.evaluatory(sid)
        res[sid] = {"yakor": AS.schitat(anc, False), "strogaya": AS.schitat(st, True)}
        print(sid, "готово", flush=True)
    for v in ("yakor", "strogaya"):
        res["A"] = res.get("A", {})
        res["A"][v] = AS.portfel_A({s: res[s][v] for s in AS.SIDS})
    hashes = []
    for imya, r in res.items():
        p = AS.zapis_ryada(imya, r["strogaya"])
        hashes.append(f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
        for v in ("strogaya", "yakor"):
            rows += AS.pasport(imya, v, r[v])
    df = pd.DataFrame(rows)
    df.to_csv(AS.OUT / "pasport.csv", index=False)
    kod = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    txt = SHAPKA.format(kod=kod, hashes="; ".join(hashes))
    txt += "\n## Без платы за шорт\n\n" + tablica(df, 0.0) + "\n\n## С платой 5 % за шорт\n\n" + tablica(df, AS.PLATA) + "\n"
    REPORT.write_text(txt, encoding="utf-8")
    print("записано", REPORT)


if __name__ == "__main__":
    main()
