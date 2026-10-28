"""М7: сравнение B на исходной матрице и на features_pit — валидационный QLIKE, только шаги 1–6 (val 2016–2021,
разработочный набор; шаги 7–9 скрыты). Формула и агрегат — как scripts/analysis/etap5_B_val.py (QLIKE пула по шагу, среднее по шагам).
Запуск: python3 scripts/analysis/m0_B_pit_cmp.py --base <data/regen копии B> --h 1
Выход: docs/otchety/m0/B_pit_cmp.csv (дописывается). Код выхода 10, если |сдвиг| >= 2 % хотя бы у одной модели."""
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--h", type=int, required=True)
ap.add_argument("--tag", default="Bpit")
a = ap.parse_args()


def qlike(y, p):
    r = y / np.clip(p, 1e-10, None)
    return float(np.mean(r - np.log(r) - 1))


rows, big = [], False
for m in ("xgb", "lgbm"):
    b = pd.read_parquet(Path(a.base) / f"B_{m}_h{a.h}_val.parquet")
    n = pd.read_parquet(ROOT / f"data/regen/{a.tag}_{m}_h{a.h}_val.parquet")
    qb = np.mean([qlike(g.rv_actual.values, g.rv_pred.values) for _, g in b[b.step.isin(range(1, 7))].groupby("step")])
    qn = np.mean([qlike(g.rv_actual.values, g.rv_pred.values) for _, g in n[n.step.isin(range(1, 7))].groupby("step")])
    shift = 100 * (qn / qb - 1)
    big |= abs(shift) >= 2
    rows.append(dict(h=a.h, model=m, qlike_val_B=qb, qlike_val_Bpit=qn, shift_pct=shift))
d = pd.DataFrame(rows)
f = ROOT / "docs/otchety/m0/B_pit_cmp.csv"
d.to_csv(f, mode="a", header=not f.exists(), index=False)
print(d.round(4).to_string(index=False))
print("сдвиг >= 2 %: перегнать все h" if big else "сдвиг < 2 %: остальные h не перегонять")
sys.exit(10 if big else 0)
