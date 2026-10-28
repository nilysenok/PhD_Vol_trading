#!/usr/bin/env python3
"""etap0, шаг 0: замер одного шага СВ для HAR-J (оценка + прогноз на год).

Повторяет блок HAR-J из moex_volatility/scripts/walk_forward_full.py:200:
пул 17 тикеров, LinearRegression на log-признаках (rv_d, rv_w, rv_m + 5 jump),
обучение = все годы до теста минус последний (валидационный), h = 1, 5, 22.
Тестовый год - первый аргумент (по умолчанию 2025, самый тяжёлый шаг).
Ничего не сохраняет. Число коэффициентов печатается.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"


def main(test_year=2025):
    t0 = time.time()
    P = ROOT / "data/prepared"
    parts = [pd.read_parquet(P / f"{n}.parquet") for n in ("train", "val", "test")]
    parts += [pd.read_parquet(P / f"walkforward_{y}.parquet") for y in range(2020, 2027)]
    al = pd.concat(parts).sort_values("date").reset_index(drop=True)
    al["year"] = pd.to_datetime(al["date"]).dt.year
    cfg = json.load(open(P / "config.json"))
    jumps = [c for c in al.columns if "jump" in c.lower() and "target" not in c.lower() and "idx_" not in c.lower()]
    feats = [f for f in cfg["har_features"] + [c for c in jumps if c not in cfg["har_features"]] if f in al.columns]
    t_load = time.time() - t0
    t1 = time.time()
    tr = al[al.year < test_year]
    val_year = tr.year.max()
    tp, td = tr[tr.year != val_year], al[al.year == test_year]
    for h in (1, 5, 22):
        tgt = f"rv_target_h{h}"
        a, b = tp.dropna(subset=[tgt]), td.dropna(subset=[tgt])
        X = lambda d: np.nan_to_num(np.log(d[feats].values.clip(1e-10)), nan=0, posinf=0, neginf=0)
        m = LinearRegression().fit(X(a), np.log(a[tgt].values + 1e-10))
        pred = np.exp(m.predict(X(b)))
        print(f"h={h}: n_train={len(a)} n_test={len(b)} коэф={len(m.coef_)}+1 const, pred mean={pred.mean():.3e}")
    print(f"признаки: {feats}; тикеров: {al.ticker.nunique()}")
    print(f"загрузка {t_load:.1f} с; шаг (3 h) {time.time() - t1:.2f} с; всего {time.time() - t0:.1f} с")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2025)
