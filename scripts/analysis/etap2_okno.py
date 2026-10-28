#!/usr/bin/env python3
"""etap2: переоценка HAR-J на каждом шаге СВ для окон (расширяющееся, скользящие 1/2/3, дисконт).
Оценка - как walk_forward_full.py:193-204 (LinearRegression, пул 17 тикеров, 8 регрессоров в логах + const,
обучение на train_pure = годы < val_year, val_year = последний год обучающего окна шага; прогноз exp(yhat)).
В walk_forward_full.py оценка не вынесена в функцию (код на уровне модуля), поэтому строки продублированы.
Тестовый год в выборе окна не участвует: хранятся val- и test-прогнозы раздельно (kind).
Выход: moex_volatility/results/etap2/har_windows.parquet. Запуск: python3 scripts/analysis/etap2_okno.py"""
import json, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
OUT = ROOT / "results/etap2/har_windows.parquet"
SPECS = {"E": None, "R1": 1, "R2": 2, "R3": 3, "D2": "decay"}   # D2: полураспад 2 года (взвешенный OLS)


def load_all():
    os.chdir(ROOT)
    tr, va, te = (pd.read_parquet(f"data/prepared/{n}.parquet") for n in ("train", "val", "test"))
    cfg = json.load(open("data/prepared/config.json"))
    cols = tr.columns.tolist()
    jc = [c for c in cols if "jump" in c.lower() and "target" not in c.lower() and "idx_" not in c.lower()]
    hf = cfg["har_features"] + [c for c in jc if c not in cfg["har_features"]]
    hf = [f for f in hf if f in cols]
    parts = [tr, va, te] + [pd.read_parquet(f"data/prepared/walkforward_{y}.parquet") for y in range(2020, 2027)
                            if os.path.exists(f"data/prepared/walkforward_{y}.parquet")]
    parts = [p for p in parts if len(p) > 100]
    a = pd.concat(parts).sort_values("date").reset_index(drop=True)
    a["year"] = pd.to_datetime(a["date"]).dt.year
    return a, hf


def clean(X):
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


def fit_predict(tp, hf, target, spec, others):
    """Оценка на tp (уже обрезанном окне); возвращает (модель, s2)."""
    X = clean(np.log(tp[hf].values.clip(1e-10))); y = np.log(tp[target].values + 1e-10)
    w = None
    if spec == "decay":
        age = (pd.to_datetime(tp["date"]).max() - pd.to_datetime(tp["date"])).dt.days.values / 365.25
        w = 0.5 ** (age / 2.0)
    m = LinearRegression().fit(X, y, sample_weight=w)
    res = y - m.predict(X)
    s2 = float(np.average(res ** 2, weights=w))
    return m, s2


def main():
    a, hf = load_all()
    rows = []
    for ty in range(2017, 2027):
        full = a[a.year < ty]; td0 = a[a.year == ty]
        vy = sorted(full.year.unique())[-1]
        tpure = full[full.year != vy]; vi0 = full[full.year == vy]
        for h in (1, 5, 22):
            tg = f"rv_target_h{h}"
            vi, td = vi0.dropna(subset=[tg]), td0.dropna(subset=[tg])
            tpa = tpure.dropna(subset=[tg])
            for name, k in SPECS.items():
                tp = tpa if k in (None, "decay") else tpa[tpa.year > tpa.year.max() - k]
                m, s2 = fit_predict(tp, hf, tg, "decay" if k == "decay" else None, None)
                for kind, d in (("val", vi), ("test", td)):
                    yh = m.predict(clean(np.log(d[hf].values.clip(1e-10))))
                    rows.append(pd.DataFrame({"kind": kind, "step": ty, "h": h, "spec": name, "date": d["date"].values,
                                              "ticker": d["ticker"].values, "y": d[tg].values,
                                              "p": np.clip(np.exp(yh), 1e-10, None),
                                              "pc": np.clip(np.exp(yh + s2 / 2), 1e-10, None)}))
        print("шаг", ty, "готов", flush=True)
    pd.concat(rows).to_parquet(OUT)
    print("сохранено", OUT)


if __name__ == "__main__":
    sys.exit(main())
