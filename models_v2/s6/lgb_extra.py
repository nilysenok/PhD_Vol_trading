"""A12: специфическая диагностика LightGBM-аналога Б-2: листья, экстремальная RV, 5 seed, S1/S2 (leaf-wise). Всё — контроли (trials 0)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6 import engine6, lgb_b2, paths6

LGB = paths6.OUT / "lightgbm"
PER = {"all": lambda y: y > 0, "2017–2021": lambda y: y <= 2021, "2022": lambda y: y == 2022, "2023–2025": lambda y: (y >= 2023) & (y <= 2025), "2026": lambda y: y == 2026}


def fixed_dates(F, grid) -> list:
    """Первая сетка-дата каждого года (по дате d) и d цели 24.02.2022. ↺ 23.02.2022 — нерабочий день (нет в ряду), d = 22.02.2022."""
    yr = F.dates[grid].astype("datetime64[Y]").astype(int) + 1970
    first = [int(grid[np.flatnonzero(yr == y)[0]]) for y in sorted(set(yr))]
    return first + [int(np.flatnonzero(F.dates == np.datetime64("2022-02-22"))[0])]


def qlike_by_period(fc: pd.DataFrame) -> dict:
    q = np.asarray(metrics.qlike(fc["rv"], fc["sigma2"]))
    y = pd.to_datetime(fc["target_date"]).dt.year.to_numpy()
    return {k: float(q[f(y)].mean()) for k, f in PER.items()}


def seeds_table() -> pd.DataFrame:
    rows = []
    for s in lgb_b2.SEEDS:
        tag = "" if s == 42 else "_s%d" % s
        fc = pd.read_parquet(lgb_b2.fc_path(tag))
        st = json.loads((LGB / ("st_lgbm_b2%s.json" % tag)).read_text())
        rows.append({"seed": s, **{"qlike_" + k: v for k, v in qlike_by_period(fc).items()}, "trees_mean": st["trees"]})
    return pd.DataFrame(rows)


def variants_table() -> pd.DataFrame:
    rows = []
    for v in ("base", "S1", "S2"):
        fc = pd.read_parquet(lgb_b2.fc_path("" if v == "base" else "_" + v))
        st = json.loads((LGB / ("st_lgbm_b2%s.json" % ("" if v == "base" else "_" + v))).read_text())
        rows.append({"variant": v, "description": {"base": "baseline: max_depth=depth, num_leaves=2^depth", "S1": "max_depth=-1, num_leaves=2^depth", "S2": "max_depth=-1, num_leaves=31"}[v],
                     **{"qlike_" + k: x for k, x in qlike_by_period(fc).items()}, "trees_mean": st["trees"]})
    return pd.DataFrame(rows)


def summary(sd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [c for c in sd.columns if c.startswith("qlike_")] + ["trees_mean"]:
        rows.append({"kind": "seeds_42_46", "metric": c, "mean": sd[c].mean(), "std": sd[c].std(ddof=1), "min": sd[c].min(), "max": sd[c].max()})
    return pd.DataFrame(rows)


def extremes(p, w, L, dates) -> pd.DataFrame:
    base = engine6.run(p, w, L["i2"], dates, 42, "base", "none", leaves=True)
    rows = []
    for ym in ("clip99", "drop5"):
        m = engine6.run(p, w, L["i2"], dates, 42, "base", ym)
        for b, r in zip(base.itertuples(), m.itertuples()):
            rows.append({"kind": "extreme_" + ym, "d": b.d, "f_base": b.f, "f_mod": r.f, "ln_change": r.f - b.f, "sigma2_ln_change": np.log(r.sigma2 / b.sigma2)})
    return base, pd.DataFrame(rows)


def stage(a) -> int:
    paths6.mkdirs()
    L = paths6.world()
    w = lgb_b2.mk_world(L)
    dates = fixed_dates(L["F"], L["grid"])
    with engine6.pool(w) as p:
        base, ex = extremes(p, w, L, dates)
    lv = base[["d", "target_date", "k", "n_leaves", "leaf_min", "leaf_median", "share_lt_2min"]]
    sd, vt = seeds_table(), variants_table()
    paths6.write_csv(lv, paths6.REP / "lightgbm/LGB_LEAVES.csv", "листья финальной модели на фиксированных датах (count узлов LightGBM, bagging 0,8)")
    paths6.write_csv(sd, paths6.REP / "lightgbm/LGB_SEEDS.csv", "5 seed 42–46, контроль, не выбор")
    ext = pd.concat([ex, summary(sd), vt.assign(kind="leafwise_" + vt["variant"])], ignore_index=True)
    paths6.write_csv(ext, paths6.REP / "lightgbm/LGB_EXTRA.csv", "экстремальная RV (clip99, drop5), разброс seed, S1/S2")
    print(vt.to_string()); print(sd.to_string())
    return 0
