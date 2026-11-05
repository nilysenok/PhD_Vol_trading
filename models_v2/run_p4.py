"""Этап 4 (Ж-197 п. 4, П4–П5): итоговая таблица всех строк × срезы и DM к HAR-J(б) (Ньюи–Уэст, лаг 5, Холм).

Запуск: python3 models_v2/run_p4.py   (после run_p1, run_p2, run_pb, run_p3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models_v2 import metrics, prov  # noqa: E402
from models_v2.s3 import dm, prov3  # noqa: E402

OUT = prov.ROOT / "data/regen/v2_sber_h1"
TAB = prov.ROOT / "docs/tables/v2_sber_h1"
BASE = "harj_b"
SAMPLES = (("all", lambda y: y > 0), ("<=2021", lambda y: y <= 2021), ("2022+", lambda y: y >= 2022))


def family(name: str) -> str:
    if name in ("rw", "rv_w", "rv_m"):
        return "etalon"
    if name.startswith(("har_", "harj_")):
        return "har"
    if name.startswith("b7_"):
        return "B7"
    if name.startswith("b") and name[1].isdigit():
        return "B" + name[1]
    return "boost" if name.startswith(("xgb_", "lgbm_")) else name[:2]  # g1…g5


def load() -> pd.DataFrame:
    p1, p2, pb_, p3 = (pd.read_parquet(OUT / ("forecasts_%s.parquet" % s)) for s in ("p1", "p2", "pb", "p3"))
    p2 = p2[~p2.model.str.endswith("_btr")]
    fc = pd.concat([p1, p2, pb_, p3], ignore_index=True)
    for c in ("d", "target_date"):
        fc[c] = fc[c].astype("datetime64[ns]")
    n = fc.groupby("model").size()
    assert n.nunique() == 1 and n.iloc[0] == 2272, n.value_counts()
    assert (fc["sigma2"] > 0).all() and np.isfinite(fc["sigma2"]).all()
    return fc.sort_values(["model", "d"]).reset_index(drop=True)


def extra(fc: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    rows = []
    for model, f in fc.groupby("model", sort=False):
        st = f["d"].map(stress).to_numpy(bool)
        masks = {"no2022": (f["target_date"].dt.year != 2022).to_numpy(), "stress_g4": st, "calm_g4": ~st}
        for sl, m in masks.items():
            rows.append({"model": model, "slice": sl, **metrics.summary(f["rv"][m], f["sigma2"][m]), "note": ""})
    return pd.DataFrame(rows)


def dm_set(fc: pd.DataFrame, mt: pd.DataFrame) -> dict:
    """5 лучших по QLIKE(all) (без harj_b, без дублей рядов) + лучшая каждого класса. -> {модель: причина}."""
    q = mt[mt.slice == "all"].set_index("model")["qlike"].drop(BASE).sort_values()
    seen, top = [], []
    for n in q.index:
        s = fc[fc.model == n]["sigma2"].to_numpy()
        if not any(np.array_equal(s, t) for t in seen):
            seen.append(s)
            top.append(n)
    out = {n: "топ-5" for n in top[:5]}
    fam = pd.Series({n: family(n) for n in q.index})
    sub = fam[~fam.isin(["etalon", "har"])]
    for f, grp in sub.groupby(sub):
        best = q[grp.index].idxmin()
        out[best] = (out[best] + "; " if best in out else "") + "класс " + f
    return out


def dm_table(fc: pd.DataFrame, names: dict) -> pd.DataFrame:
    base = fc[fc.model == BASE].reset_index(drop=True)
    rows = []
    for samp, mk in SAMPLES:
        m = mk(base["target_date"].dt.year).to_numpy()
        part = []
        for n, why in names.items():
            f = fc[fc.model == n].reset_index(drop=True)
            assert (f["d"].values == base["d"].values).all()
            part.append({"model": n, "why": why, "sample": samp, **dm.dm(f["rv"][m], base["sigma2"][m], f["sigma2"][m])})
        t = pd.DataFrame(part)
        t["p_holm"] = dm.holm(t["p"].to_numpy())
        rows.append(t)
    return pd.concat(rows)


def main():
    fc = load()
    rvi = pd.read_parquet(prov.RVI)
    rvi = rvi[rvi["date"] <= pd.Timestamp("2021-12-31")]  # RVI >= 2022 не используется
    stress = pd.read_parquet(OUT / "stress_p3.parquet").set_index("d")["stress"]
    mt = pd.concat([metrics.metrics_table(fc, rvi), extra(fc, stress)], ignore_index=True)
    mt.insert(1, "family", mt["model"].map(family))
    prov3.write_csv(mt, TAB / "p4_all.csv", "все строки (этапы 1, 2, Б, 3) × срезы; stress_g4/calm_g4 по индикатору Г4 на день d; no2022 по дате цели")
    names = dm_set(fc, mt)
    prov3.write_csv(dm_table(fc, names), TAB / "p4_dm.csv",
                    "DM к harj_b: d = QLIKE(harj_b) − QLIKE(модель), плюс = модель лучше; NW, Бартлетт, лаг 5; Холм внутри выборки; "
                    "набор выбран по QLIKE(all) на оценочном периоде")
    print("моделей", fc["model"].nunique(), "срезов", mt["slice"].nunique(), "DM:", list(names))


if __name__ == "__main__":
    main()
