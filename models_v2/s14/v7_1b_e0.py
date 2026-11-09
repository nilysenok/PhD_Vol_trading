"""V7-1b-A: E0 на данных SBER (план п. 5, шаг E0). Частный R² каждой из 4 компонент z на X_ref = (S, Σ, ln H, ℓ) по прошлому (`v7_1e_e0.novelty_gate`, гребень λ = 1, помесячная переоценка).
Объявлено до счёта: для года Y∈2018…2021 берётся z ЭНКОДЕРА ГОДА Y (одна система компонент, прошлое — z дат < Y того же энкодера), окно оценки — сам год Y; R² по году взвешивается числом оценённых дат.
Решение энкодера: R² компоненты усредняется по seeds 0–4; «новой информации нет» — все 4 усреднённых R² ≥ 0,9. Данные читаются только при вызове."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_io as IO, v7_1b_sta as STA, v7_1e_e0 as E0

V6 = IO.ROOT / "data/regen/harj_hybrid_v6"
E0_YEARS = (2018, 2019, 2020, 2021)
THRESH = E0.THRESH


def x_ref() -> pd.DataFrame:
    """Опорные признаки по дате d: S (s1…s9), Σ (22 столбца), ln H, ℓ = ln(M/H). Строки с нечисловыми значениями отбрасывает вызывающий."""
    from models_v2.s14 import v7_1m_data as DATA
    tab = DATA.load_tab()
    rd = lambda f: pd.read_parquet(V6 / f).assign(d=lambda t: pd.to_datetime(t["d"])).set_index("d").drop(columns=["d"], errors="ignore")
    S, SG = rd("S_features.parquet"), rd("SIGMA_features.parquet")
    M = np.sqrt(tab["K2"] * tab["K3"])
    out = pd.concat([S.add_prefix("S_"), SG.add_prefix("G_")], axis=1).reindex(tab.index)
    out["ln_H"], out["ell"] = np.log(tab["H"]), np.log(M / tab["H"])
    return out


def seed_r2(kind: str, seed: int, X: pd.DataFrame) -> dict:
    """-> dict(r2 (4,) взвешенное по годам, by_year {Y: r2}, n {Y: n_eval})."""
    by, n = {}, {}
    for y in E0_YEARS:
        a = STA.load(STA.directory() / STA.name(kind, y, seed))
        d = pd.DatetimeIndex(a["days"].astype("datetime64[ns]"))
        Xr = X.reindex(d)
        ok = np.isfinite(Xr.to_numpy(float)).all(axis=1)
        res = E0.novelty_gate(a["z"][ok], Xr.to_numpy(float)[ok], d[ok], start="%d-01-01" % y, end="%d-12-31" % y)
        by[y], n[y] = res["r2"], res["n_eval"]
    w = np.array([n[y] for y in E0_YEARS], float)
    return dict(r2=np.average(np.array([by[y] for y in E0_YEARS]), axis=0, weights=w), by_year=by, n=n)


def decide(r2_by_seed: list, thr: float = THRESH) -> dict:
    """r2_by_seed: список (4,) по seeds. -> dict(r2_mean (4,), no_new_info bool): все компоненты среднего R² ≥ thr."""
    m = np.mean(np.array(r2_by_seed, float), axis=0)
    return dict(r2_mean=m, no_new_info=bool(np.all(m >= thr)))


def run() -> dict:
    X, rows, out = x_ref(), [], {}
    for kind in STA.KINDS:
        per = [seed_r2(kind, s, X) for s in STA.SEEDS]
        dec = decide([p["r2"] for p in per])
        out[kind] = dict(r2_mean=[float(v) for v in dec["r2_mean"]], no_new_info=dec["no_new_info"], r2_seeds=[[float(v) for v in p["r2"]] for p in per],
                         r2_min_year_comp=float(min(min(p["by_year"][y]) for p in per for y in E0_YEARS)), n_eval={str(y): int(per[0]["n"][y]) for y in E0_YEARS})
        for s, p in zip(STA.SEEDS, per):
            rows += [dict(kind=kind, seed=s, component=c + 1, r2_weighted=p["r2"][c], **{"r2_%d" % y: p["by_year"][y][c] for y in E0_YEARS}) for c in range(4)]
    out["table"] = pd.DataFrame(rows)
    out["both_no_new_info"] = bool(all(out[k]["no_new_info"] for k in STA.KINDS))
    out["threshold"] = THRESH
    return out
