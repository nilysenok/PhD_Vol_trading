"""Ж-214, лист 3, стадия `sea` (V8, п. 15 блока): Z1 `sa_exp_input`, Z2 `sa_roll250_input`, Z4 `sub20_input`, Z5 `trim_first_input` — 4 trials из 8 → V13 к A0 на канонической RV.
Колонка `change_type` разделяет: input_measurement (меняется вход, цель — каноническая RV) / forecast / target_definition. Z6 (другая цель) — только если V13 пройден, в отдельной таблице
`Z6_TARGET_VERSION.csv`, сравнение на исходной RV, в основную таблицу и в V13 не входит. Z3, Z7, Z8 не считаются (У20)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_a0, hv2_dec as DC, hv2_eval as E, hv2_ledger as L, hv2_m10 as M, hv2_paths as P, hv2_sea_in as SI, hv2_seas as Z, hv2_world as W

DIR, SUB = "SEA", "seasonality"
MODS = (SI, DC, Z, W, hv2_a0)
CTRL = ("A0", "XJ", "LF")
DESC = {"Z1": "RV_sa (профиль — расширяющаяся медиана долей слота по дням < d) вместо RV в d, w, m; J прежний; цель — RV", "Z2": "то же, профиль — медиана долей за последние 250 дней",
        "Z4": "RV на 20-минутной сетке (среднее двух сдвигов) вместо RV в d, w, m", "Z5": "RV без первого бара (слот 0) вместо RV в d, w, m"}


def compute(n: str, k: str, w: DC.DWorld, a0) -> dict:
    s2, nfb = SI.run_input(w, k, a0)
    return {n: s2, n + "__nfb": np.full(len(s2), float(nfb))}


def trunc_builder(M2: np.ndarray, R: np.ndarray, index: pd.Index, i0: int, seed: int = 11) -> float:
    """Главный тест обрезки построителя входов: порча матриц слотов после строки i0 не меняет ряды Z1…Z5 на строках <= i0 (NaN-маски совпадают); max |разности| (0 = утечки нет)."""
    rng = np.random.default_rng(seed)
    late = (np.arange(len(M2)) > i0)[:, None]
    M2c = np.where(late, M2 * np.exp(rng.normal(0, 1.0, M2.shape)), M2)
    Rc = np.where(late, R * rng.normal(1, 1.0, R.shape), R)
    a, b = SI.build_inputs(M2, R, index).iloc[:i0 + 1].to_numpy(), SI.build_inputs(M2c, Rc, index).iloc[:i0 + 1].to_numpy()
    return float("inf") if (np.isnan(a) != np.isnan(b)).any() else float(np.nanmax(np.abs(a - b)))


def row_z(ctx: dict, w: DC.DWorld, n: str, s2, nfb: int, key: str, tb: float, change: str = "input_measurement") -> dict:
    tr = max(DC.trunc_diff(lambda x, n=n: SI.sigma(x, SI.KEYS[n]), w, int(w.el[k])) for k in E.TRUNC_AT)
    return E.row(ctx, n, s2, {"kind": SI.KIND[n], "change_type": change, "key": key, "n_fallback": nfb, "trunc_max_diff": tr, "trunc_builder_max_diff": tb}, CTRL)


def stage_sea(_a=None) -> int:
    ctx = E.context()
    m10 = pd.read_parquet(P.M10)
    r = M.bar_returns(pd.read_parquet(P.SBER))
    M2, R = Z.slot_matrix(r, m10.index), Z.slot_matrix(r, m10.index, power=1)
    w = SI.with_inputs(DC.build(ctx["w"], m10, ()), SI.build_inputs(M2, R, m10.index))
    a0 = ctx["refs"]["A0"]
    keys = {n: L.register("seasonality", {"name": n, "kind": SI.KIND[n], "change_type": "input_measurement", "spec": DESC[n], "controls": ["A0 (каноническая RV)"]}) for n in SI.KEYS}
    tb = max(trunc_builder(M2, R, m10.index, m10.index.get_loc(pd.Timestamp(w.p.dates[int(w.el[k])]))) for k in E.TRUNC_AT)
    ser = {n: E.cached(DIR, n, MODS, lambda n=n, k=k: compute(n, k, w, a0), ctx) for n, k in SI.KEYS.items()}
    rows = [row_z(ctx, w, n, ser[n][n], int(ser[n][n + "__nfb"][0]), keys[n], tb) for n in SI.KEYS]
    refs = [E.row(ctx, k, v, {"kind": "контроль", "change_type": "control"}, CTRL) for k, v in ctx["refs"].items() if k in CTRL]
    P.write_csv(pd.DataFrame(refs + rows), "SEASONALITY_V8.csv", "V8; change_type: input_measurement (цель — каноническая RV); Δ — строка минус контроль A0, минус — строка лучше; потолок: best по crit_wf на всей истории", SUB)
    dec = E.v13(rows, ("A0",), len(rows), L.LIMITS["seasonality"])
    out = {**dec, "minimal_set": list(SI.KEYS), "unused_trials": L.LIMITS["seasonality"] - len(rows), "Z6": "не запускалось: V13 V8 не пройден", "not_run": ["Z3", "Z7", "Z8"]}
    if dec["continue"]:
        out["Z6"] = run_z6(ctx, w, dec["best"], a0)
    E.decide("V8", out)
    cols = ["model", "change_type", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_XJ", "d_qlike_no_240222_A0", "years_better_A0", "n_fallback", "trunc_max_diff", "trunc_builder_max_diff"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False))
    print("V13:", dec["best"], dec["continue"], dec["controls"], "| Z6:", out["Z6"], flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 and r["trunc_builder_max_diff"] == 0 for r in rows) else 1


def run_z6(ctx: dict, w: DC.DWorld, best: str, a0) -> str:
    """Z6: цель RV_sa профиля лучшего из Z1/Z2 (если лучший Z4/Z5 — Z1); таблица отдельно, change_type = target_definition."""
    n = best if best in ("Z1", "Z2") else "Z1"
    key = L.register("seasonality", {"name": "Z6", "kind": "sa_target_version", "change_type": "target_definition", "profile": n, "spec": "A0-спецификация по RV_sa, шкала c_d по закрытым, сравнение на исходной RV",
                                     "controls": ["A0"]})
    s2, sa, nfb = SI.run_target(w, SI.KEYS[n], a0)
    r = E.row(ctx, "Z6", s2, {"kind": "sa_target_version", "change_type": "target_definition", "profile": n, "key": key, "n_fallback": nfb}, CTRL)
    P.write_csv(pd.DataFrame([r]), "Z6_TARGET_VERSION.csv", "Z6: ДРУГАЯ ЦЕЛЬ (RV_sa); оценка на исходной RV через шкалу c_d по закрытым; в основную таблицу и V13 не входит", SUB)
    return json.dumps({"profile": n, "crit_wf": r["crit_wf"], "d_crit_A0": r["d_crit_A0"]})
