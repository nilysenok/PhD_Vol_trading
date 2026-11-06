"""Ж-214, лист 3, стадия `sea2` (остаток V8; У23, принято дирижёром): Z3 `sa_roll60_input` (профиль — медиана долей за последние 60 дней) и Z7 `sa_roll500_input` (то же, 500 дней — сосед лучшей Z2);
вход RV_sa вместо RV в d, w, m, J прежний, цель — каноническая RV; change_type = input_measurement. Z8 не тратится. Профиль — `hv2_sea_in.median_profile` (дни строго < d, >= 60 дней и >= 20 наблюдений слота)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_a0, hv2_dec as DC, hv2_eval as E, hv2_ledger as L, hv2_m10 as M, hv2_paths as P, hv2_sea_in as SI, hv2_seas as Z, hv2_seastage as SS, hv2_world as W

DIR, SUB = "SEA2", "seasonality"
MODS = (SI, DC, Z, W, hv2_a0)
CTRL = ("A0", "XJ", "LF")
WIN = {"z3": 60, "z7": 500}
NAME = {"Z3": "z3", "Z7": "z7"}
KIND = {"Z3": "sa_roll60_input", "Z7": "sa_roll500_input"}
DESC = {"Z3": "RV_sa (профиль — медиана долей слота за последние 60 дней) вместо RV в d, w, m; J прежний; цель — RV", "Z7": "то же, профиль — последние 500 дней (сосед Z2)"}


def build_extra(M2: np.ndarray, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    for k, win in WIN.items():
        out[k] = Z.rv_sa(M2, SI.median_profile(M2, win))
    return out


def trunc_builder(M2: np.ndarray, index: pd.Index, i0: int, seed: int = 13) -> float:
    """Порча матрицы слотов после строки i0 не меняет Z3, Z7 на строках <= i0 (NaN-маски совпадают)."""
    rng = np.random.default_rng(seed)
    late = (np.arange(len(M2)) > i0)[:, None]
    M2c = np.where(late, M2 * np.exp(rng.normal(0, 1.0, M2.shape)), M2)
    a, b = build_extra(M2, index).iloc[:i0 + 1].to_numpy(), build_extra(M2c, index).iloc[:i0 + 1].to_numpy()
    return float("inf") if (np.isnan(a) != np.isnan(b)).any() else float(np.nanmax(np.abs(a - b)))


def stage_sea2(_a=None) -> int:
    ctx = E.context()
    m10 = pd.read_parquet(P.M10)
    r = M.bar_returns(pd.read_parquet(P.SBER))
    M2 = Z.slot_matrix(r, m10.index)
    w = SI.with_inputs(DC.build(ctx["w"], m10, ()), build_extra(M2, m10.index))
    a0 = ctx["refs"]["A0"]
    keys = {n: L.register("seasonality", {"name": n, "kind": KIND[n], "change_type": "input_measurement", "spec": DESC[n], "controls": ["A0 (каноническая RV)"]}) for n in KIND}
    tb = max(trunc_builder(M2, m10.index, m10.index.get_loc(pd.Timestamp(w.p.dates[int(w.el[k])]))) for k in E.TRUNC_AT)
    ser = {n: E.cached(DIR, n, MODS, lambda n=n: _compute(n, w, a0), ctx) for n in KIND}
    rows = []
    for n in KIND:
        tr = max(DC.trunc_diff(lambda x, n=n: SI.sigma(x, NAME[n]), w, int(w.el[k])) for k in E.TRUNC_AT)
        rows.append(E.row(ctx, n, ser[n][n], {"kind": KIND[n], "change_type": "input_measurement", "key": keys[n], "n_fallback": int(ser[n][n + "__nfb"][0]), "trunc_max_diff": tr, "trunc_builder_max_diff": tb}, CTRL))
    refs = [E.row(ctx, k, v, {"kind": "контроль", "change_type": "control"}, CTRL) for k, v in ctx["refs"].items() if k in CTRL]
    P.write_csv(pd.DataFrame(refs + rows), "SEASONALITY_V8_REST.csv", "V8 остаток (У23): Z3, Z7; change_type: input_measurement; Δ — строка минус контроль A0, минус — строка лучше; потолок: best по crit_wf на всей истории", SUB)
    E.decide("V8_rest", {"rows": {r["model"]: {"d_crit_A0": r["d_crit_A0"], "years_better_A0": r["years_better_A0"], "d_qlike_no_240222_A0": r["d_qlike_no_240222_A0"]} for r in rows},
                         "unused_trials": L.LIMITS["seasonality"] - 7, "not_run": ["Z8"], "Z6": "посчитана в sea (профиль Z2); не пересчитывается"})
    cols = ["model", "change_type", "qlike_all", "crit_wf", "d_crit_A0", "d_crit_XJ", "d_qlike_A0", "d_qlike_XJ", "d_qlike_no_240222_A0", "years_better_A0", "n_fallback", "trunc_max_diff", "trunc_builder_max_diff"]
    print(pd.DataFrame(rows)[cols].round(5).to_string(index=False), flush=True)
    return 0 if all(r["trunc_max_diff"] == 0 and r["trunc_builder_max_diff"] == 0 for r in rows) else 1


def _compute(n: str, w: DC.DWorld, a0) -> dict:
    return SS.compute(n, NAME[n], w, a0)
