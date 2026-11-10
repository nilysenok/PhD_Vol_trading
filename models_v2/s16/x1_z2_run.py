"""Z2, ветка H1 / Hc (ТОЛЬКО при истинном условии L1–L3; запись и проверка). compute_h1 — чистая функция от входов (тестируется на синтетике); write_h1: реестр (guard) → прогнозы `data/regen/harj_x1/Z2/{h1,hc}.parquet` → `Z2` → `Z2.eval` (БЕЗ p) → таблицы Z2_*.csv с p → `Z2.rules`.
Окружение (X1, K1–K3, NTP2, Y2, Y3 — `x1_z_stage.load`; ZB, ZB0, ZC — сохранённые ряды Z по sha256) только читается."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_z2_eval as EZ, x1_z2_reg as ZR, x1_z2_walk as WK

TABLES = {"Z2_%s" % k: k for k in ("METRICS", "PAIRS", "YEARS", "STATES", "CRIT", "GO", "MCS", "STRESS", "PARAMS")}


def compute_h1(inp: dict, ctx1: dict, ctx2: dict, zser: dict) -> tuple:
    """inp: x1 (d, t_plus, rv, F, s), d, tp, rv, ser; ctx1 — события H1, ctx2 — Hc (перестановка); zser — ряды ZB, ZB0, ZC на общих датах. -> (прогнозы {H1, Hc}, таблицы, числа)."""
    x1, d, e = inp["x1"], inp["d"], inp["x1"]["s"].to_numpy(int)
    zs = {"H1": WK.walk(x1, ctx1, WK.stage1(ctx1, d), e), "Hc": WK.walk(x1, ctx2, WK.stage1(ctx2, d), e)}
    s = {**inp["ser"], **zser, **{k: f["F"].to_numpy(float) for k, f in zs.items()}}
    t = EZ.main_tables(inp["rv"], s, inp["tp"], d, e, zs)
    num = dict(n_dates=int(len(d)), n_inactive={k: int((f["active"] == 0).sum()) for k, f in zs.items()}, n_holm=EZ.N_HOLM, n_wide=EZ.N_WIDE, configs_spent=2, lams=[0.0, 0.001, 0.01, 0.1])
    return zs, t, num


def eval_numbers(t: dict) -> dict:
    """`Z2.eval`: QLIKE по окнам, Δ с интервалами, годы, состояния, K1–K5 — БЕЗ p-значений."""
    m, pr, st, cr, yr = t["METRICS"], t["PAIRS"], t["STATES"], t["CRIT"], t["YEARS"]
    return dict(qlike={s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")},
                pairs={"%s|%s|%s" % (r.y, r.base, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg)] for r in pr.itertuples()},
                states={"%s|%s|%s" % (r.y, r.state, r.window): [float(r.delta_pct), int(r.n)] for r in st.itertuples()}, crit={"%s|%s" % (r.y, r.crit): [r.name, float(r.value), bool(r.passed)] for r in cr.itertuples()},
                years={"%s|%s|%d" % (r.y, r.base, r.year): float(r.delta_pct) for r in yr.itertuples()}, p_values="не входят: записаны в Z2.rules после этого ключа")


def rules(t: dict) -> dict:
    pr = t["PAIRS"]
    return dict(p={"%s|%s|%s" % (r.y, r.base, r.window): dict(p_one=float(r.p_one), p_holm_N=float(r.p_holm_N), p_holm_Nwide=float(r.p_holm_Nwide)) for r in pr.itertuples()},
                rule62=bool(t["CRIT"].set_index("crit").at["6.2", "passed"]), confirmed=bool(t["CRIT"].set_index("crit").at["6.2", "passed"] and t["CRIT"].set_index("crit").at["Холм", "passed"]))


def check_register() -> None:
    for c in ZR.COUNTED:
        ZR.guard(c)
