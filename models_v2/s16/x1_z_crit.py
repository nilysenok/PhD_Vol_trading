"""Этап Z, состояния E/N и критерии K1–K5 (план Z раздел 5–6; записаны до счёта, после просмотра не меняются). Чистые функции от таблиц/массивов.
K1 — весь период: Δ к X1 < 0; K2 — нет вреда обычным дням: Δ к X1 в состоянии N за весь период ≤ +0,10 %; K3 — Δ к X1 ≤ 0 не менее чем в 5 из 8 полных лет 2018–2025;
K4 — лучше своего контроля за весь период (ZB < ZB0; ZC < ZCp; ZA — против ZB); K5 — Δ к X1 на 2018–2021 ≤ +0,10 %. Решение: все пять — «продолжить», иначе «закрыть»; это решение разработки, не подтверждение
(подтверждение — только правило 6.2 против X1 и p Холма N = 749 < 0,05, см. `x1_z_doc`). ZB0 и ZCp — контроли, критериев не получают."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_ya_eval as EA

HYP = ("ZA", "ZB", "ZC")
CTRL = {"ZA": "ZB", "ZB": "ZB0", "ZC": "ZCp"}
TOL_PCT, YEARS_MIN, CHECK_YEARS = 0.10, 5, tuple(range(2018, 2026))
SWIN = ("all", "2018_2021", "2022_2025", "2026", "after") + tuple(str(y) for y in range(2018, 2026))


def states_table(rv, s: dict, tp, d, e, zs: dict) -> pd.DataFrame:
    """E (e_d = 1) и N (e_d = 0) × окна: QLIKE ряда и X1, Δ к X1 (в % QLIKE X1 состояния), средние g и затвора (по прогнозам `zs`, для ZC/ZCp — σ(γ0 + γᵀz̃))."""
    rv, tp, e, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e), []
    for y, f in zs.items():
        for st, ev in (("E", 1), ("N", 0)):
            for w in SWIN:
                m = EA.mask(tp, d, w) & (e == ev)
                if m.sum() < 2:
                    continue
                gate = f["gate"].to_numpy(float)[m]
                rows.append(dict(y=y, state=st, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s["X1"], float)[m], tp[m]),
                                 mean_g=float(f["g"].to_numpy(float)[m].mean()), mean_gate=float(np.nanmean(gate)) if np.isfinite(gate).any() else float("nan")))
    return pd.DataFrame(rows)


def main_tables(rv, s: dict, tp, d, e, zs: dict) -> dict:
    """s — ряды (K1, K2, K3, M, K8, V4:NTP2, V6:A2, V7:cfg6, V7:cfg8, X1, Y1, Y2, Y3, ZA…ZCp); zs — прогнозы Z (кадры)."""
    from models_v2.s16 import x1_z_eval as EZ
    pr, yr, stt = EZ.pairs_table(rv, s, tp, d), EZ.years_table(rv, s, tp), states_table(rv, s, tp, d, e, zs)
    return {"METRICS": EA.metrics_table(rv, s, tp, d), "PAIRS": pr, "RULE62": EZ.rule62_table(rv, s, tp, pr), "YEARS": yr, "STATES": stt, "CELLS": EZ.cells_table(rv, s, tp, e), "GO": EZ.go_all(rv, s, tp),
            "MCS": EZ.mcs_table(rv, s, tp), "STRESS": EA.EV.stress_table(rv, {k: s[k] for k in EZ.STRESS_SERIES}, tp), "CRIT": crit_table(pr, stt, yr), "PARAMS": EZ.params_table(zs)}


def _row(t: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        m &= (t[k] == v).to_numpy()
    return t[m].iloc[0]


def crit_table(pairs: pd.DataFrame, states: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    """K1–K5 по ZA, ZB, ZC и итог «продолжить / закрыть»."""
    rows = []
    for y in HYP:
        p_all, p_dev = _row(pairs, y=y, base="X1", window="all"), _row(pairs, y=y, base="X1", window="2018_2021")
        sn, k4 = _row(states, y=y, state="N", window="all"), _row(pairs, y=y, base=CTRL[y], window="all")
        yr = years[(years["y"] == y) & (years["base"] == "X1") & years["year"].isin(CHECK_YEARS)]
        cnt = int((yr["delta_q"] <= 0).sum())
        items = [("K1", "весь период: Δ к X1 < 0, %", float(p_all["delta_pct"]), 0.0, bool(p_all["delta_pct"] < 0)),
                 ("K2", "нет вреда обычным дням: Δ к X1 в состоянии N, весь период, %", float(sn["delta_pct"]), TOL_PCT, bool(sn["delta_pct"] <= TOL_PCT)),
                 ("K3", "лет 2018–2025 с Δ к X1 ≤ 0 (из 8)", float(cnt), float(YEARS_MIN), cnt >= YEARS_MIN),
                 ("K4", "лучше контроля %s за весь период: Δ к %s, %%" % (CTRL[y], CTRL[y]), float(k4["delta_pct"]), 0.0, bool(k4["delta_q"] < 0)),
                 ("K5", "окно разработки 2018–2021: Δ к X1 ≤ +0,10 %, %", float(p_dev["delta_pct"]), TOL_PCT, bool(p_dev["delta_pct"] <= TOL_PCT))]
        rows += [dict(y=y, crit=c, name=n, value=v, threshold=t, passed=ok, strict_years_neg=int((yr["delta_q"] < 0).sum()) if c == "K3" else np.nan) for c, n, v, t, ok in items]
        npass = sum(r[4] for r in items)
        rows.append(dict(y=y, crit="итог", name="продолжить" if npass == 5 else "закрыть", value=float(npass), threshold=5.0, passed=npass == 5, strict_years_neg=np.nan))
    return pd.DataFrame(rows)
