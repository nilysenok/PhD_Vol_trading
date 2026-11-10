"""Z2, оценка H1 / Hc (план Z2 раздел 6; ТОЛЬКО при истинном условии) — чистые функции от массивов. Реализации Y-A/Z — импортом (`x1_ya_eval`: mask, metrics_table, pair_row, go_table; `x1_z_eval`: ничего не меняется).
Холм N = 751 (749 + 2) / N_wide = 3207 (3203 + 4) по окну; Δ — в % QLIKE ЯВНО НАЗВАННОЙ базы; блок 20 · 2000 · seed 0; DM лаг 5. K1–K5 — для H1 (контроль — Hc), записаны до счёта: K1 Δ к X1 < 0 (весь период); K2 состояние N: Δ к X1 ≤ +0,10 %;
K3 Δ к X1 ≤ 0 в ≥ 5 из 8 лет 2018–2025; K4 H1 < Hc (весь период); K5 2018–2021: Δ к X1 ≤ +0,10 %. Подтверждение — правило 6.2 против X1 и p Холма (N = 751) < 0,05."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_ya_eval as EA

ST, EVX = EA.ST, EA.EVX
N_HOLM, N_WIDE, HYP, TOL_PCT, YEARS_MIN = 751, 3207, ("H1", "Hc"), 0.10, 5
BASES = ("X1", "ZB", "ZB0", "Y2", "Y3", "K2", "K3", "V4:NTP2")
CONTRASTS = (("H1", "Hc"), ("H1", "ZB"), ("H1", "ZC"))
PAIRS = [(y, b) for y in HYP for b in BASES] + list(CONTRASTS)
SERIES = ("K1", "K2", "K3", "V4:NTP2", "V7:cfg8", "X1", "Y2", "Y3", "ZB", "ZB0", "ZC", "H1", "Hc")
CHECK_YEARS = tuple(range(2018, 2026))


def holm(t: pd.DataFrame, windows) -> pd.DataFrame:
    t = t.copy()
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in windows:
        k = (t["window"] == w) & t["p_one"].notna()
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    return t


def pairs_table(rv, s: dict, tp, d) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y, b in PAIRS:
        for w in EA.PWIN:
            m = EA.mask(tp, d, "2018_2021" if w == "2018_2021_trim" else w)
            if w == "2018_2021_trim":
                m = EA._trim(m, Q.ql(rv, s[y]), Q.ql(rv, s[b]))
            rows.append(dict(y=y, base=b, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])))
    return holm(pd.DataFrame(rows), EA.PWIN)


def years_table(rv, s: dict, tp) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y in HYP:
        for b in ("X1", "ZB"):
            for yr in EA.YEARS:
                m = ST.wmask(tp, yr)
                qy, qb = float(Q.ql(rv[m], s[y][m]).mean()), float(Q.ql(rv[m], s[b][m]).mean())
                rows.append(dict(y=y, base=b, year=int(yr), n=int(m.sum()), y_qlike=qy, base_qlike=qb, delta_q=qy - qb, delta_pct=100 * (qy - qb) / qb))
    return pd.DataFrame(rows)


def states_table(rv, s: dict, tp, d, e) -> pd.DataFrame:
    """E (e_d = 1) и N (e_d = 0) × окна: QLIKE ряда и X1, Δ к X1 (в % QLIKE X1 состояния), по годам и целиком."""
    rv, tp, e, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e), []
    for y in HYP + ("ZB",):
        for st, ev in (("E", 1), ("N", 0)):
            for w in ("all", "2018_2021", "2022_2025") + EA.YEARS:
                m = EA.mask(tp, d, w) & (e == ev)
                if m.sum() >= 2:
                    rows.append(dict(y=y, state=st, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s["X1"], float)[m], tp[m])))
    return pd.DataFrame(rows)


def _row(t: pd.DataFrame, **kw) -> pd.Series:
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        m &= (t[k] == v).to_numpy()
    return t[m].iloc[0]


def crit_table(pairs: pd.DataFrame, states: pd.DataFrame, years: pd.DataFrame, rule62: dict) -> pd.DataFrame:
    """K1–K5 для H1 (контроль Hc; Hc критериев не получает) + правило 6.2 и p Холма N = 751 (отдельно от «продолжить»)."""
    p_all, p_dev = _row(pairs, y="H1", base="X1", window="all"), _row(pairs, y="H1", base="X1", window="2018_2021")
    sn, k4 = _row(states, y="H1", state="N", window="all"), _row(pairs, y="H1", base="Hc", window="all")
    yr = years[(years["y"] == "H1") & (years["base"] == "X1") & years["year"].isin(CHECK_YEARS)]
    cnt = int((yr["delta_q"] <= 0).sum())
    items = [("K1", "весь период: Δ к X1 < 0, %", float(p_all["delta_pct"]), 0.0, bool(p_all["delta_pct"] < 0)), ("K2", "состояние N: Δ к X1, весь период, %", float(sn["delta_pct"]), TOL_PCT, bool(sn["delta_pct"] <= TOL_PCT)),
             ("K3", "лет 2018–2025 с Δ к X1 ≤ 0 (из 8)", float(cnt), float(YEARS_MIN), cnt >= YEARS_MIN), ("K4", "лучше контроля Hc: Δ к Hc, %", float(k4["delta_pct"]), 0.0, bool(k4["delta_q"] < 0)),
             ("K5", "2018–2021: Δ к X1 ≤ +0,10 %, %", float(p_dev["delta_pct"]), TOL_PCT, bool(p_dev["delta_pct"] <= TOL_PCT))]
    rows = [dict(y="H1", crit=c, name=n, value=v, threshold=t, passed=ok) for c, n, v, t, ok in items]
    npass = sum(r["passed"] for r in rows)
    rows.append(dict(y="H1", crit="итог", name="продолжить" if npass == 5 else "закрыть", value=float(npass), threshold=5.0, passed=npass == 5))
    rows.append(dict(y="H1", crit="6.2", name="правило 6.2 против X1 (2018–2021)", value=float(rule62["delta"]), threshold=0.05, passed=bool(rule62["new_information"])))
    rows.append(dict(y="H1", crit="Холм", name="p Холма N = %d < 0,05" % N_HOLM, value=float(p_dev["p_holm_N"]), threshold=0.05, passed=bool(p_dev["p_holm_N"] < 0.05)))
    return pd.DataFrame(rows)


def main_tables(rv, s: dict, tp, d, e, zs: dict) -> dict:
    """s — ряды SERIES; zs — прогнозы {H1, Hc} (кадры)."""
    pr, yr, stt = pairs_table(rv, s, tp, d), years_table(rv, s, tp), states_table(rv, s, tp, d, e)
    r62 = ST.new_information(rv, s["H1"], s["X1"], tp)
    rf = {k: s[k] for k in ("K1", "K2", "K3", "V4:NTP2", "X1")}
    mcs = pd.concat([ST.mcs90(rv, {k: s[k] for k in SERIES}, tp, w).assign(window=w) for w in ("2018_2021", "2022_2025", "all")], ignore_index=True)
    cols = ["lam", "a", "mu", "n", "beta", "eta", "rho", "nev", "n_ext"]
    par = pd.concat([f[f["active"] == 1].groupby(f["d"].dt.to_period("M"))[cols].first().reset_index().assign(y=y).astype({"d": str}).rename(columns={"d": "month"}) for y, f in zs.items()], ignore_index=True)
    return {"METRICS": EA.metrics_table(rv, s, tp, d), "PAIRS": pr, "YEARS": yr, "STATES": stt, "CRIT": crit_table(pr, stt, yr, r62), "GO": pd.concat([EA.go_table(rv, s[y], rf, tp).assign(y=y) for y in HYP], ignore_index=True),
            "MCS": mcs, "STRESS": EA.EV.stress_table(rv, {k: s[k] for k in SERIES}, tp), "PARAMS": par}
