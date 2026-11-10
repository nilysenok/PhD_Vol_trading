"""Y-B, оценка Y3/Y4 — те же таблицы и окна, что `x1_ya_eval` (функции переиспользуются импортом: mask, metrics_table, pair_row, holm, go_table, _trim; Холм N = 744 / N_wide = 3193), плюс
Y3 против Y4, Y3/Y4 против Y1/Y2 (ряды Y1/Y2 — файлы листа Y-A), MCS 90 % с Y1–Y4 вместе и CONDITION (условие Y5/Y6 по RULE62 листов Y-A и Y-B). Чистые функции от массивов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_ya_eval as EA

ST, EV = EA.ST, EA.EV
YB, YA = ("Y3", "Y4"), ("Y1", "Y2")
PAIRS = [(y, b) for y in YB for b in EA.BASES] + [("Y3", "Y4")] + [(y, b) for y in YB for b in YA]


def pairs_table(rv, s: dict, tp, d) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y, b in PAIRS:
        for w in EA.PWIN:
            m = EA.mask(tp, d, "2018_2021" if w == "2018_2021_trim" else w)
            if w == "2018_2021_trim":
                m = EA._trim(m, Q.ql(rv, s[y]), Q.ql(rv, s[b]))
            rows.append(dict(y=y, base=b, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])))
    return EA.holm(pd.DataFrame(rows), EA.PWIN)


def rule62_table(rv, s: dict, tp, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in YB:
        r = ST.new_information(rv, s[y], s["X1"], tp)
        h = pairs[(pairs["y"] == y) & (pairs["base"] == "X1") & (pairs["window"] == "2018_2021")].iloc[0]
        rows.append(dict(y=y, base="X1", window="2018_2021", delta_q=r["delta"], p_dm_one=r["p_dm_one"], years_neg=r["years_neg"], n_years=4, rule62="да" if r["new_information"] else "нет", p_holm_N=float(h["p_holm_N"])))
    return pd.DataFrame(rows)


def years_table(rv, s: dict, tp) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y in YB:
        for b in ("X1", "K2"):
            for yr in EA.YEARS:
                m = ST.wmask(tp, yr)
                qy, qb = float(Q.ql(rv[m], s[y][m]).mean()), float(Q.ql(rv[m], s[b][m]).mean())
                rows.append(dict(y=y, base=b, year=int(yr), n=int(m.sum()), y_qlike=qy, base_qlike=qb, delta_q=qy - qb, delta_pct=100 * (qy - qb) / qb))
    return pd.DataFrame(rows)


def cells_table(rv, s: dict, tp, e) -> pd.DataFrame:
    """Клетки остатка X1: после экстремального бара (e_d = 1) 2022–2025; обычные дни (e_d = 0) 2019–2021; 2022 год."""
    rv, tp, e = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e)
    cells = {"после экстремального бара 2022–2025": ST.wmask(tp, "2022_2025") & (e == 1), "обычные дни 2019–2021": ST.wmask(tp, "2019_2021") & (e == 0), "2022 год": ST.wmask(tp, "2022")}
    pl = (("X1", "K2"),) + tuple((y, b) for y in YB for b in ("X1", "K2")) + (("Y3", "Y4"),)
    return pd.DataFrame([dict(cell=c, y=y, base=b, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])) for c, m in cells.items() for y, b in pl])


def params_table(ys: dict) -> pd.DataFrame:
    """Параметры по месяцам после включения (среднее по seeds для Y3)."""
    rows = []
    for y, f in ys.items():
        a = f[f["active"] == 1]
        g = a.groupby(a["d"].dt.to_period("M"))[["lam", "theta", "g1", "g2", "g3", "g4"]].first()
        rows += [dict(y=y, month=str(k), **{c: float(v[c]) for c in g.columns}) for k, v in g.iterrows()]
    return pd.DataFrame(rows)


def condition_table(rule_a: pd.DataFrame, rule_b: pd.DataFrame) -> pd.DataFrame:
    """Условие Y5/Y6: «правило 6.2 против X1 = да у Y1 или у Y3». rule_a — RULE62 листа Y-A, rule_b — листа Y-B. Последняя строка — итог."""
    a, b = rule_a.set_index("y")["rule62"], rule_b.set_index("y")["rule62"]
    v = {"Y1": str(a["Y1"]), "Y3": str(b["Y3"])}
    ok = "да" if "да" in v.values() else "нет"
    return pd.DataFrame([dict(item=k, rule62=x) for k, x in v.items()] + [dict(item="условие Y5/Y6", rule62=ok)])


def main_tables(rv, s: dict, tp, d, e, ys: dict, rule_a: pd.DataFrame) -> dict:
    """s — ряды (K1, K2, K3, M, K8, V4:NTP2, V6:A2, V7:cfg6, V7:cfg8, X1, Y1…Y4); ys — прогнозы Y3/Y4 (кадры); rule_a — YA_RULE62."""
    pr = pairs_table(rv, s, tp, d)
    r62 = rule62_table(rv, s, tp, pr)
    mc = pd.concat([ST.mcs90(rv, {k: s[k] for k in ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1") + YA + YB}, tp, w).assign(window=w) for w in ("2018_2021", "2022_2025", "all")], ignore_index=True)
    rf = {k: s[k] for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "X1")}
    return {"METRICS": EA.metrics_table(rv, s, tp, d), "PAIRS": pr, "RULE62": r62, "YEARS": years_table(rv, s, tp), "CONDITION": condition_table(rule_a, r62),
            "GO": pd.concat([EA.go_table(rv, s[y], rf, tp).assign(y=y) for y in YB], ignore_index=True), "MCS": mc,
            "STRESS": EV.stress_table(rv, {k: s[k] for k in ("K1", "K2", "K3", "V4:NTP2", "V7:cfg8", "X1") + YA + YB}, tp), "CELLS": cells_table(rv, s, tp, e), "PARAMS": params_table(ys)}
