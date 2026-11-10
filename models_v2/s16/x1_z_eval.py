"""Этап Z, оценка (план Z раздел 5) — чистые функции от массивов (файлов нет). Реализации Y-A/V6/V7 — импортом (`x1_ya_eval`: mask, metrics_table, pair_row, go_table, _trim; Δ в % QLIKE ЯВНО НАЗВАННОЙ базы окна; блок 20 · 2000 · seed 0; DM лаг 5; MCS 90 %; правило 6.2; стресс).
Холм N = 749 (744 + 5) / N_wide = 3203 (3193 + 10) по окну, семейство = все пары окна. Окна — по году t⁺; «после включения» — d ≥ 2018-07-02. Ряды на ОДИНАКОВЫХ датах."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_ya_eval as EA

ST, EV, EVX = EA.ST, EA.EV, EA.EVX
N_HOLM, N_WIDE = 749, 3203
ZS = ("ZA", "ZB", "ZB0", "ZC", "ZCp")
ZBASES = ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "Y2", "Y3")
CONTRASTS = (("ZB", "ZB0"), ("ZC", "ZCp"), ("ZA", "ZB"), ("ZA", "ZC"), ("ZB", "ZC"))
PAIRS = [(y, b) for y in ZS for b in ZBASES] + list(CONTRASTS)
MCS_SERIES = ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1", "Y1", "Y2", "Y3") + ZS
STRESS_SERIES = ("K1", "K2", "K3", "V4:NTP2", "V7:cfg8", "X1", "Y2", "Y3") + ZS
YEARS, PWIN = EA.YEARS, EA.PWIN


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
        for w in PWIN:
            m = EA.mask(tp, d, "2018_2021" if w == "2018_2021_trim" else w)
            if w == "2018_2021_trim":
                m = EA._trim(m, Q.ql(rv, s[y]), Q.ql(rv, s[b]))
            rows.append(dict(y=y, base=b, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])))
    return holm(pd.DataFrame(rows), PWIN)


def rule62_table(rv, s: dict, tp, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in ZS:
        r = ST.new_information(rv, s[y], s["X1"], tp)
        h = pairs[(pairs["y"] == y) & (pairs["base"] == "X1") & (pairs["window"] == "2018_2021")].iloc[0]
        rows.append(dict(y=y, base="X1", window="2018_2021", delta_q=r["delta"], p_dm_one=r["p_dm_one"], years_neg=r["years_neg"], n_years=4, rule62="да" if r["new_information"] else "нет", p_holm_N=float(h["p_holm_N"])))
    return pd.DataFrame(rows)


def years_table(rv, s: dict, tp) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y in ZS:
        for b in ("X1", "K2"):
            for yr in YEARS:
                m = ST.wmask(tp, yr)
                qy, qb = float(Q.ql(rv[m], s[y][m]).mean()), float(Q.ql(rv[m], s[b][m]).mean())
                rows.append(dict(y=y, base=b, year=int(yr), n=int(m.sum()), y_qlike=qy, base_qlike=qb, delta_q=qy - qb, delta_pct=100 * (qy - qb) / qb))
    return pd.DataFrame(rows)


def cells_table(rv, s: dict, tp, e) -> pd.DataFrame:
    """Клетки остатка X1 (YA_CELLS): после экстремального бара (e_d = 1) 2022–2025; обычные дни (e_d = 0) 2019–2021; 2022 год. Описательно (p без Холма)."""
    rv, tp, e = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e)
    cells = {"после экстремального бара 2022–2025": ST.wmask(tp, "2022_2025") & (e == 1), "обычные дни 2019–2021": ST.wmask(tp, "2019_2021") & (e == 0), "2022 год": ST.wmask(tp, "2022")}
    pl = [(y, b) for y in ZS for b in ("X1", "K2")] + list(CONTRASTS)
    return pd.DataFrame([dict(cell=c, y=y, base=b, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])) for c, m in cells.items() for y, b in pl])


def mcs_table(rv, s: dict, tp) -> pd.DataFrame:
    return pd.concat([ST.mcs90(rv, {k: s[k] for k in MCS_SERIES}, tp, w).assign(window=w) for w in ("2018_2021", "2022_2025", "all")], ignore_index=True)


def go_all(rv, s: dict, tp) -> pd.DataFrame:
    rf = {k: s[k] for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "X1")}
    return pd.concat([EA.go_table(rv, s[y], rf, tp).assign(y=y) for y in ZS], ignore_index=True)


def params_table(zs: dict) -> pd.DataFrame:
    """Параметры по месяцам после включения (λ, ω, b, a, γ0…γ4, число закрытых экстремальных пар)."""
    cols = ["lam", "omega", "b", "a", "gam0", "gam1", "gam2", "gam3", "gam4", "n_ext"]
    rows = []
    for y, f in zs.items():
        a = f[f["active"] == 1]
        g = a.groupby(a["d"].dt.to_period("M"))[cols].first()
        rows += [dict(y=y, month=str(k), **{c: float(v[c]) for c in cols}) for k, v in g.iterrows()]
    return pd.DataFrame(rows)
