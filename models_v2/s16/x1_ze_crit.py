"""Лист ZE, оценка и критерии (план ZE разд. 3.7–3.8; план Z разд. 5–6; записаны до счёта). Чистые функции от массивов/таблиц; реализации Y-A/V6/V7 — импортом (`x1_ya_eval`: mask, pair_row; блок 20 · 2000 · seed 0, DM лаг 5, MCS 90 %, правило 6.2, стресс).
Холм N = 751 (749 + 2) / N_wide = 3207 (3203 + 4) — не число строк (мутация № 103). Копия структуры `x1_z_eval`/`x1_z_crit` (правке не подлежат: в хэше кода Z) с рядами ZE, ZEp и базой ZB.
K1 — Δ к X1 за весь период < 0; K2 — Δ к X1 в состоянии N за весь период ≤ +0,10 %; K3 — Δ к X1 ≤ 0 в ≥ 5 из 8 лет 2018–2025; K4 — ZE < ZEp за весь период (Δ к ZEp < 0); K5 — Δ к X1 на 2018–2021 ≤ +0,10 %;
K6 — Δ(ZE − ZB) в состоянии E за 2023–2025 ≤ 0. Все шесть — «продолжить», иначе «закрыть» (решение разработки, не подтверждение; подтверждение — правило 6.2 против X1 и p Холма N < 0,05). ZEp критериев не получает."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s16 import x1_ya_eval as EA, x1_z_crit as ZC

ST, EVX, EV = EA.ST, EA.EVX, EA.EV
N_HOLM, N_WIDE = 751, 3207
ZS, HYP = ("ZE", "ZEp"), "ZE"
BASES = ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "Y2", "Y3", "ZB", "ZB0", "ZA")
PAIRS = [(y, b) for y in ZS for b in BASES] + [("ZE", "ZEp")]
MCS_SERIES = ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1", "Y1", "Y2", "Y3", "ZA", "ZB", "ZB0") + ZS
STRESS_SERIES = ("K1", "K2", "K3", "V4:NTP2", "V7:cfg8", "X1", "Y2", "Y3", "ZB") + ZS
DWIN = ("ex2022", "2023_2025")
SWIN = ZC.SWIN + DWIN
TOL_PCT, YEARS_MIN, CHECK_YEARS = ZC.TOL_PCT, ZC.YEARS_MIN, ZC.CHECK_YEARS


def wmask(tp, d, w: str) -> np.ndarray:
    """Окна по году t⁺; «без 2022» — все даты, кроме t⁺ в 2022 (в т. ч. 2026); «2023_2025» — 2023–2025."""
    y = pd.DatetimeIndex(tp).year.to_numpy()
    return (y != 2022) if w == "ex2022" else ((y >= 2023) & (y <= 2025)) if w == "2023_2025" else EA.mask(tp, d, w)


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
        for b in ("X1", "K2", "ZB"):
            for yr in EA.YEARS:
                m = ST.wmask(tp, yr)
                qy, qb = float(Q.ql(rv[m], s[y][m]).mean()), float(Q.ql(rv[m], s[b][m]).mean())
                rows.append(dict(y=y, base=b, year=int(yr), n=int(m.sum()), y_qlike=qy, base_qlike=qb, delta_q=qy - qb, delta_pct=100 * (qy - qb) / qb))
    return pd.DataFrame(rows)


def states_table(rv, s: dict, tp, d, e, zs: dict) -> pd.DataFrame:
    """E (e_d = 1) и N (e_d = 0) × окна × база (X1, ZB): QLIKE, Δ в % QLIKE базы состояния, средний g."""
    rv, tp, e, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e), []
    for y, f in zs.items():
        for st, ev in (("E", 1), ("N", 0)):
            for w in SWIN:
                m = wmask(tp, d, w) & (e == ev)
                for b in ("X1", "ZB"):
                    if m.sum() >= 2:
                        rows.append(dict(y=y, state=st, window=w, base=b, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m]), mean_g=float(f["g"].to_numpy(float)[m].mean())))
    return pd.DataFrame(rows)


def diag_table(rv, s: dict, tp, d) -> pd.DataFrame:
    """Диагностика (не критерий): окна «без 2022» и 2023–2025 для ZE, ZB, X1 (ZB против X1, ZE против X1 и ZB; ZEp против X1)."""
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y, b in (("ZE", "X1"), ("ZB", "X1"), ("ZE", "ZB"), ("ZEp", "X1")):
        for w in DWIN:
            m = wmask(tp, d, w)
            rows.append(dict(y=y, base=b, window=w, **EA.pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])))
    return pd.DataFrame(rows)


def mcs_table(rv, s: dict, tp) -> pd.DataFrame:
    return pd.concat([ST.mcs90(rv, {k: s[k] for k in MCS_SERIES}, tp, w).assign(window=w) for w in ("2018_2021", "2022_2025", "all")], ignore_index=True)


def crit_table(pairs: pd.DataFrame, states: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    """K1–K6 по ZE и итог «продолжить / закрыть»."""
    y, r = HYP, ZC._row
    p_all, p_dev, k4 = r(pairs, y=y, base="X1", window="all"), r(pairs, y=y, base="X1", window="2018_2021"), r(pairs, y=y, base="ZEp", window="all")
    sn, se = r(states, y=y, state="N", window="all", base="X1"), r(states, y=y, state="E", window="2023_2025", base="ZB")
    yr = years[(years["y"] == y) & (years["base"] == "X1") & years["year"].isin(CHECK_YEARS)]
    cnt = int((yr["delta_q"] <= 0).sum())
    items = [("K1", "весь период: Δ к X1 < 0, %", float(p_all["delta_pct"]), 0.0, bool(p_all["delta_pct"] < 0)),
             ("K2", "нет вреда обычным дням: Δ к X1 в состоянии N, весь период, %", float(sn["delta_pct"]), TOL_PCT, bool(sn["delta_pct"] <= TOL_PCT)),
             ("K3", "лет 2018–2025 с Δ к X1 ≤ 0 (из 8)", float(cnt), float(YEARS_MIN), cnt >= YEARS_MIN),
             ("K4", "лучше контроля ZEp за весь период: Δ к ZEp, %", float(k4["delta_pct"]), 0.0, bool(k4["delta_q"] < 0)),
             ("K5", "окно разработки 2018–2021: Δ к X1 ≤ +0,10 %, %", float(p_dev["delta_pct"]), TOL_PCT, bool(p_dev["delta_pct"] <= TOL_PCT)),
             ("K6", "не хуже ZB в состоянии E за 2023–2025: Δ к ZB ≤ 0, %", float(se["delta_pct"]), 0.0, bool(se["delta_q"] <= 0))]
    rows = [dict(y=y, crit=c, name=n, value=v, threshold=t, passed=ok, strict_years_neg=int((yr["delta_q"] < 0).sum()) if c == "K3" else np.nan) for c, n, v, t, ok in items]
    npass = sum(i[4] for i in items)
    return pd.DataFrame(rows + [dict(y=y, crit="итог", name="продолжить" if npass == 6 else "закрыть", value=float(npass), threshold=6.0, passed=npass == 6, strict_years_neg=np.nan)])


def params_table(zs: dict) -> pd.DataFrame:
    """Параметры по месяцам после включения (λ, b, a, c, число закрытых экстремальных пар)."""
    cols, rows = ["lam", "b", "a", "c", "n_ext"], []
    for y, f in zs.items():
        a = f[f["active"] == 1]
        g = a.groupby(a["d"].dt.to_period("M"))[cols].first()
        rows += [dict(y=y, month=str(k), **{c: float(v[c]) for c in cols}) for k, v in g.iterrows()]
    return pd.DataFrame(rows)


def main_tables(rv, s: dict, tp, d, e, zs: dict) -> dict:
    """s — ряды (K1, K2, K3, V4:NTP2, V6:A2, V7:cfg6, V7:cfg8, X1, Y1, Y2, Y3, ZA, ZB, ZB0, ZE, ZEp); zs — прогнозы ZE, ZEp (кадры)."""
    pr, yr, stt = pairs_table(rv, s, tp, d), years_table(rv, s, tp), states_table(rv, s, tp, d, e, zs)
    return {"METRICS": EA.metrics_table(rv, s, tp, d), "PAIRS": pr, "RULE62": rule62_table(rv, s, tp, pr), "YEARS": yr, "STATES": stt, "MCS": mcs_table(rv, s, tp), "DIAG": diag_table(rv, s, tp, d),
            "STRESS": EV.stress_table(rv, {k: s[k] for k in STRESS_SERIES}, tp), "CRIT": crit_table(pr, stt, yr), "PARAMS": params_table(zs)}
