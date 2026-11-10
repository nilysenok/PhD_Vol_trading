"""Y-A, оценка Y1/Y2 (план раздел 4 и 8) — чистые функции от массивов (файлов нет). Реализации V6/V7 — импортом: Δ и интервал `v6_1_eval.delta_ci` (блок 20 · 2000 · seed 0), DM `dm_one` (лаг 5), Холм `holm_both`, MCS `v7_1m_stat.mcs90`,
правило 6.2 `v7_1m_stat.new_information`, стресс `v7_1b_ev.stress_table`. Δ — в % QLIKE ЯВНО НАЗВАННОЙ БАЗЫ окна (не уровня K1). Холм N = 744 / N_wide = 3193 (738 + 6 мест; 3181 + 6·2) по окну, семейство = все пары окна.
Окна — по году t⁺; «после включения» — d ≥ 2018-07-02. Ряды на ОДИНАКОВЫХ датах."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EVX

EV, ST = (importlib.import_module("models_v2.s14." + m) for m in ("v7_1b_ev", "v7_1m_stat"))
N_HOLM, N_WIDE = 744, 3193
AFTER = pd.Timestamp("2018-07-02")
YEARS = tuple(str(y) for y in range(2018, 2027))
MWIN = ("all", "2018_2021", "2019_2021", "2022_2025", "after") + YEARS          # 2026 — в YEARS
PWIN = ("all", "2018_2021", "2018_2021_trim", "2019_2021", "2022_2025", "2026", "after")
BASES = ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1")
GO_BASES7 = EV.GO_BASES + ("X1",)
YS = ("Y1", "Y2")


def mask(tp, d, w: str) -> np.ndarray:
    if w == "after":
        return pd.DatetimeIndex(d).to_numpy() >= AFTER.to_datetime64()
    return ST.wmask(tp, w)


def metrics_table(rv, s: dict, tp, d) -> pd.DataFrame:
    rows, rv, tp = [], np.asarray(rv, float), pd.DatetimeIndex(tp)
    for name, f in s.items():
        for w in MWIN:
            m = mask(tp, d, w)
            if m.any():
                rows.append(dict(series=name, window=w, first_tplus=str(tp[m].min().date()), last_tplus=str(tp[m].max().date()), **Q.metr_row(rv[m], np.asarray(f, float)[m])))
    return pd.DataFrame(rows)


def _trim(m, qy, qb) -> np.ndarray:
    keep, idx = m.copy(), np.flatnonzero(m)
    keep[idx[np.argsort(-np.abs(qy - qb)[idx], kind="stable")[: int(round(Q.TRIM * len(idx)))]]] = False
    return keep


def pair_row(rv, fy, fb, tp) -> dict:
    qy, qb = Q.ql(rv, fy), Q.ql(rv, fb)
    ay, ab = Q.annual(qy, tp, range(2018, 2027)), Q.annual(qb, tp, range(2018, 2027))
    return dict(n=len(rv), base_qlike=float(qb.mean()), y_qlike=float(qy.mean()), delta_q=float((qy - qb).mean()), **EVX.delta_ci(qy, qb, float(qb.mean())),
                p_one=EVX.dm_one(rv, fb, fy)["p_one"], years_neg=int(sum(ay[k] < ab[k] for k in ay)), n_years=len(ay))


def holm(t: pd.DataFrame, windows) -> pd.DataFrame:
    t = t.copy()
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in windows:
        k = (t["window"] == w) & t["p_one"].notna()
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    return t


def pairs_table(rv, s: dict, tp, d) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y, b in [(y, b) for y in YS for b in BASES] + [("Y1", "Y2")]:
        for w in PWIN:
            m = mask(tp, d, "2018_2021" if w == "2018_2021_trim" else w)
            if w == "2018_2021_trim":
                m = _trim(m, Q.ql(rv, s[y]), Q.ql(rv, s[b]))
            rows.append(dict(y=y, base=b, window=w, **pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])))
    return holm(pd.DataFrame(rows), PWIN)


def rule62_table(rv, s: dict, tp, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in YS:
        r = ST.new_information(rv, s[y], s["X1"], tp)
        h = pairs[(pairs["y"] == y) & (pairs["base"] == "X1") & (pairs["window"] == "2018_2021")].iloc[0]
        rows.append(dict(y=y, base="X1", window="2018_2021", delta_q=r["delta"], p_dm_one=r["p_dm_one"], years_neg=r["years_neg"], n_years=4, rule62="да" if r["new_information"] else "нет", p_holm_N=float(h["p_holm_N"])))
    return pd.DataFrame(rows)


def years_table(rv, s: dict, tp) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), pd.DatetimeIndex(tp), []
    for y in YS:
        for b in ("X1", "K2"):
            for yr in YEARS:
                m = ST.wmask(tp, yr)
                qy, qb = float(Q.ql(rv[m], s[y][m]).mean()), float(Q.ql(rv[m], s[b][m]).mean())
                rows.append(dict(y=y, base=b, year=int(yr), n=int(m.sum()), y_qlike=qy, base_qlike=qb, delta_q=qy - qb, delta_pct=100 * (qy - qb) / qb))
    return pd.DataFrame(rows)


def go_table(rv, f, rf: dict, tp, bases=GO_BASES7) -> pd.DataFrame:
    """Шесть условий «go» 6.1 (V7): условия 1–3 — к min(K2, K3, M, K8, NTP2, A2, X1); 4–6 — к лучшему из K2/K3 (bases=GO_BASES даёт ровно `v7_1b_ev.go_table`)."""
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    q, qb = Q.ql(rv, f), [Q.ql(rv, rf[b]) for b in bases]
    q2, q3 = Q.ql(rv, rf["K2"]), Q.ql(rv, rf["K3"])
    m1, m2, m3 = (ST.wmask(tp, w) for w in ("2018_2021", "2019_2021", "2022_2025"))
    keep, idx = m1.copy(), np.flatnonzero(m1)
    keep[idx[np.argsort(-np.abs(q - np.minimum(q2, q3))[idx], kind="stable")[: int(round(Q.TRIM * len(idx)))]]] = False
    gap = lambda m: float(q[m].mean() - min(b[m].mean() for b in qb))
    an = {y: (float(q[ST.wmask(tp, str(y))].mean()), min(float(q2[ST.wmask(tp, str(y))].mean()), float(q3[ST.wmask(tp, str(y))].mean())), int(ST.wmask(tp, str(y)).sum())) for y in range(2018, 2022)}
    better, gains = sum(a < b for a, b, _ in an.values()), [n * (b - a) for a, b, n in an.values()]
    total = float(sum(gains))
    share = max(gains) / total if total > 0 else float("inf")
    g6 = float(q[m3].mean() - min(q2[m3].mean(), q3[m3].mean()))
    nm = "min(" + ", ".join(bases) + ")"
    rows = [(1, "QLIKE 2018–2021 − " + nm, gap(m1), -Q.MARGIN, gap(m1) <= -Q.MARGIN), (2, "то же 2019–2021 (без 2018)", gap(m2), -Q.MARGIN, gap(m2) <= -Q.MARGIN),
            (3, "то же без 1 % дат окна с наибольшим |q − min(q_K2, q_K3)|", gap(keep), -Q.MARGIN, gap(keep) <= -Q.MARGIN), (4, "годов из 4 лучше лучшего из K2/K3", float(better), 3.0, better >= 3),
            (5, "доля максимального года в суммарном выигрыше над лучшим из K2/K3", share, 0.5, bool(total > 0 and share <= 0.5)), (6, "Δ к лучшему из K2/K3 на 2022–2025 (точка)", g6, 0.0, g6 <= 0)]
    t = pd.DataFrame(rows, columns=["cond", "name", "value", "threshold", "passed"])
    return pd.concat([t, pd.DataFrame([dict(cond=0, name="go (все шесть)", value=float(t["passed"].sum()), threshold=6.0, passed=bool(t["passed"].all()))])], ignore_index=True)


def cells_table(rv, s: dict, tp, e) -> pd.DataFrame:
    """Клетки остатка X1: после экстремального бара (e_d = 1) 2022–2025; обычные дни (e_d = 0) 2019–2021; 2022 год. Описательно (p без Холма)."""
    rv, tp, e = np.asarray(rv, float), pd.DatetimeIndex(tp), np.asarray(e)
    cells = {"после экстремального бара 2022–2025": ST.wmask(tp, "2022_2025") & (e == 1), "обычные дни 2019–2021": ST.wmask(tp, "2019_2021") & (e == 0), "2022 год": ST.wmask(tp, "2022")}
    rows = [dict(cell=c, y=y, base=b, **pair_row(rv[m], np.asarray(s[y], float)[m], np.asarray(s[b], float)[m], tp[m])) for c, m in cells.items() for y, b in (("X1", "K2"), ("Y1", "X1"), ("Y2", "X1"), ("Y1", "K2"), ("Y2", "K2"))]
    return pd.DataFrame(rows)


def params_table(ys: dict) -> pd.DataFrame:
    """Параметры по месяцам (a_n, a_e, κ, λ) после включения."""
    rows = []
    for y, f in ys.items():
        a = f[f["active"] == 1]
        g = a.groupby(a["d"].dt.to_period("M"))[["a_n", "a_e", "kappa", "lam"]].first()
        rows += [dict(y=y, month=str(k), **{c: float(v[c]) for c in g.columns}) for k, v in g.iterrows()]
    return pd.DataFrame(rows)


def main_tables(rv, s: dict, tp, d, e, ys: dict) -> dict:
    """s — ряды (K1, K2, K3, M, K8, V4:NTP2, V6:A2, V7:cfg6, V7:cfg8, X1, Y1, Y2); ys — прогнозы Y (кадры) для PARAMS."""
    pr = pairs_table(rv, s, tp, d)
    mc = pd.concat([ST.mcs90(rv, {k: s[k] for k in ("K1", "K2", "K3", "V4:NTP2", "V6:A2", "V7:cfg6", "V7:cfg8", "X1") + YS}, tp, w).assign(window=w) for w in ("2018_2021", "2022_2025", "all")], ignore_index=True)
    rf = {k: s[k] for k in ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "X1")}
    return {"METRICS": metrics_table(rv, s, tp, d), "PAIRS": pr, "RULE62": rule62_table(rv, s, tp, pr), "YEARS": years_table(rv, s, tp),
            "GO": pd.concat([go_table(rv, s[y], rf, tp).assign(y=y) for y in YS], ignore_index=True), "MCS": mc,
            "STRESS": EV.stress_table(rv, {k: s[k] for k in ("K1", "K2", "K3", "V4:NTP2", "V7:cfg8", "X1") + YS}, tp), "CELLS": cells_table(rv, s, tp, e), "PARAMS": params_table(ys)}
