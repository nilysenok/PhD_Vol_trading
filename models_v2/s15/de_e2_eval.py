"""E2: оценка X2 — чистые функции от массивов. Метрики, Δ, DM, «go» 6.1, стресс, MCS, пары 6.2 — функции V7 (импорт через `de_e1_eval`), Холм N = 738 / N_wide = 3181 (как E1).
Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, NTP2, A2, A0, A0g, cfg8, cfg4, X1 и X2. Пары 6.2 X2: против A0g и cfg4 (абляции плана, G12) + K2, K3 для справки.
Калибровка p: по децилям p (граница — квантили p окна) средний p против доли дней со стрессом (метка — порог закрытых пар переоценки, `thr`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de_e1_eval as E1V

EV, ST, E2, Q = E1V.EV, E1V.ST, E1V.E2, E1V.Q
SERIES = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g", "cfg8", "cfg4", "X1", "X2")
PAIRS = (("X2 против A0g (6.2)", "X2", "V6:A0g"), ("X2 против cfg4 (6.2)", "X2", "cfg4"), ("X2 против K2", "X2", "K2"), ("X2 против K3", "X2", "K3"), ("X2 против X1", "X2", "X1"))
STATE_SERIES = ("X2", "X1", "V4:NTP2", "cfg8", "cfg4", "V6:A0g", "K2", "K3")
CAL_WIN = ("2018_2021", "2022_2025", "all")
NDEC = 10


def main_tables(rv, s: dict, tp) -> dict:
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    refs = {k: s[k] for k in EV.REFS}
    return {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "TESTS": E1V._holm(EV.tests_table(rv, s["X2"], refs, tp), EV.TEST_WIN).assign(config="X2"),
            "GO": EV.go_table(rv, s["X2"], refs, tp).assign(config="X2"), "YEARS": EV.years_table(rv, s["X2"], refs, tp).assign(config="X2"),
            "STRESS": EV.stress_table(rv, {k: s[k] for k in E2.CTRL + ("cfg8", "cfg4", "X1", "X2")}, tp), "PAIRS": E1V._holm(E2.pairs_table(rv, s, tp, PAIRS), E2.PAIR_WIN)}


def state_qlike(rv, s: dict, tp, state) -> pd.DataFrame:
    rv, tp, state, rows = np.asarray(rv, float), np.asarray(tp), np.asarray(state), []
    for name in STATE_SERIES:
        q = Q.ql(rv, np.asarray(s[name], float))
        for w in E1V.STATE_WIN:
            for k, lab in ((0, "норма"), (1, "экстрем")):
                m = ST.wmask(tp, w) & (state == k)
                rows.append(dict(series=name, window=w, state=lab, n=int(m.sum()), qlike=float(q[m].mean()) if m.any() else float("nan")))
    return pd.DataFrame(rows)


def calibration(tp, p, rv, thr) -> pd.DataFrame:
    """Строки (окно, дециль p): n, средний p, доля стресса, |разность|; дециль «итого» — по окну целиком (ece = Σ n_k/n·|p̄_k − доля_k|)."""
    tp, p, y, rows = np.asarray(tp), np.asarray(p, float), (np.asarray(rv, float) > np.asarray(thr, float)).astype(float), []
    for w in CAL_WIN:
        m = ST.wmask(tp, w) & np.isfinite(p) & np.isfinite(thr)
        pw, yw = p[m], y[m]
        edges = np.quantile(pw, np.linspace(0, 1, NDEC + 1))
        k = np.clip(np.searchsorted(edges[1:-1], pw, side="right"), 0, NDEC - 1)
        ece = 0.0
        for j in range(NDEC):
            s = k == j
            if s.any():
                ece += s.sum() / len(pw) * abs(pw[s].mean() - yw[s].mean())
                rows.append(dict(window=w, decile=str(j + 1), n=int(s.sum()), mean_p=float(pw[s].mean()), freq=float(yw[s].mean()), abs_gap=float(abs(pw[s].mean() - yw[s].mean()))))
        rows.append(dict(window=w, decile="итого", n=int(len(pw)), mean_p=float(pw.mean()), freq=float(yw.mean()), abs_gap=float(ece)))
    return pd.DataFrame(rows)


def beta_table(x2: pd.DataFrame) -> pd.DataFrame:
    """Выбранный β по годам переоценки (дни) и средний p, c_n, c_s по годам цели t⁺."""
    y, rows = pd.DatetimeIndex(x2["t_plus"]).year.to_numpy(), []
    for yy in list(range(2018, 2027)):
        m = y == yy
        if m.any():
            rows.append(dict(year=str(yy), n=int(m.sum()), mean_p=float(x2["p"][m].mean()), mean_cn=float(x2["cn"][m].mean()), mean_cs=float(x2["cs"][m].mean()),
                             share_beta_1=float((x2["beta"][m] == 1.0).mean()), stress_freq=float((x2["rv"][m] > x2["thr"][m]).mean())))
    return pd.DataFrame(rows)


def eval_numbers(t: dict, st: pd.DataFrame, cal: pd.DataFrame, bt: pd.DataFrame) -> dict:
    """`E2.eval`: QLIKE по окнам, Δ с интервалами, пары, «go», QLIKE по состояниям, калибровка — БЕЗ p-значений."""
    m, tt, pr = t["METRICS"], t["TESTS"], t["PAIRS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in tt.itertuples()}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    stq = {"%s|%s|%s" % (r.series, r.window, r.state): (float(r.qlike) if np.isfinite(r.qlike) else None) for r in st.itertuples()}
    cl = {"%s|%s" % (r.window, r.decile): [float(r.mean_p), float(r.freq), int(r.n)] for r in cal.itertuples()}
    return dict(qlike=ql, delta_pct_x2=dl, pairs=pair, go_x2=go, qlike_by_state=stq, calibration=cl, p_values="не входят: записаны в E2.rules после этого ключа",
                years={r.year: [float(r.mean_p), float(r.mean_cn), float(r.mean_cs)] for r in bt.itertuples()})


def rules(t: dict) -> dict:
    """6.2 (с p) и «go» 6.1: X2 против A0g и cfg4 на 2018–2021; «подтверждено» — p Холма (N = 738) < 0,05 против K2 и K3."""
    g, tt, pr = t["GO"], t["TESTS"][(t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])], t["PAIRS"]
    return dict(go=bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]],
                pairs_62={"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62, float(r.p_holm_N)] for r in pr.itertuples()})


def own_base(pairs: pd.DataFrame, y: str) -> dict:
    """Правило 6.2 для X2 против базы y на 2018–2021."""
    r = pairs[(pairs["x"] == "X2") & (pairs["y"] == y) & (pairs["window"] == "2018_2021")].iloc[0]
    return dict(delta_pct=float(r["delta_pct"]), lo90=float(r["lo90"]), hi90=float(r["hi90"]), delta_q=float(r["delta_q"]), p_one=float(r["p_one"]), years_neg=int(r["years_neg"]), n_years=int(r["n_years"]), rule62=str(r["rule62"]))
