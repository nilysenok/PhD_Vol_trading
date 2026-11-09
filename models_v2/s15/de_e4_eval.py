"""E4: оценка X4 — чистые функции от массивов. Метрики, Δ, DM, «go» 6.1, стресс, MCS, пары 6.2 — функции V7 (импорт через `de_e1_eval`), Холм N = 738 / N_wide = 3181 (как E1–E3).
Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, NTP2, A2, A0, A0g, cfg8, X1, ENS (ансамбль без затвора, α ≡ 1, справочный, НЕ конфигурация) и X4. Пары 6.2 X4: против ENS и против A0g (G14), справочно cfg8, X1, K2, K3."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de_e1_eval as E1V

EV, ST, E2 = E1V.EV, E1V.ST, E1V.E2
SERIES = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g", "cfg8", "X1", "ENS", "X4")
PAIRS = (("X4 против ансамбля без затвора (6.2, польза затвора)", "X4", "ENS"), ("X4 против A0g (6.2)", "X4", "V6:A0g"), ("X4 против cfg8", "X4", "cfg8"), ("X4 против X1", "X4", "X1"), ("X4 против K2", "X4", "K2"), ("X4 против K3", "X4", "K3"))
STATE_SERIES = ("X4", "ENS", "V6:A0g", "X1", "V4:NTP2", "cfg8", "K2", "K3")


def main_tables(rv, s: dict, tp) -> dict:
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    refs = {k: s[k] for k in EV.REFS}
    return {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "TESTS": E1V._holm(EV.tests_table(rv, s["X4"], refs, tp), EV.TEST_WIN).assign(config="X4"),
            "GO": EV.go_table(rv, s["X4"], refs, tp).assign(config="X4"), "YEARS": EV.years_table(rv, s["X4"], refs, tp).assign(config="X4"),
            "STRESS": EV.stress_table(rv, {k: s[k] for k in E2.CTRL + ("cfg8", "X1", "ENS", "X4")}, tp), "PAIRS": E1V._holm(E2.pairs_table(rv, s, tp, PAIRS), E2.PAIR_WIN)}


def state_qlike(rv, s: dict, tp, state) -> pd.DataFrame:
    rv, tp, state, rows = np.asarray(rv, float), np.asarray(tp), np.asarray(state), []
    for name in STATE_SERIES:
        q = E1V.Q.ql(rv, np.asarray(s[name], float))
        for w in E1V.STATE_WIN:
            for k, lab in ((0, "норма"), (1, "экстрем")):
                m = ST.wmask(tp, w) & (state == k)
                rows.append(dict(series=name, window=w, state=lab, n=int(m.sum()), qlike=float(q[m].mean()) if m.any() else float("nan")))
    return pd.DataFrame(rows)


def alpha_table(x4: pd.DataFrame) -> pd.DataFrame:
    """α по годам цели t⁺: дней, доля дней с act, средний α, доля α < 0,5, средний sd_r, средний |ḡ|, доли κ = 0,5 / 1 / 2."""
    y, rows = pd.DatetimeIndex(x4["t_plus"]).year.to_numpy(), []
    for yy in list(range(2018, 2027)):
        m = y == yy
        if m.any():
            rows.append(dict(year=str(yy), n=int(m.sum()), share_act=float(x4["act"][m].mean()), mean_alpha=float(x4["alpha"][m].mean()), share_alpha_lt_half=float((x4["alpha"][m] < 0.5).mean()),
                             mean_sd=float(x4["sd"][m].mean()), mean_abs_gbar=float(x4["gbar"][m].abs().mean()), **{"share_kappa_%s" % k.replace(".", "_"): float((x4["kappa"][m] == float(k)).mean()) for k in ("0.5", "1.0", "2.0")}))
    return pd.DataFrame(rows)


def eval_numbers(t: dict, st: pd.DataFrame, at: pd.DataFrame) -> dict:
    """`E4.eval`: QLIKE по окнам, Δ с интервалами, пары, «go», QLIKE по состояниям, α по годам — БЕЗ p-значений."""
    m, tt, pr = t["METRICS"], t["TESTS"], t["PAIRS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in tt.itertuples()}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    stq = {"%s|%s|%s" % (r.series, r.window, r.state): (float(r.qlike) if np.isfinite(r.qlike) else None) for r in st.itertuples()}
    return dict(qlike=ql, delta_pct_x4=dl, pairs=pair, go_x4=go, qlike_by_state=stq, p_values="не входят: записаны в E4.rules после этого ключа",
                alpha_by_year={r.year: [float(r.mean_alpha), float(r.mean_sd), float(r.share_act)] for r in at.itertuples()})


def rules(t: dict) -> dict:
    """6.2 (с p) и «go» 6.1: X4 против ENS и A0g на 2018–2021; «подтверждено» — p Холма (N = 738) < 0,05 против K2 и K3."""
    g, tt, pr = t["GO"], t["TESTS"][(t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])], t["PAIRS"]
    return dict(go=bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]],
                pairs_62={"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62, float(r.p_holm_N)] for r in pr.itertuples()})


def own_base(pairs: pd.DataFrame, y: str) -> dict:
    r = pairs[(pairs["x"] == "X4") & (pairs["y"] == y) & (pairs["window"] == "2018_2021")].iloc[0]
    return dict(delta_pct=float(r["delta_pct"]), lo90=float(r["lo90"]), hi90=float(r["hi90"]), delta_q=float(r["delta_q"]), p_one=float(r["p_one"]), years_neg=int(r["years_neg"]), n_years=int(r["n_years"]), rule62=str(r["rule62"]))
