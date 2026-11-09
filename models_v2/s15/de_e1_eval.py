"""E1: оценка X1 — чистые функции от массивов (файлов нет). Метрики, Δ, DM, «go» 6.1, стресс, MCS, пары и правило 6.2 — функции V7 (`v7_1b_ev`, `v7_2_ev`) импортом; единственное отличие — Холм с N = 738 / N_wide = 3181
(733 + 5 мест, 3166 + 5·3; план раздел 4), пересчитанный по p тех же таблиц. Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, V4:NTP2, V6:A2, V6:A0, V6:A0g, cfg8, X1 (A0 — база Δ таблицы V7)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EVX

# функции оценки V7 (не обучатели) — через importlib: `test_de0_modules_import_no_trainers` (список S14_OK только для загрузчиков DE0) их по AST не пропускает; сам импорт `models_v2.s14` тянет torch
EV, ST, E2 = (importlib.import_module("models_v2.s14." + m) for m in ("v7_1b_ev", "v7_1m_stat", "v7_2_ev"))

N_HOLM, N_WIDE = 738, 3181
SERIES = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g", "cfg8", "X1")
PAIRS = (("X1 против NTP2 (6.2, польза состояния)", "X1", "V4:NTP2"), ("X1 против cfg8", "X1", "cfg8"), ("X1 против K2", "X1", "K2"), ("X1 против K3", "X1", "K3"), ("X1 против A2", "X1", "V6:A2"))
STATE_WIN = ("2018_2021", "2019_2021", "2022_2025", "2026", "all")
STATE_SERIES = ("X1", "V4:NTP2", "cfg8", "M", "K2", "K3")


def _holm(t: pd.DataFrame, windows) -> pd.DataFrame:
    t = t.copy()
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in windows:
        k = (t["window"] == w) & t["p_one"].notna()
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), {"N": N_HOLM, "N_wide": N_WIDE})
    return t


def main_tables(rv, s: dict, tp) -> dict:
    """METRICS, MCS, TESTS (Δ X1 к базам), GO, YEARS, STRESS, PAIRS — Холм N = 738 / 3181."""
    rv, tp = np.asarray(rv, float), np.asarray(tp)
    refs = {k: s[k] for k in EV.REFS}
    pairs = E2.pairs_table(rv, s, tp, PAIRS)
    t = {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "TESTS": _holm(EV.tests_table(rv, s["X1"], refs, tp), EV.TEST_WIN).assign(config="X1"),
         "GO": EV.go_table(rv, s["X1"], refs, tp).assign(config="X1"), "YEARS": EV.years_table(rv, s["X1"], refs, tp).assign(config="X1"),
         "STRESS": EV.stress_table(rv, {k: s[k] for k in E2.CTRL + ("cfg8", "X1")}, tp), "PAIRS": _holm(pairs, E2.PAIR_WIN)}
    return t


def state_qlike(rv, s: dict, tp, state) -> pd.DataFrame:
    """QLIKE по состояниям (0 — норма, 1 — экстрем) и окнам: ряд, окно, состояние, n, QLIKE; X1 и ряды сравнения. Состояние — s_d дня прогноза."""
    rv, tp, state, rows = np.asarray(rv, float), np.asarray(tp), np.asarray(state), []
    for name in STATE_SERIES:
        q = Q.ql(rv, np.asarray(s[name], float))
        for w in STATE_WIN:
            for k, lab in ((0, "норма"), (1, "экстрем")):
                m = ST.wmask(tp, w) & (state == k)
                rows.append(dict(series=name, window=w, state=lab, n=int(m.sum()), qlike=float(q[m].mean()) if m.any() else float("nan")))
    return pd.DataFrame(rows)


def pi_table(x1: pd.DataFrame) -> pd.DataFrame:
    """π по годам цели t⁺ и состояниям: дней состояния, средний π, применённый в дни своего состояния; средний π процесса состояния за все дни; средний выбранный α."""
    y, rows = pd.DatetimeIndex(x1["t_plus"]).year.to_numpy(), []
    for yy in list(range(2018, 2027)) + ["до 2021", "после 2021"]:
        sel = (y == yy) if isinstance(yy, int) else (y <= 2021 if yy == "до 2021" else y >= 2022)
        for k, lab in ((0, "норма"), (1, "экстрем")):
            m = sel & (x1["s"].to_numpy() == k)
            rows.append(dict(period=str(yy), state=lab, n=int(m.sum()), mean_pi_on_days=float(x1["pi"].to_numpy()[m].mean()) if m.any() else float("nan"),
                             mean_pi_process=float(x1["pi%d" % k].to_numpy()[sel].mean()) if sel.any() else float("nan"), mean_alpha=float(x1["a%d" % k].to_numpy()[sel].mean()) if sel.any() else float("nan")))
    return pd.DataFrame(rows)


def eval_numbers(t: dict, st: pd.DataFrame, pit: pd.DataFrame) -> dict:
    """`E1.eval`: QLIKE по окнам, Δ с интервалами, пары (Δ, интервал, годы), «go», π, QLIKE по состояниям — БЕЗ p."""
    m, tt, pr = t["METRICS"], t["TESTS"], t["PAIRS"]
    ql = {s: {w: float(g.loc[g["window"] == w, "qlike"].iloc[0]) for w in g["window"]} for s, g in m.groupby("series")}
    dl = {"%s|%s" % (r.window, r.base): [float(r.delta_pct), float(r.lo90), float(r.hi90)] for r in tt.itertuples()}
    pair = {"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_pct), float(r.lo90), float(r.hi90), float(r.delta_q), int(r.years_neg), int(r.n_years), r.rule62] for r in pr.itertuples()}
    go = {str(int(r.cond)): [float(r.value), bool(r.passed)] for r in t["GO"].itertuples()}
    stq = {"%s|%s|%s" % (r.series, r.window, r.state): (float(r.qlike) if np.isfinite(r.qlike) else None) for r in st.itertuples()}
    pi = {"%s|%s" % (r.period, r.state): (float(r.mean_pi_process) if np.isfinite(r.mean_pi_process) else None) for r in pit.itertuples()}
    return dict(qlike=ql, delta_pct_x1=dl, pairs=pair, go_x1=go, qlike_by_state=stq, pi_process=pi, p_values="не входят: записаны в E1.rules после этого ключа")


def rules(t: dict) -> dict:
    """6.2 (с p) и «go» 6.1: X1 против NTP2 на 2018–2021 — «польза состояния» при rule62 = «да»; «подтверждено» — p Холма (N = 738) < 0,05 против K2 и K3 на 2018–2021."""
    g, tt = t["GO"], t["TESTS"][(t["TESTS"]["window"] == "2018_2021") & t["TESTS"]["base"].isin(["K2", "K3"])]
    pr = t["PAIRS"]
    return dict(go=bool(g.loc[g["cond"] == 0, "passed"].iloc[0]), p_holm_N_vs_K2_K3=[float(x) for x in tt["p_holm_N"]],
                pairs_62={"%s|%s|%s|%s" % (r.hyp, r.x, r.y, r.window): [float(r.delta_q), float(r.p_one), int(r.years_neg), r.rule62, float(r.p_holm_N)] for r in pr.itertuples()})


def state_benefit(pairs: pd.DataFrame) -> dict:
    """Правило 6.2 для X1 против NTP2 на 2018–2021 (польза состояния): строка пары, rule62."""
    r = pairs[(pairs["x"] == "X1") & (pairs["y"] == "V4:NTP2") & (pairs["window"] == "2018_2021")].iloc[0]
    return dict(delta_pct=float(r["delta_pct"]), lo90=float(r["lo90"]), hi90=float(r["hi90"]), delta_q=float(r["delta_q"]), p_one=float(r["p_one"]), years_neg=int(r["years_neg"]), n_years=int(r["n_years"]), rule62=str(r["rule62"]))
