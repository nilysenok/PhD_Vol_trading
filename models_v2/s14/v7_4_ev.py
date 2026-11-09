"""V7-4a, G46: оценка конфигурации 10 (чистые функции от массивов; файлы не читаются, ничего не пишется). Метрики, Δ, DM, Холм, «go», стресс, MCS — `v7_1b_ev` (импорт); пары и правило 6.2 — `v7_2_ev.pairs_table` (импорт) — как в `v7_3_ev`.
Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, V4:NTP2, V6:A2, V6:A0, V6:A0g и конфигурации 1–8 (рядом), 10. Основная пара — 10 против 6 (вклад совместного обучения; правило 6.2 на 2018–2021; стресс — без 1 % крайних дат пары и без 2018);
справочно — 10 против 8. Δ к K2, K3, M, K8, NTP2, A2, A0g (и K1, A0) — в % уровня K1 с 90 % интервалом и DM; «go» 6.1. Веса π по годам — по строкам Stage B (2018 — прогноз Stage A, в веса не входит).
`cond11` — условие конфигурации 11 (план): 10 несёт новую информацию против 6 по правилу 6.2 на 2018–2021. 2018 у обеих конфигураций одна и та же строка (Δ = 0), поэтому «лет Δ < 0» считается по 2019–2021 и 2018 не помогает."""
from __future__ import annotations

import pandas as pd

from models_v2.s14 import v7_1b_ev as EV, v7_2_ev as E2, v7_3_ev as E3, v7_2_evspec as SP

CFGS = E3.CFGS + ("cfg10",)
MAIN = ("H10 совместное", "cfg10", "cfg6")
PAIRS10 = (MAIN, ("10 против 8 (справочно)", "cfg10", "cfg8"))
QUANT = ("pi0", "pi1", "pi2")
PERIODS = SP.PERIODS
COND_WIN = "2018_2021"


def weights_table(df: pd.DataFrame) -> pd.DataFrame:
    """Средние π_0, π_1, π_2 по годам цели 2018…2026 и до / после 2021; ТОЛЬКО строки Stage B (stageA_2018 = 0, fitted > 0)."""
    d = df[(df["stageA_2018"] == 0) & (df["fitted"] > 0)]
    y = pd.DatetimeIndex(d["t_plus"]).year.to_numpy()
    sel = [(str(yy), y == yy) for yy in range(2018, 2027)] + [(PERIODS[-2], y <= 2021), (PERIODS[-1], y >= 2022)]
    return pd.DataFrame([dict(config="cfg10", quantity=q, period=n, n=int(m.sum()), mean=float(d[q].to_numpy(float)[m].mean()) if m.any() else float("nan")) for q in QUANT for n, m in sel])


def main_tables(rv, s: dict, tp, df10: pd.DataFrame) -> dict:
    """METRICS, MCS, PAIRS (10 против 6 и 8), TESTS / GO / YEARS (конфигурация 10), STRESS, WEIGHTS."""
    refs = {k: s[k] for k in EV.REFS}
    t = {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "PAIRS": E2.pairs_table(rv, s, tp, PAIRS10),
         "TESTS": EV.tests_table(rv, s["cfg10"], refs, tp).assign(config="cfg10"), "GO": EV.go_table(rv, s["cfg10"], refs, tp).assign(config="cfg10"),
         "YEARS": EV.years_table(rv, s["cfg10"], refs, tp).assign(config="cfg10"), "STRESS": EV.stress_table(rv, {k: s[k] for k in E2.CTRL + CFGS}, tp)}
    t["WEIGHTS"] = weights_table(df10)
    return t


def cond11(pairs: pd.DataFrame) -> dict:
    """Условие 11: правило 6.2 для пары 10 против 6 на 2018–2021 (Δ < 0, DM p < 0,05, Δ < 0 не менее чем в 3 годах из 4). МУТАНТ «пара 6 против 10 / другое окно» меняет результат."""
    r = pairs[(pairs["hyp"] == MAIN[0]) & (pairs["x"] == MAIN[1]) & (pairs["y"] == MAIN[2]) & (pairs["window"] == COND_WIN)].iloc[0]
    return dict(cond11=bool(r["rule62"] == "да"), pair="10 против 6", window=COND_WIN, delta_pct=float(r["delta_pct"]), lo90=float(r["lo90"]), hi90=float(r["hi90"]), delta_q=float(r["delta_q"]), p_one=float(r["p_one"]),
                years_neg=int(r["years_neg"]), n_years=int(r["n_years"]), p_holm_N=float(r["p_holm_N"]), rule="6.2: Δ < 0, DM p < 0,05, Δ < 0 не менее чем в 3 годах из 4",
                note="окно только 2018–2021; 2022+ в условие не входит; конфигурация 11 здесь не считается (часть 4b — по этому ключу)")
