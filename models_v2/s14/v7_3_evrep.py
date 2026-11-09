"""V7-3: текст `LEAF3_REPORT.md`. Каждое число берётся из таблиц оценки `t` (`3_*.csv`) и CHECKPOINT; в шаблоне чисел-результатов нет (тест: изменение ячейки таблицы меняет текст).
Форматирование и `md` — из `v7_1b_evrep` (импорт); QLIKE-сводки — `v7_2_evrep.q_pivot` по своему списку рядов."""
from __future__ import annotations

import pandas as pd

from models_v2.s14.v7_1b_evrep import f, md
from models_v2.s14.v7_2_evrep import NAME as NAME2, WIN5, YEARS9

NAME = {**NAME2, "cfg8": "8 (MoE B + якорь)"}
ORDER = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g") + tuple("cfg%d" % i for i in range(1, 9))
SHOWN = ("K2", "K3", "V4:NTP2", "V6:A2", "cfg4", "cfg6", "cfg8")
CI = lambda lo, hi: "[%s; %s]" % (f(lo, 2), f(hi, 2))


def q_pivot(m: pd.DataFrame, wins, order=ORDER) -> pd.DataFrame:
    p = m.pivot(index="series", columns="window", values="qlike")
    return pd.DataFrame([[s] + [f(p.loc[s, w], 6) for w in wins] for s in order], columns=["ряд"] + list(wins))


def delta8(tt: pd.DataFrame) -> pd.DataFrame:
    """Δ конфигурации 8 (QLIKE 8 − QLIKE базы) в % K1: окна 2018–2021 и 2022–2025, 90 %, DM p, Холм N."""
    rows = []
    for b in ("K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0g"):
        a, z = (tt[(tt["window"] == w) & (tt["base"] == b)].iloc[0] for w in ("2018_2021", "2022_2025"))
        rows.append([b, f(a["delta_pct"], 2), CI(a["lo90"], a["hi90"]), f(a["p_one"], 4), f(a["p_holm_N"], 4), f(z["delta_pct"], 2), CI(z["lo90"], z["hi90"]), f(z["p_one"], 4)])
    return pd.DataFrame(rows, columns=["база", "Δ 2018–2021", "90 %", "DM p", "Холм N=733", "Δ 2022–2025", "90 %", "DM p"])


def pairs_md(p: pd.DataFrame) -> pd.DataFrame:
    rows = [[r.hyp, "%s против %s" % (NAME[r.x], NAME[r.y]), r.window, f(r.delta_pct, 2), CI(r.lo90, r.hi90), f(r.delta_q, 6), f(r.p_one, 4), f(r.p_holm_N, 4), f(r.p_holm_Nwide, 4), r.years_neg, r.n_years, r.rule62]
            for r in p.itertuples()]
    return pd.DataFrame(rows, columns=["гипотеза", "пара X против Y", "окно", "Δ, % K1", "90 %", "ΔQLIKE", "DM p", "Холм N=733", "Холм N=3166", "лет Δ<0", "лет", "6.2"])


def go_md(g: pd.DataFrame) -> pd.DataFrame:
    return g.assign(value=g["value"].map(lambda x: f(x, 6)), threshold=g["threshold"].map(lambda x: f(x, 4)))[["cond", "name", "value", "threshold", "passed"]]


def weights_md(w: pd.DataFrame) -> pd.DataFrame:
    p = w.assign(m=w["mean"].map(lambda x: f(x, 3))).pivot_table(index="quantity", columns="period", values="m", aggfunc="first")
    return p.reset_index()


def stress_md(s: pd.DataFrame) -> pd.DataFrame:
    return s[s["series"].isin(SHOWN)].assign(qlike=lambda d: d["qlike"].map(lambda x: f(x, 6)), gap_best_k2k3=lambda d: d["gap_best_k2k3"].map(lambda x: f(x, 6)))


def render(t: dict, inp: dict, cp3: dict) -> str:
    r = cp3["rules"]["cfg8"]
    L = ["# LEAF3 — лист 3 V7: конфигурация 8, Residual MoE B с якорем на бустинг (SBER, h = 1)", "",
         "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; все цифры — потолок (первый счёт, дизайн K2/K3 выбирался на SBER 2017–2021; окна 2022+ — только оценка, выбора по ним нет). Расходуется 1 конфигурация (8 из 11); 9 не считается (условие 2.cond9 ложно). Числа — из таблиц `3_*.csv` и CHECKPOINT ключ `3`.", "",
         "Одинаковые даты: %d, %s … %s. F = H·exp(w·ℓ + Σ π_k(Z, z)·g_k(z)), ℓ = ln(M/H), M = √(K2·K3); 78 параметров. Конфигурации 1–7 — версия 2 (после листа FX)." % (len(inp["idx"]), inp["idx"][0].date(), inp["idx"][-1].date()), "",
         "## 1. QLIKE по окнам (одинаковые даты)", "", md(q_pivot(t["METRICS"], WIN5)), "",
         "## 2. QLIKE по годам цели 2018–2026", "", md(q_pivot(t["METRICS"], YEARS9)), "",
         "## 3. Δ конфигурации 8 к K2, K3, M, K8, NTP2, A2, A0g: Δ = QLIKE(8) − QLIKE(база) в % среднего QLIKE K1 окна (минус = лучше базы)", "", md(delta8(t["TESTS"])), "",
         "## 4. Восемь против шести (вклад якоря; правило 6.2: Δ < 0, DM p < 0,05, Δ < 0 в ≥ 3 годах из 4; стресс — без 1 % крайних дат пары и без 2018 (окно 2019–2021))", "", md(pairs_md(t["PAIRS"])), "",
         "## 5. «Go» 6.1 (шесть условий), конфигурация 8", "", md(go_md(t["GO"])), "",
         "Значение условия 5 «inf» означает: суммарный выигрыш над лучшим из K2/K3 за 2018–2021 не положителен. Статус: %s (p Холма N = 733 против K2 и K3 на 2018–2021: %s)." % (r["status"], "; ".join(f(x, 4) for x in r["p_holm_N_vs_K2_K3"])), "",
         "## 6. Средние веса π_0, π_1, π_2 и общий скаляр w по годам цели, до и после 2021 (строки с переоценкой; π, w — среднее по 5 seeds)", "", md(weights_md(t["WEIGHTS"])), "",
         "## 7. Стресс (QLIKE и разность к лучшему из K2/K3)", "", md(stress_md(t["STRESS"])), "",
         "## 8. MCS 90 %", "", md(t["MCS"][t["MCS"]["in_mcs"]].assign(mean_qlike=lambda d: d["mean_qlike"].map(lambda x: f(x, 6)), p=lambda d: d["p"].map(f))[["window", "series", "mean_qlike", "p"]]), "",
         "Холм с N = 733 и N_wide = 3166 (семейство — все строки окна в таблице). Полные таблицы — `3_METRICS.csv`, `3_TESTS.csv`, `3_PAIRS.csv`, `3_GO.csv`, `3_YEARS.csv`, `3_STRESS.csv`, `3_MCS.csv`, `3_WEIGHTS.csv`."]
    return "\n".join(L) + "\n"
