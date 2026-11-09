"""V7-2b: текст `LEAF2_REPORT.md`. Каждое число берётся из таблиц оценки `t` (`2_*.csv`) и CHECKPOINT; в шаблоне чисел-результатов нет (тест: изменение ячейки таблицы меняет текст).
Форматирование и `md` — из `v7_1b_evrep` (импорт)."""
from __future__ import annotations

import pandas as pd

from models_v2.s14.v7_1b_evrep import f, md

WIN5 = ("2018_2021", "2019_2021", "2022_2025", "2026", "all")
YEARS9 = tuple(str(y) for y in range(2018, 2026)) + ("2026",)
ORDER = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g") + tuple("cfg%d" % i for i in range(1, 8))
NAME = {"cfg1": "1 (TCN-w5, g(z))", "cfg2": "2 (CNN-w5, g(z))", "cfg3": "3 (w1)", "cfg4": "4 (g(z,Z))", "cfg5": "5 (MoE A)", "cfg6": "6 (MoE B)", "cfg7": "7 (MoE C)"}
DBASE = ("K2", "K3", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g")


def q_pivot(m: pd.DataFrame, wins) -> pd.DataFrame:
    p = m.pivot(index="series", columns="window", values="qlike")
    return pd.DataFrame([[s] + [f(p.loc[s, w], 6) for w in wins] for s in ORDER], columns=["ряд"] + list(wins))


def v1_q(v1: pd.DataFrame) -> pd.DataFrame:
    rows = [[NAME[r.config], r.window, f(r.qlike_v1, 6), f(r.qlike_v2, 6), f(r.d_k3_v1, 2), "[%s; %s]" % (f(r.lo90_v1, 2), f(r.hi90_v1, 2)), f(r.d_k3_v2, 2), "[%s; %s]" % (f(r.lo90_v2, 2), f(r.hi90_v2, 2))] for r in v1.itertuples()]
    return pd.DataFrame(rows, columns=["конфигурация", "окно", "QLIKE v1", "QLIKE v2", "Δ к K3 v1, % K1", "90 % v1", "Δ к K3 v2, % K1", "90 % v2"])


def delta_all(tt: pd.DataFrame) -> pd.DataFrame:
    """Сводная Δ (Ж-236 п. 6): конфигурация × база, окна 2018–2021 и 2022–2025: Δ в % K1, 90 %, DM p, Холм N."""
    rows = []
    for c in ("cfg%d" % i for i in range(1, 8)):
        for b in DBASE:
            r = {w: tt[(tt["config"] == c) & (tt["window"] == w) & (tt["base"] == b)].iloc[0] for w in ("2018_2021", "2022_2025")}
            a, z = r["2018_2021"], r["2022_2025"]
            rows.append([NAME[c], b, f(a["delta_pct"], 2), "[%s; %s]" % (f(a["lo90"], 2), f(a["hi90"], 2)), f(a["p_one"], 4), f(a["p_holm_N"], 4), f(z["delta_pct"], 2), "[%s; %s]" % (f(z["lo90"], 2), f(z["hi90"], 2)), f(z["p_one"], 4)])
    return pd.DataFrame(rows, columns=["конфигурация", "база", "Δ 2018–2021", "90 %", "DM p", "Холм N=733", "Δ 2022–2025", "90 %", "DM p"])


def pairs_md(p: pd.DataFrame) -> pd.DataFrame:
    rows = [[r.hyp, "%s против %s" % (NAME.get(r.x, r.x), NAME.get(r.y, r.y)), r.window, f(r.delta_pct, 2), "[%s; %s]" % (f(r.lo90, 2), f(r.hi90, 2)), f(r.p_one, 4), f(r.p_holm_N, 4), f(r.p_holm_Nwide, 4), r.years_neg, r.n_years, r.rule62]
            for r in p.itertuples()]
    return pd.DataFrame(rows, columns=["гипотеза", "пара X против Y", "окно", "Δ, % K1", "90 %", "DM p", "Холм N=733", "Холм N=3166", "лет Δ<0", "лет", "6.2"])


def go_md(g: pd.DataFrame) -> pd.DataFrame:
    return g.assign(value=g["value"].map(lambda x: f(x, 6)), threshold=g["threshold"].map(lambda x: f(x, 4)), config=g["config"].map(NAME))[["config", "cond", "name", "value", "threshold", "passed"]]


def spec_md(s: pd.DataFrame) -> pd.DataFrame:
    c = ["share_pi_gt03", "switch_freq", "mean_abs_dpi", "spearman_pi_gain_hindsight"]
    g = ["gain_on", "gain_rest", "diff_on_rest_lo90", "diff_on_rest_hi90"]
    return s.assign(config=s["config"].map(NAME), **{k: s[k].map(lambda x: f(x, 4)) for k in c}, **{k: s[k].map(lambda x: f(x, 6)) for k in g},
                    gain_on_ci=[("[%s; %s]" % (f(a, 6), f(b, 6))) for a, b in zip(s["gain_on_lo90"], s["gain_on_hi90"])], gain_rest_ci=[("[%s; %s]" % (f(a, 6), f(b, 6))) for a, b in zip(s["gain_rest_lo90"], s["gain_rest_hi90"])])[
        ["config", "k", "window", "n", "share_pi_gt03", "switch_freq", "mean_abs_dpi", "gain_on", "gain_on_ci", "gain_rest", "gain_rest_ci", "diff_on_rest_lo90", "diff_on_rest_hi90", "spearman_pi_gain_hindsight", "specialized"]]


def weights_md(w: pd.DataFrame) -> pd.DataFrame:
    p = w.assign(m=w["mean_pi"].map(lambda x: f(x, 3))).pivot_table(index=["config", "k"], columns="period", values="m", aggfunc="first")
    return p.reset_index().assign(config=lambda d: d["config"].map(NAME))


def cond9_md(c: dict) -> pd.DataFrame:
    return pd.DataFrame([[k, f(v, 4) if isinstance(v, float) else str(v)] for k, v in c.items()], columns=["показатель", "значение"])


def render(t: dict, inp: dict, cp: dict) -> str:
    c2, cd = cp["rules"], cp["cond9"]
    L = ["# LEAF2 — лист 2b V7: оценка конфигураций 1–7 версии 2 (SBER, h = 1)", "",
         "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; все цифры — потолок (первый счёт, дизайн K2/K3 выбирался на SBER 2017–2021; окна 2022+ — только оценка, выбора по ним нет). Числа — из таблиц `2_*.csv` и CHECKPOINT ключ `2`.", "",
         "Одинаковые даты: %d, %s … %s. Конфигурации — версия 2 (после листа FX); версия 1 (дефект согласования z) показана рядом в разделе 9 только для сравнения. Правила 6.1 и 6.2 считаются по версии 2." % (len(inp["idx"]), inp["idx"][0].date(), inp["idx"][-1].date()), "",
         "## 1. QLIKE по окнам (одинаковые даты)", "", md(q_pivot(t["METRICS"], WIN5)), "",
         "## 2. QLIKE по годам цели 2018–2026", "", md(q_pivot(t["METRICS"], YEARS9)), "",
         "## 3. Сводная Δ к K2, K3, NTP2, A2, A0, A0g: Δ = QLIKE(конфигурация) − QLIKE(база) в % среднего QLIKE K1 окна (минус = лучше базы)", "", md(delta_all(t["TESTS"])), "",
         "## 4. Гипотезы H1, H2, H3 и 4–7 против баз (правило 6.2: Δ < 0, DM p < 0,05, Δ < 0 в ≥ 3 годах из 4; стресс — без 1 % крайних дат пары и без 2018)", "", md(pairs_md(t["PAIRS"])), "",
         "## 5. «Go» 6.1 (шесть условий), конфигурации 1–7", "", md(go_md(t["GO"])), "",
         "Значение условия 5 «inf» означает: суммарный выигрыш над лучшим из K2/K3 за 2018–2021 не положителен. Статусы: " + "; ".join("%s — %s" % (NAME[c], c2[c]["status"]) for c in NAME), "",
         "## 6. Специализация экспертов 5–7 (6.3; описательно; приближение «без эксперта» — см. `approx_err` в `2_SPEC.csv`)", "", md(spec_md(t["SPEC"])), "",
         "Связь π с преимуществом (`spearman_pi_gain_hindsight`) — задним числом, по фактическим RV; выбора по ней нет. Оговорки: π и g_k усреднены по 5 seeds (частота переключений |Δπ| > 0,5 по средним весам); "
         "порог 0,3 лежит ниже равномерного веса 1/3 при трёх выходах, поэтому доля дней π_k > 0,3 близка к единице у почти равномерных весов и специализации сама по себе не показывает — решает интервал разности «на днях π_k > 0,3 − остальные».", "",
         "## 7. Средние веса π по годам, до и после 2021", "", md(weights_md(t["WEIGHTS"])), "",
         "## 8. Условие конфигурации 9 (по конфигурации 6, окно 2018–2021)", "", md(cond9_md(cd)), "",
         "## 9. Версия 1 рядом (QLIKE по окнам и Δ к K3)", "", md(v1_q(t["V1"])), "",
         "## 10. Стресс (QLIKE и разность к лучшему из K2/K3)", "", md(t["STRESS"][t["STRESS"]["series"].isin(["K2", "K3", "V6:A2"] + ["cfg%d" % i for i in range(1, 8)])].assign(
             qlike=lambda d: d["qlike"].map(lambda x: f(x, 6)), gap_best_k2k3=lambda d: d["gap_best_k2k3"].map(lambda x: f(x, 6)))), "",
         "## 11. MCS 90 %", "", md(t["MCS"][t["MCS"]["in_mcs"]].assign(mean_qlike=lambda d: d["mean_qlike"].map(lambda x: f(x, 6)), p=lambda d: d["p"].map(f))[["window", "series", "mean_qlike", "p"]]), "",
         "Холм с N = 733 и N_wide = 3166 (семейство — все строки окна в таблице). Полные таблицы — `2_METRICS.csv`, `2_TESTS.csv`, `2_PAIRS.csv`, `2_GO.csv`, `2_YEARS.csv`, `2_STRESS.csv`, `2_MCS.csv`, `2_SPEC.csv`, `2_WEIGHTS.csv`, `2_V1.csv`."]
    return "\n".join(L) + "\n"
