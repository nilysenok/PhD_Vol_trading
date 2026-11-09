"""V7-4a: текст `LEAF4_REPORT.md`. Каждое число берётся из таблиц оценки `t` (`4_*.csv`), `4_DRIFT.csv` и CHECKPOINT ключ `4`; в шаблоне чисел-результатов нет (тест: изменение ячейки таблицы меняет текст).
Форматирование и `md` — `v7_1b_evrep` (импорт); QLIKE-сводки и Δ — `v7_3_evrep` (импорт, без копий)."""
from __future__ import annotations

import pandas as pd

from models_v2.s14.v7_1b_evrep import f, md, verdict
from models_v2.s14 import v7_3_evrep as R3
from models_v2.s14.v7_2_evrep import WIN5, YEARS9

NAME = {**R3.NAME, "cfg10": "10 (совместное B)"}
ORDER = R3.ORDER + ("cfg10",)
SHOWN = ("K2", "K3", "V4:NTP2", "V6:A2", "cfg4", "cfg6", "cfg8", "cfg10")
CI = R3.CI


def pairs_md(p: pd.DataFrame) -> pd.DataFrame:
    rows = [[r.hyp, "%s против %s" % (NAME[r.x], NAME[r.y]), r.window, f(r.delta_pct, 2), CI(r.lo90, r.hi90), f(r.delta_q, 6), f(r.p_one, 4), f(r.p_holm_N, 4), f(r.p_holm_Nwide, 4), r.years_neg, r.n_years, r.rule62] for r in p.itertuples()]
    return pd.DataFrame(rows, columns=["гипотеза", "пара X против Y", "окно", "Δ, % K1", "90 %", "ΔQLIKE", "DM p", "Холм N=733", "Холм N=3166", "лет Δ<0", "лет", "6.2"])


def stress_md(s: pd.DataFrame) -> pd.DataFrame:
    return s[s["series"].isin(SHOWN)].assign(qlike=lambda d: d["qlike"].map(lambda x: f(x, 6)), gap_best_k2k3=lambda d: d["gap_best_k2k3"].map(lambda x: f(x, 6)))


def weights_md(w: pd.DataFrame) -> pd.DataFrame:
    """Как `R3.weights_md`, но 2018 (прогноз Stage A, весов Stage B нет) — «—», а не «nan»."""
    p = w.assign(m=w["mean"].map(lambda x: f(x, 3) if x == x else "—")).pivot_table(index="quantity", columns="period", values="m", aggfunc="first")
    return p.reset_index()


def drift_md(dr: pd.DataFrame) -> pd.DataFrame:
    """Средние по 5 seeds по годам: потери задачи-предлога A и B на днях обучения и года Y, согласие z (|corr| после выравнивания)."""
    g = dr.groupby("year").mean(numeric_only=True)
    cols = ["pretext_A_train", "pretext_B_train", "pretext_A_year", "pretext_B_year", "corr1", "corr2", "corr3", "corr4", "corr_mean_year", "corr_mean_past"]
    out = g[cols].apply(lambda c: c.map(lambda x: f(x, 4))).reset_index()
    return out.rename(columns={"year": "год", "pretext_A_train": "потеря A, обучение", "pretext_B_train": "потеря B, обучение", "pretext_A_year": "потеря A, год Y", "pretext_B_year": "потеря B, год Y",
                               "corr_mean_year": "|corr| среднее, год Y", "corr_mean_past": "|corr| среднее, до Y"})


def cond11_md(c: dict) -> pd.DataFrame:
    return pd.DataFrame([["Δ, % K1", f(c["delta_pct"], 2)], ["90 %", CI(c["lo90"], c["hi90"])], ["ΔQLIKE", f(c["delta_q"], 6)], ["DM p (односторонний)", f(c["p_one"], 4)], ["лет Δ<0", "%d из %d" % (c["years_neg"], c["n_years"])],
                         ["Холм N=733", f(c["p_holm_N"], 4)], ["условие 11", "истинно" if c["cond11"] else "ложно"]], columns=["показатель", "значение"])


def render(t: dict, inp: dict, cp4: dict, dr: pd.DataFrame) -> str:
    r, c = cp4["rules"]["cfg10"], cp4["cond11"]
    L = ["# LEAF4 — лист 4a V7: конфигурация 10, совместное обучение Stage B для 6 (SBER, h = 1)", "",
         "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; все цифры — потолок (первый счёт, дизайн K2/K3 выбирался на SBER 2017–2021; окна 2022+ — только оценка, выбора по ним нет). Расходуется 1 конфигурация (10); 11 здесь не считается. Числа — из таблиц `4_*.csv` и CHECKPOINT ключ `4`.", "",
         "Число одинаковых дат у всех рядов: %d, от %s до %s. Календарь (Ж-235): совместное переобучение энкодера, экспертов и маршрутизатора — 1-й торговый день января Y = 2019…2026 (пары t⁺ < 01.01.Y, бары < 01.01.Y, старт — веса Stage A года Y того же seed, штраф λ_e = %g); внутри года — помесячно только эксперты и маршрутизатор при замороженном энкодере; 2018 — прогноз 6 (Stage A), %d строк с пометкой `stageA_2018`. Конфигурации 1–7 — версия 2 (после листа FX), 8 — лист 3."
         % (len(inp["idx"]), inp["idx"][0].date(), inp["idx"][-1].date(), cp4["cfg10"]["lam_e"], cp4["cfg10"]["stageA_2018_rows"]), "",
         "## 1. QLIKE по окнам (одинаковые даты)", "", md(R3.q_pivot(t["METRICS"], WIN5, ORDER)), "",
         "## 2. QLIKE по годам цели 2018–2026", "", md(R3.q_pivot(t["METRICS"], YEARS9, ORDER)), "",
         "## 3. Δ конфигурации 10 к K2, K3, M, K8, NTP2, A2, A0g: Δ = QLIKE(10) − QLIKE(база) в % среднего QLIKE K1 окна (минус = лучше базы)", "", md(R3.delta8(t["TESTS"])), "",
         "## 4. Десять против шести (правило 6.2: Δ < 0, DM p < 0,05, Δ < 0 в ≥ 3 годах из 4; стресс — без 1 % крайних дат пары и без 2018 (окно 2019–2021)); десять против восьми — справочно", "", md(pairs_md(t["PAIRS"])), "",
         "## 5. «Go» 6.1 (шесть условий), конфигурация 10", "", md(R3.go_md(t["GO"])), "",
         "Статус: %s (p Холма N = 733 против K2 и K3 на 2018–2021: %s)." % (r["status"], "; ".join(f(x, 4) for x in r["p_holm_N_vs_K2_K3"])), "",
         "## 6. Условие конфигурации 11 (`4.cond11`: 10 несёт новую информацию против 6 по правилу 6.2 на 2018–2021)", "", md(cond11_md(c)), "",
         "## 7. Дрейф энкодера Stage B против Stage A (диагностика Ж-235 п. 5; частота переобучения не менялась): потеря задачи-предлога и согласие z после выравнивания, среднее по 5 seeds", "",
         "Оговорка: потеря предлога измерена старой головой Stage A на скрытых состояниях обученного энкодера (голова при совместном обучении не переобучается), поэтому её рост — смещение представления, а не прямая мера потери информации; |corr| — после однозначного сопоставления компонент.", "", md(drift_md(dr)), "",
         "## 8. Средние веса π_0, π_1, π_2 по годам цели, до и после 2021 (строки Stage B; среднее по 5 seeds)", "", md(weights_md(t["WEIGHTS"])), "",
         "## 9. Стресс (QLIKE и разность к лучшему из K2/K3)", "", md(stress_md(t["STRESS"])), "",
         "## 10. MCS 90 %", "", md(t["MCS"][t["MCS"]["in_mcs"]].assign(mean_qlike=lambda d: d["mean_qlike"].map(lambda x: f(x, 6)), p=lambda d: d["p"].map(f))[["window", "series", "mean_qlike", "p"]]), "",
         "Холм с N = 733 и N_wide = 3166 (семейство — все строки окна в таблице). Полные таблицы — `4_METRICS.csv`, `4_TESTS.csv`, `4_PAIRS.csv`, `4_GO.csv`, `4_YEARS.csv`, `4_STRESS.csv`, `4_MCS.csv`, `4_WEIGHTS.csv`, `4_DRIFT.csv`."]
    return "\n".join(L) + "\n"
