"""E3: `E3_REPORT.md` — генератор из таблиц (числа руками не вписываются; `--check` сверяет файл с `build`). Первая строка — правило 6.2 для X3 против A0g (своя база) и «go»."""
from __future__ import annotations

import pandas as pd

from models_v2.s15.de_e1_doc import WINS, YEARS, f, md

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-239/Ж-240, лист E3 (SBER, h = 1; израсходовано 3 конфигурации из 5: X1, X2, X3); окна 2022+ — только оценка, выбора по ним нет."


def build(t: dict, sb: dict, st: pd.DataFrame, at: pd.DataFrame, num: dict) -> str:
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    sers = ["X3", "V6:A0g", "X1", "V4:NTP2", "cfg8", "K2", "K3", "V6:A2"]
    q = lambda s, w: m.get((s, w), float("nan"))
    tt, pr, go = t["TESTS"], t["PAIRS"], t["GO"]
    first = "**X3 (устойчивая + адаптивная поправка), 2018–2021, правило 6.2 против A0g: %s** — Δ = %s %% уровня K1 [%s; %s], DM p = %s, Δ < 0 в %d из %d лет; «go» 6.1: %s. QLIKE X3 %s против A0g %s." % (
        sb["rule62"], f(sb["delta_pct"], 2), f(sb["lo90"], 2), f(sb["hi90"], 2), f(sb["p_one"], 3), sb["years_neg"], sb["n_years"], "да" if bool(go.loc[go["cond"] == 0, "passed"].iloc[0]) else "нет", f(q("X3", "2018_2021")), f(q("V6:A0g", "2018_2021")))
    out = [first, "", "# E3 — X3 (B1): устойчивая поправка A0g + адаптивная g_a с весом a_t (CUSUM)", "", STATUS, "",
           "Даты %s … %s, n = %d; g_a — QLIKE-гребень на последних %d закрытых парах, λ ∈ {1; 10} — внутренняя валидация (%d закрытых пар), помесячно; a_t = σ((S − 5s)/s), S — CUSUM δ = QLIKE(A0g) − QLIKE(A0g·e^g_a). Холм N = 738 / N_wide = 3181." % (num["first"], num["last"], num["n_dates"], num["win"], num["hold"]), "",
           "## 1. QLIKE по годам и окнам (X3 рядом с A0g, X1, NTP2, cfg8, K2, K3, A2)", "", md([[s] + [f(q(s, w)) for w in YEARS + WINS] for s in sers], ["ряд"] + YEARS + WINS), "",
           "## 2. Δ X3 в % уровня K1 (90 % интервал, блок 20 · 2000 · seed 0), DM лаг 5, Холм N = 738", ""]
    sel = tt[tt["base"].isin(["V6:A0g", "V4:NTP2", "V6:A2", "K2", "K3", "M", "K8"]) & tt["window"].isin(WINS)]
    out += [md([[r.window, r.base, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), f(r.p_holm_N, 3)] for r in sel.itertuples()], ["окно", "база", "Δ %", "lo90", "hi90", "DM p", "p Холм N"]), "",
            "## 3. Пары (правило 6.2; окна разработки — «да»/«нет», 2022+ — «—»)", "",
            md([[r.hyp, r.window, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in pr.itertuples()], ["пара", "окно", "Δ %", "lo90", "hi90", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 4. «Go» 6.1 для X3", "", md([[int(r.cond), r.name, f(r.value), f(r.threshold), "да" if r.passed else "нет"] for r in go.itertuples()], ["№", "условие", "значение", "порог", "выполнено"]), "",
            "## 5. a_t и g_a по годам цели", "", md([[r.year, r.n, f(r.mean_a, 3), f(r.share_a_gt_half, 3), f(r.mean_abs_ga, 4), f(r.mean_a_ga, 4), f(r.share_lam_10, 2)] for r in at.itertuples()], ["год", "дней", "средний a_t", "доля a_t > 0,5", "средний |g_a|", "средний a_t·g_a", "доля λ = 10"]), "",
            "## 6. QLIKE по состояниям (норма / экстрем)", "", md([[r.series, r.window, r.state, r.n, f(r.qlike)] for r in st.itertuples() if r.series in ("X3", "X1", "V6:A0g")], ["ряд", "окно", "состояние", "n", "QLIKE"]), "",
            "## 7. Стресс (QLIKE, разность к лучшему из K2/K3)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3)] for r in t["STRESS"].itertuples() if r.series in ("X3", "X1", "V6:A0g")], ["ряд", "стресс", "n", "QLIKE", "разность"]), "",
            "## 8. MCS 90 %", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples() if r.series in ("X3", "X1", "V6:A0g")], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), ""]
    return "\n".join(out)
