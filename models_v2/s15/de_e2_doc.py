"""E2: `E2_REPORT.md` — генератор из таблиц (числа руками не вписываются; `--check` сверяет файл с `build`). Первая строка — правило 6.2 для X2 против A0g (своя база) и против cfg4."""
from __future__ import annotations

import pandas as pd

from models_v2.s15.de_e1_doc import WINS, YEARS, f, md

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-239/Ж-240, лист E2 (SBER, h = 1; израсходовано 2 конфигурации из 5: X1, X2); окна 2022+ — только оценка, выбора по ним нет."


def _line(name: str, sb: dict, q: dict) -> str:
    return "%s: **%s** — Δ = %s %% уровня K1 [%s; %s], DM p = %s, Δ < 0 в %d из %d лет" % (name, sb["rule62"], f(sb["delta_pct"], 2), f(sb["lo90"], 2), f(sb["hi90"], 2), f(sb["p_one"], 3), sb["years_neg"], sb["n_years"])


def build(t: dict, sb_a0g: dict, sb_cfg4: dict, st: pd.DataFrame, cal: pd.DataFrame, bt: pd.DataFrame, num: dict) -> str:
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    sers = ["X2", "X1", "V4:NTP2", "cfg8", "cfg4", "K2", "K3", "V6:A2", "V6:A0g"]
    q = lambda s, w: m.get((s, w), float("nan"))
    tt, pr, go = t["TESTS"], t["PAIRS"], t["GO"]
    first = "**X2 (хвостовая смесь), 2018–2021, правило 6.2.** %s; %s; «go» 6.1: %s. QLIKE X2 %s против A0g %s, cfg4 %s." % (
        _line("против A0g", sb_a0g, q), _line("против cfg4", sb_cfg4, q), "да" if bool(go.loc[go["cond"] == 0, "passed"].iloc[0]) else "нет", f(q("X2", "2018_2021")), f(q("V6:A0g", "2018_2021")), f(q("cfg4", "2018_2021")))
    out = [first, "", "# E2 — X2 (D2): хвостовая смесь поправок с меткой стресса", "", STATUS, "",
           "Даты %s … %s, n = %d; β ∈ {0,1; 1} — внутренняя валидация (126 закрытых пар), 5 seeds, помесячно; метка стресса — RV(t⁺) > PIT-90 закрытых пар. Холм N = 738 / N_wide = 3181." % (num["first"], num["last"], num["n_dates"]), "",
           "## 1. QLIKE по годам и окнам (X2 рядом с X1, NTP2, cfg8, cfg4, K2, K3, A2, A0g)", "", md([[s] + [f(q(s, w)) for w in YEARS + WINS] for s in sers], ["ряд"] + YEARS + WINS), "",
           "## 2. Δ X2 в % уровня K1 (90 % интервал, блок 20 · 2000 · seed 0), DM лаг 5, Холм N = 738", ""]
    sel = tt[tt["base"].isin(["V6:A0g", "V4:NTP2", "V6:A2", "K2", "K3", "M", "K8"]) & tt["window"].isin(WINS)]
    out += [md([[r.window, r.base, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), f(r.p_holm_N, 3)] for r in sel.itertuples()], ["окно", "база", "Δ %", "lo90", "hi90", "DM p", "p Холм N"]), "",
            "## 3. Пары (правило 6.2; окна разработки — «да»/«нет», 2022+ — «—»)", "",
            md([[r.hyp, r.window, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in pr.itertuples()], ["пара", "окно", "Δ %", "lo90", "hi90", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 4. «Go» 6.1 для X2", "", md([[int(r.cond), r.name, f(r.value), f(r.threshold), "да" if r.passed else "нет"] for r in go.itertuples()], ["№", "условие", "значение", "порог", "выполнено"]), "",
            "## 5. Калибровка p (децили p: средний p против доли дней со стрессом; «итого» — abs_gap = ECE)", "", md([[r.window, r.decile, r.n, f(r.mean_p, 3), f(r.freq, 3), f(r.abs_gap, 3)] for r in cal.itertuples()], ["окно", "дециль", "n", "средний p", "доля стресса", "|разность| / ECE"]), "",
            "## 6. p, c_n, c_s и доля β = 1 по годам цели", "", md([[r.year, r.n, f(r.mean_p, 3), f(r.mean_cn, 3), f(r.mean_cs, 3), f(r.share_beta_1, 2), f(r.stress_freq, 3)] for r in bt.itertuples()], ["год", "дней", "p", "c_n", "c_s", "доля β = 1", "доля стресса"]), "",
            "## 7. QLIKE по состояниям (норма / экстрем)", "", md([[r.series, r.window, r.state, r.n, f(r.qlike)] for r in st.itertuples() if r.series in ("X2", "X1", "cfg8", "V6:A0g")], ["ряд", "окно", "состояние", "n", "QLIKE"]), "",
            "## 8. Стресс (QLIKE, разность к лучшему из K2/K3)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3)] for r in t["STRESS"].itertuples() if r.series in ("X2", "X1", "cfg4", "V6:A0g")], ["ряд", "стресс", "n", "QLIKE", "разность"]), "",
            "## 9. MCS 90 %", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples() if r.series in ("X2", "X1", "cfg4", "V6:A0g")], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), ""]
    return "\n".join(out)
