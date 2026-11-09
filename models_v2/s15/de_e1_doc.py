"""E1: `E1_REPORT.md` — генератор из таблиц (числа не вписываются руками; `--check` сверяет файл с `build`). Первая строка — количественный ответ на «польза состояния» (X1 против NTP2, правило 6.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-239/Ж-240, лист E1 (SBER, h = 1; израсходовано 1 конфигурация из 5: X1); окна 2022+ — только оценка, выбора по ним нет."
YEARS = [str(y) for y in range(2018, 2026)] + ["2026"]
WINS = ["2018_2021", "2019_2021", "2022_2025", "2026"]


def f(x, nd=4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%.*f" % (nd, x))


def md(rows: list, head: list) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def build(t: dict, sb: dict, st: pd.DataFrame, pit: pd.DataFrame, num: dict) -> str:
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    sers = ["X1", "V4:NTP2", "cfg8", "K2", "K3", "V6:A2"]
    q = lambda s, w: m.get((s, w), float("nan"))
    tt, pr, go = t["TESTS"], t["PAIRS"], t["GO"]
    first = "**Польза состояния (X1 против NTP2, 2018–2021, правило 6.2): %s** — Δ = %s %% уровня K1 [%s; %s], DM p = %s, Δ < 0 в %d из %d лет; QLIKE X1 %s против NTP2 %s." % (
        sb["rule62"], f(sb["delta_pct"], 2), f(sb["lo90"], 2), f(sb["hi90"], 2), f(sb["p_one"], 3), sb["years_neg"], sb["n_years"], f(q("X1", "2018_2021")), f(q("V4:NTP2", "2018_2021")))
    out = [first, "", "# E1 — X1 (D1): доверие к бустингу по состоянию «экстремальный M10-бар»", "", STATUS, "",
           "Даты %s … %s, n = %d; экстремальных дней s = 1: %d; дней без баров M10 (состояние «норма»): %d. Холм N = 738 / N_wide = 3181." % (num["first"], num["last"], num["n_dates"], num["n_extreme_days"], num["n_no_bar_days"]), "",
           "## 1. QLIKE по годам и окнам (X1 рядом с NTP2, cfg8, K2, K3, A2)", "", md([[s] + [f(q(s, w)) for w in YEARS + WINS] for s in sers], ["ряд"] + YEARS + WINS), "",
           "## 2. Δ X1 в % уровня K1 (90 % интервал, блок 20 · 2000 · seed 0), DM лаг 5, Холм N = 738", ""]
    sel = tt[tt["base"].isin(["V4:NTP2", "V6:A2", "K2", "K3", "M", "K8"]) & tt["window"].isin(WINS)]
    out += [md([[r.window, r.base, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), f(r.p_holm_N, 3)] for r in sel.itertuples()], ["окно", "база", "Δ %", "lo90", "hi90", "DM p", "p Холм N"]), "",
            "## 3. Пары (правило 6.2; окна разработки — «да»/«нет», 2022+ — «—»)", "",
            md([[r.hyp, r.window, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in pr.itertuples()], ["пара", "окно", "Δ %", "lo90", "hi90", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 4. «Go» 6.1 для X1", "", md([[int(r.cond), r.name, f(r.value), f(r.threshold), "да" if r.passed else "нет"] for r in go.itertuples()], ["№", "условие", "значение", "порог", "выполнено"]), "",
            "## 5. π по состояниям и годам цели", "", md([[r.period, r.state, r.n, f(r.mean_pi_on_days, 3), f(r.mean_pi_process, 3), f(r.mean_alpha, 3)] for r in pit.itertuples()], ["период", "состояние", "дней", "π в дни состояния", "π процесса", "α"]), "",
            "## 6. QLIKE по состояниям (норма / экстрем)", "", md([[r.series, r.window, r.state, r.n, f(r.qlike)] for r in st.itertuples() if r.series in ("X1", "V4:NTP2", "cfg8")], ["ряд", "окно", "состояние", "n", "QLIKE"]), "",
            "## 7. Стресс (QLIKE, разность к лучшему из K2/K3)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3)] for r in t["STRESS"].itertuples() if r.series in ("X1", "V4:NTP2", "cfg8")], ["ряд", "стресс", "n", "QLIKE", "разность"]), "",
            "## 8. MCS 90 %", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples() if r.series in ("X1", "V4:NTP2", "cfg8")], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), ""]
    return "\n".join(out)
