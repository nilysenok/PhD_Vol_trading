"""E4: `E4_REPORT.md` — генератор из таблиц (числа руками не вписываются; `--check` сверяет файл с `build`). Первая строка — правило 6.2 для X4 против ансамбля без затвора (польза затвора) и против A0g, и «go»."""
from __future__ import annotations

import pandas as pd

from models_v2.s15.de_e1_doc import WINS, YEARS, f, md

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-239/Ж-240, лист E4 (SBER, h = 1; израсходовано 4 конфигурации из 5: X1, X2, X3, X4); окна 2022+ — только оценка, выбора по ним нет. ENS (ансамбль без затвора, α ≡ 1) — справочный ряд, не конфигурация."


def _one(sb: dict, name: str) -> str:
    return "%s: **%s** — Δ = %s %% уровня K1 [%s; %s], DM p = %s, Δ < 0 в %d из %d лет" % (name, sb["rule62"], f(sb["delta_pct"], 2), f(sb["lo90"], 2), f(sb["hi90"], 2), f(sb["p_one"], 3), sb["years_neg"], sb["n_years"])


def build(t: dict, sb_ens: dict, sb_a0g: dict, st: pd.DataFrame, at: pd.DataFrame, num: dict) -> str:
    m = t["METRICS"].set_index(["series", "window"])["qlike"]
    sers = ["X4", "ENS", "V6:A0g", "X1", "V4:NTP2", "cfg8", "K2", "K3", "V6:A2"]
    q = lambda s, w: m.get((s, w), float("nan"))
    tt, pr, go = t["TESTS"], t["PAIRS"], t["GO"]
    first = "**X4 (затвор по неопределённости), 2018–2021, правило 6.2** — %s; %s; «go» 6.1: %s. QLIKE X4 %s, ENS %s, A0g %s." % (
        _one(sb_ens, "против ансамбля без затвора"), _one(sb_a0g, "против A0g"), "да" if bool(go.loc[go["cond"] == 0, "passed"].iloc[0]) else "нет", f(q("X4", "2018_2021")), f(q("ENS", "2018_2021")), f(q("V6:A0g", "2018_2021")))
    out = [first, "", "# E4 — X4 (C1): затвор по неопределённости ансамбля из 5 QLIKE-GBM", "", STATUS, "",
           "Даты %s … %s, n = %d; окна ансамбля: расширяющееся и %s закрытых пар; затвор α = exp(−κ·sd_r/s̄), κ ∈ %s — внутренняя валидация (%d закрытых пар), помесячная переоценка GBM. Холм N = 738 / N_wide = 3181." % (
               num["first"], num["last"], num["n_dates"], " / ".join(str(w) for w in num["windows"] if w), num["kappas"], num["val"]), "",
           "## 1. QLIKE по годам и окнам (X4 рядом с ENS, A0g, X1, NTP2, cfg8, K2, K3, A2)", "", md([[s] + [f(q(s, w)) for w in YEARS + WINS] for s in sers], ["ряд"] + YEARS + WINS), "",
           "## 2. Δ X4 в % уровня K1 (90 % интервал, блок 20 · 2000 · seed 0), DM лаг 5, Холм N = 738", ""]
    sel = tt[tt["base"].isin(["V6:A0g", "V4:NTP2", "V6:A2", "K2", "K3", "M", "K8"]) & tt["window"].isin(WINS)]
    out += [md([[r.window, r.base, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), f(r.p_holm_N, 3)] for r in sel.itertuples()], ["окно", "база", "Δ %", "lo90", "hi90", "DM p", "p Холм N"]), "",
            "## 3. Пары (правило 6.2; окна разработки — «да»/«нет», 2022+ — «—»)", "",
            md([[r.hyp, r.window, f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in pr.itertuples()], ["пара", "окно", "Δ %", "lo90", "hi90", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 4. «Go» 6.1 для X4", "", md([[int(r.cond), r.name, f(r.value), f(r.threshold), "да" if r.passed else "нет"] for r in go.itertuples()], ["№", "условие", "значение", "порог", "выполнено"]), "",
            "## 5. Затвор α по годам цели", "", md([[r.year, r.n, f(r.share_act, 3), f(r.mean_alpha, 3), f(r.share_alpha_lt_half, 3), f(r.mean_sd, 4), f(r.mean_abs_gbar, 4), f(r.share_kappa_0_5, 2), f(r.share_kappa_1_0, 2), f(r.share_kappa_2_0, 2)] for r in at.itertuples()],
                                                ["год", "дней", "доля act", "средний α", "доля α < 0,5", "средний sd_r", "средний |ḡ|", "κ = 0,5", "κ = 1", "κ = 2"]), "",
            "## 6. QLIKE по состояниям (норма / экстрем)", "", md([[r.series, r.window, r.state, r.n, f(r.qlike)] for r in st.itertuples() if r.series in ("X4", "ENS", "V6:A0g")], ["ряд", "окно", "состояние", "n", "QLIKE"]), "",
            "## 7. Стресс (QLIKE, разность к лучшему из K2/K3)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3)] for r in t["STRESS"].itertuples() if r.series in ("X4", "ENS", "V6:A0g")], ["ряд", "стресс", "n", "QLIKE", "разность"]), "",
            "## 8. MCS 90 %", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples() if r.series in ("X4", "ENS", "V6:A0g")], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), ""]
    return "\n".join(out)
