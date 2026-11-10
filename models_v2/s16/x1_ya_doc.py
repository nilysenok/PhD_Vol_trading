"""Y-A: `YA_REPORT.md` — генератор из таблиц (числа не вписываются руками; `--check` сверяет файл с `build`). «Снижение исторической ошибки» (точка, интервал) и «подтверждено» (p Холма N < 0,05) — РАЗДЕЛЬНО."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-243, лист Y-A (SBER, h = 1; израсходовано 2 конфигурации из 4 утверждённых: Y1, Y2)."
PW = ("all", "2018_2021", "2022_2025", "2026", "after")
SERS = ("Y1", "Y2", "X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1")


def f(x, nd=4) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%.*f" % (nd, x))


def md(rows: list, head: list) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def conf(p) -> str:
    """«подтверждено» — только при p Холма (N = 744) < 0,05."""
    return "подтверждено" if isinstance(p, float) and np.isfinite(p) and p < 0.05 else "не доказано"


def build(t: dict, num: dict) -> str:
    pr, mt = t["PAIRS"], t["METRICS"].set_index(["series", "window"])["qlike"]
    sel = lambda y, b, w: pr[(pr["y"] == y) & (pr["base"] == b) & (pr["window"] == w)].iloc[0]
    r62 = t["RULE62"].set_index("y")
    head = ["**%s: правило 6.2 против X1 на 2018–2021 — %s** (Δ QLIKE = %s, DM p = %s, лет Δ<0: %d из %d; p Холма N = %s). Вывод по Холму (N = 744, порог 0,05) против X1 на 2018–2021: %s." % (
        y, r62.at[y, "rule62"], f(r62.at[y, "delta_q"], 6), f(r62.at[y, "p_dm_one"], 3), r62.at[y, "years_neg"], r62.at[y, "n_years"], f(r62.at[y, "p_holm_N"], 3), conf(float(r62.at[y, "p_holm_N"]))) for y in ("Y1", "Y2")]
    out = head + ["", "# Y-A — Y1 и Y2: затухание состояния закрытия (М1) на базе X1", "", STATUS, "",
                  "Даты %s … %s, n = %d; включение с 2018-07-02 (закрытых пар на эту дату: %d, порог %d); до включения F = X1 побитно (%d дней, active = 0). Холм N = 744 / N_wide = 3193." % (num["first"], num["last"], num["n_dates"], num["pairs_at_activation"], num["min_pairs"], num["n_inactive"]), "",
                  "## 1. Признаки дня d (PIT)", "",
                  "Карта слотов (паспорт V7 `v7_0_seq`: слот = (старт бара − 10:00)/10 минут): часть закрытия — старты %s … %s = слоты %s; всего слотов 53 (слот 52 — старт 18:40, в часть закрытия не входит, но входит в n_d и RV_d)." % (num["close_from"], num["close_to"], num["close_slots"]),
                  "Дней с s_d = 0 по правилу «нет части закрытия» (n^last = 0 или RV^last = 0): %d из %d. Пар с e_d = 1 (экстремальный бар): %d." % (num["n_zero_days"], num["n_dates"], num["n_extreme_days"]), "",
                  "## 2. QLIKE по окнам", "", md([[s] + [f(mt.get((s, w))) for w in PW] for s in SERS], ["ряд"] + list(PW)), "",
                  "## 3. Снижение исторической ошибки (точка и 90 % интервал; Δ в % QLIKE базы) — это ещё не статистический вывод", ""]
    for y in ("Y1", "Y2"):
        out += ["### %s" % y, "", md([[b, w, f(sel(y, b, w)["delta_pct"], 2), f(sel(y, b, w)["lo90"], 2), f(sel(y, b, w)["hi90"], 2)] for b in ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1") for w in PW], ["база", "окно", "Δ %", "lo90", "hi90"]), ""]
    out += ["## 4. Статистический вывод (порог: p Холма N = 744 < 0,05); DM лаг 5, односторонний «Y лучше базы»", ""]
    for y in ("Y1", "Y2"):
        out += ["### %s" % y, "", md([[b, w, f(sel(y, b, w)["p_one"], 3), f(sel(y, b, w)["p_holm_N"], 3), f(sel(y, b, w)["p_holm_Nwide"], 3), conf(float(sel(y, b, w)["p_holm_N"]))] for b in ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1") for w in PW], ["база", "окно", "DM p", "p Холм N", "p Холм N_wide", "вывод по Холму"]), ""]
    out += ["## 5. Y1 против Y2 (ценность затухания; X = Y1, база = Y2)", "", md([[w, f(sel("Y1", "Y2", w)["delta_pct"], 2), f(sel("Y1", "Y2", w)["lo90"], 2), f(sel("Y1", "Y2", w)["hi90"], 2), f(sel("Y1", "Y2", w)["p_one"], 3), f(sel("Y1", "Y2", w)["p_holm_N"], 3), sel("Y1", "Y2", w)["years_neg"]] for w in pr["window"].unique()], ["окно", "Δ %", "lo90", "hi90", "DM p", "p Холм N", "лет Δ<0"]), "",
            "## 6. Правило 6.2 против X1 (2018–2021: Δ < 0, DM p < 0,05, ≥ 3 из 4 лет)", "", md([[r.y, f(r.delta_q, 6), f(r.p_dm_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in t["RULE62"].itertuples()], ["ряд", "Δ QLIKE", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 7. По годам: Δ к X1 и K2 (в % QLIKE базы)", "", md([[r.y, r.base, r.year, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2)] for r in t["YEARS"].itertuples()], ["ряд", "база", "год t⁺", "n", "QLIKE Y", "QLIKE базы", "Δ %"]), "",
            "## 8. Клетки остатка X1 (описательно; p без поправки)", "", md([[r.cell, r.y, r.base, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_q, 5), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3)] for r in t["CELLS"].itertuples()], ["клетка", "ряд", "база", "n", "QLIKE", "QLIKE базы", "Δ QLIKE", "Δ %", "lo90", "hi90", "DM p"]), "",
            "## 9. «Go» 6.1 (база min(K2, K3, M, K8, NTP2, A2, X1))", "", md([[r.y, int(r.cond), r.name, f(r.value, 5), f(r.threshold, 4), "да" if r.passed else "нет"] for r in t["GO"].itertuples()], ["ряд", "№", "условие", "значение", "порог", "выполнено"]), "",
            "## 10. MCS 90 %", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples()], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), "",
            "## 11. Стресс (без 1 % крайних, без 2018)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3, 5)] for r in t["STRESS"].itertuples() if r.series in ("X1", "Y1", "Y2", "K2")], ["ряд", "стресс", "n", "QLIKE", "разность к лучшему из K2/K3"]), "",
            "## 12. Параметры по годам (среднее по месяцам переоценки)", ""]
    pa = t["PARAMS"].assign(year=lambda x: x["month"].str[:4]).groupby(["y", "year"])[["a_n", "a_e", "kappa", "lam"]].mean().reset_index()
    out += [md([[r.y, r.year, f(r.a_n), f(r.a_e), f(r.kappa), f(r.lam, 3)] for r in pa.itertuples()], ["ряд", "год", "a_n", "a_e", "κ", "λ"]), "",
            "## 13. Условие для Y5/Y6 (только отчёт; не запуск): «правило 6.2 против X1 = да у Y1 или у Y3»", "",
            "У Y1: %s. Y3 в этом листе не считалась (лист Y-B). Условие, ограниченное Y1: %s. Y5/Y6 не запускаются без отдельного разрешения владельца (Ж-243)." % (r62.at["Y1", "rule62"], "выполнено" if r62.at["Y1", "rule62"] == "да" else "не выполнено"), ""]
    return "\n".join(out)
