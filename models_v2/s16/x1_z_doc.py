"""Этап Z: `Z_REPORT.md` — генератор из таблиц (числа в текст не вписываются; все цифры — из таблиц `t` и словаря запуска `num`; `--check` сверяет файл с `build`). «Снижение исторической ошибки» (точка, интервал),
«устойчивость» (годы, K3) и «подтверждено» (правило 6.2 и p Холма N < 0,05) — РАЗДЕЛЬНО."""
from __future__ import annotations

import pandas as pd

from models_v2.s16.x1_ya_doc import PW, conf, f, md
from models_v2.s16.x1_z_eval import CONTRASTS, STRESS_SERIES, ZBASES, ZS

STATUS = "разработка, историческая пошаговая проверка, не независимая; ZB, ZC — после просмотра Y1–Y4 (Ж-246); первая цифра — потолок"
SERS = ZS + ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "Y2", "Y3")
HYPS = ("ZA", "ZB", "ZC")
DIAG = "Критерии K1–K5 — решение разработки о продолжении гипотезы, НЕ подтверждение; они используют уже просмотренные 2022+."


def sel(pr, y, b, w):
    return pr[(pr["y"] == y) & (pr["base"] == b) & (pr["window"] == w)].iloc[0]


def head(t: dict, num: dict) -> list:
    cr, r62 = t["CRIT"], t["RULE62"].set_index("y")
    out = ["Статус: " + STATUS + "."]
    for y in HYPS:
        c = cr[(cr["y"] == y) & (cr["crit"] == "итог")].iloc[0]
        out.append("**%s: K1–K5 — пройдено %d из 5 → %s** (решение разработки). Правило 6.2 против X1 на 2018–2021: %s (Δ QLIKE = %s, DM p = %s, лет Δ<0: %d из %d; p Холма N = %s, N = %d) → %s." % (
            y, int(c["value"]), c["name"], r62.at[y, "rule62"], f(r62.at[y, "delta_q"], 6), f(r62.at[y, "p_dm_one"], 3), r62.at[y, "years_neg"], r62.at[y, "n_years"], f(r62.at[y, "p_holm_N"], 3), num["n_holm"], conf(float(r62.at[y, "p_holm_N"]))))
    return out + ["", DIAG, ""]


def proto(num: dict) -> list:
    return ["# Этап Z — ZA, ZB, ZB0, ZC, ZCp: специализированная коррекция X1 (SBER, h = 1)", "",
            "Реестр (`ALL_TRIALS.csv`) записан до прогноза: %s. Конфигураций израсходовано %d (ZA, ZB, ZB0, ZC, ZCp); Y5, Y6 не запускались. ZB и ZC предложены после просмотра Y1–Y4." % (num["registered_at"], num["configs_spent"]),
            "Даты %s … %s, n = %d; включение с 2018-07-02 (закрытых пар на эту дату: %d, порог %d; из них с e_d = 1: %d, порог ZB/ZB0 %d). До включения F = X1 побитно. Дней без включения (active = 0): %s." % (
                num["first"], num["last"], num["n_dates"], num["pairs_at_activation"], num["min_pairs"], num["n_ext_at_activation"], num["min_ext"], num["n_inactive"]),
            "Дней с e_d = 1: %d; дней с вырожденным RV (x_d = v_d = 0): %d. Решётки: λ %s; ω (ZA) %s; V = %d, V_e = %d. Холм N = %d / N_wide = %d." % (
                num["n_extreme_days"], num["n_degenerate_days"], num["lams"], num["omegas"], num["v_full"], num["v_ext"], num["n_holm"], num["n_wide"]), ""]


def qlike(t: dict) -> list:
    mt = t["METRICS"].set_index(["series", "window"])["qlike"]
    return ["## 1. QLIKE по окнам", "", md([[s] + [f(mt.get((s, w))) for w in PW] for s in SERS], ["ряд"] + list(PW)), ""]


def delta(t: dict) -> list:
    pr, out = t["PAIRS"], ["## 2. Снижение исторической ошибки (точка, 90 % интервал; Δ в % QLIKE базы) — ещё не статистический вывод", ""]
    for y in ZS:
        out += ["### %s" % y, "", md([[b, w, f(sel(pr, y, b, w)["delta_pct"], 2), f(sel(pr, y, b, w)["lo90"], 2), f(sel(pr, y, b, w)["hi90"], 2)] for b in ZBASES for w in PW], ["база", "окно", "Δ %", "lo90", "hi90"]), ""]
    return out


def stat(t: dict, num: dict) -> list:
    pr, out = t["PAIRS"], ["## 3. Статистический вывод (порог: p Холма N = %d < 0,05); DM лаг 5, односторонний «ряд лучше базы»" % num["n_holm"], ""]
    for y in ZS:
        out += ["### %s" % y, "", md([[b, w, f(sel(pr, y, b, w)["p_one"], 3), f(sel(pr, y, b, w)["p_holm_N"], 3), f(sel(pr, y, b, w)["p_holm_Nwide"], 3), conf(float(sel(pr, y, b, w)["p_holm_N"]))] for b in ZBASES for w in PW],
                         ["база", "окно", "DM p", "p Холм N", "p Холм N_wide", "вывод по Холму"]), ""]
    return out


def contrasts(t: dict) -> list:
    pr = t["PAIRS"]
    rows = [[y, b, w, f(sel(pr, y, b, w)["delta_pct"], 2), f(sel(pr, y, b, w)["lo90"], 2), f(sel(pr, y, b, w)["hi90"], 2), f(sel(pr, y, b, w)["p_one"], 3), f(sel(pr, y, b, w)["p_holm_N"], 3), sel(pr, y, b, w)["years_neg"]] for y, b in CONTRASTS for w in PW]
    return ["## 4. Контрасты: ZB против ZB0, ZC против ZCp, ZA против ZB против ZC (X = первый, база = второй)", "", md(rows, ["ряд", "база", "окно", "Δ %", "lo90", "hi90", "DM p", "p Холм N", "лет Δ<0"]), ""]


def tables(t: dict) -> list:
    r62 = t["RULE62"]
    return ["## 5. Правило 6.2 против X1 (2018–2021: Δ < 0, DM p < 0,05, ≥ 3 из 4 лет)", "", md([[r.y, f(r.delta_q, 6), f(r.p_dm_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in r62.itertuples()], ["ряд", "Δ QLIKE", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 6. По годам: Δ к X1 и K2 (в % QLIKE базы)", "", md([[r.y, r.base, r.year, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2)] for r in t["YEARS"].itertuples()], ["ряд", "база", "год t⁺", "n", "QLIKE", "QLIKE базы", "Δ %"]), "",
            "## 7. Состояния E (e_d = 1) и N (e_d = 0): Δ к X1 (в % QLIKE X1 состояния), средние g и затвора", "",
            md([[r.y, r.state, r.window, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.mean_g, 4), f(r.mean_gate, 3)] for r in t["STATES"].itertuples()], ["ряд", "состояние", "окно", "n", "QLIKE", "QLIKE X1", "Δ %", "lo90", "hi90", "средний g", "средний затвор"]), "",
            "## 8. Клетки остатка X1 (описательно; p без поправки)", "", md([[r.cell, r.y, r.base, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_q, 5), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3)] for r in t["CELLS"].itertuples()], ["клетка", "ряд", "база", "n", "QLIKE", "QLIKE базы", "Δ QLIKE", "Δ %", "lo90", "hi90", "DM p"]), ""]


def crit(t: dict) -> list:
    rows = [[r.y, r.crit, r.name, f(r.value, 3), f(r.threshold, 2), "да" if r.passed else "нет"] for r in t["CRIT"].itertuples()]
    return ["## 9. Критерии K1–K5 (записаны до счёта; ZB0 и ZCp критериев не получают)", "", md(rows, ["гипотеза", "критерий", "условие / итог", "значение", "порог", "выполнено"]), "", DIAG, ""]


def rest(t: dict) -> list:
    out = ["## 10. «Go» 6.1 (база min(K2, K3, M, K8, NTP2, A2, X1))", "", md([[r.y, int(r.cond), r.name, f(r.value, 5), f(r.threshold, 4), "да" if r.passed else "нет"] for r in t["GO"].itertuples()], ["ряд", "№", "условие", "значение", "порог", "выполнено"]), "",
           "## 11. MCS 90 % (все ряды)", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples()], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), "",
           "## 12. Стресс (без 1 % крайних, без 2018)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3, 5)] for r in t["STRESS"].itertuples() if r.series in STRESS_SERIES], ["ряд", "стресс", "n", "QLIKE", "разность к лучшему из K2/K3"]), "",
           "## 13. Параметры по годам (среднее по месяцам переоценки)", ""]
    pa = t["PARAMS"].assign(year=lambda x: x["month"].str[:4]).groupby(["y", "year"])[["lam", "omega", "b", "a", "gam0", "gam1", "gam2", "gam3", "gam4", "n_ext"]].mean().reset_index()
    return out + [md([[r.y, r.year] + [f(getattr(r, c), 3) for c in ("lam", "omega", "b", "a", "gam0", "gam1", "gam2", "gam3", "gam4", "n_ext")] for r in pa.itertuples()], ["ряд", "год", "λ", "ω", "b", "a", "γ0", "γ1 (e)", "γ2 (x)", "γ3 (s)", "γ4 (v)", "закрытых e-пар"]), ""]


def notes(num: dict) -> list:
    return ["## 14. Трактовки и отступления", ""] + ["- " + x for x in num["notes"]] + [""]


def build(t: dict, num: dict) -> str:
    return "\n".join(head(t, num) + proto(num) + qlike(t) + delta(t) + stat(t, num) + contrasts(t) + tables(t) + crit(t) + rest(t) + notes(num))
