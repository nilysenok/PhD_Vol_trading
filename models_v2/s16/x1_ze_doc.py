"""Лист ZE: `ZE_REPORT.md` — генератор из таблиц (числа в текст не вписываются; все цифры — из таблиц `t` и словаря запуска `num`; `--check` сверяет файл с `build`).
«Снижение исторической ошибки» (точка, интервал), «устойчивость» (годы, K3) и «подтверждено» (правило 6.2 и p Холма N = 751 < 0,05) — РАЗДЕЛЬНО."""
from __future__ import annotations

from models_v2.s16.x1_ya_doc import PW, conf, f, md
from models_v2.s16.x1_ze_crit import BASES, DWIN, ZS

STATUS = "разработка, историческая пошаговая проверка, не независимая; ZE — после просмотра Y1–Y4, Z, ZD"
DIAG = "Критерии K1–K6 — решение разработки о продолжении гипотезы, НЕ подтверждение; они используют уже просмотренные 2022+. Окна «без 2022» и 2023–2025 — диагностика, не критерий."
ALLW = PW + DWIN


def sel(pr, y, b, w):
    return pr[(pr["y"] == y) & (pr["base"] == b) & (pr["window"] == w)].iloc[0]


def head(t: dict, num: dict) -> list:
    c, r = t["CRIT"].query("crit == 'итог'").iloc[0], t["RULE62"].set_index("y").loc["ZE"]
    return [STATUS + "; первая цифра — потолок.", "",
            "**ZE: K1–K6 — пройдено %d из 6 → %s** (решение разработки). Правило 6.2 против X1 на 2018–2021: %s (Δ QLIKE = %s, DM p = %s, лет Δ<0: %d из %d; p Холма N = %s, N = %d) → %s." % (
                int(c["value"]), c["name"], r["rule62"], f(r["delta_q"], 6), f(r["p_dm_one"], 3), r["years_neg"], r["n_years"], f(r["p_holm_N"], 3), num["n_holm"], conf(float(r["p_holm_N"]))), "", DIAG, ""]


def proto(num: dict) -> list:
    p = num["precondition"]
    return ["# Лист ZE — ZE, ZEp: частичная подстройка X1 к разрыву G_d = ln RV_d − ln X1_d (SBER, h = 1)", "",
            "Предусловие (до реестра): на %d парах e_d = 1 с t⁺ ≤ 2021-12-31 corr(G, v_d) = %s, corr(G, s_d) = %s (справочно corr(G, x_d) = %s), максимум |corr| = %s при пороге %s → %s; записано %s." % (
                p["n"], f(p["corr_G_v"], 3), f(p["corr_G_s"], 3), f(p["corr_G_x_info"], 3), f(p["max_abs"], 3), p["limit"], "условие выполнено" if p["ok"] else "условие ложно", num["precondition_at"]),
            "Реестр (`ALL_TRIALS.csv`) записан до прогноза: %s. Конфигураций израсходовано %d (ZE, ZEp); Y5 — не запускалось (план ZE, разд. 1: формула не определена полностью), Y6 условна, не запускалась." % (num["registered_at"], num["configs_spent"]),
            "Даты %s … %s, n = %d; включение с 2018-07-02 (закрытых пар: %d, порог %d; с e_d = 1: %d, порог %d). Дней без включения (active = 0): %s. Дней с e_d = 1: %d; дней с вырожденным RV_d или X1_d (G = 0): %d." % (
                num["first"], num["last"], num["n_dates"], num["pairs_at_activation"], num["min_pairs"], num["n_ext_at_activation"], num["min_ext"], num["n_inactive"], num["n_extreme_days"], num["n_degenerate_days"]),
            "Решётка λ %s; V_e = %d; Холм N = %d / N_wide = %d." % (num["lams"], num["v_ext"], num["n_holm"], num["n_wide"]), ""]


def qlike(t: dict) -> list:
    pr = t["PAIRS"]
    mt = t["METRICS"].set_index(["series", "window"])["qlike"]
    sers = ZS + ("ZB", "ZB0", "ZA", "X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "Y2", "Y3")
    dg = {(r.y, r.window): r.y_qlike for r in t["DIAG"].itertuples()} | {(r.base, r.window): r.base_qlike for r in t["DIAG"].itertuples()}
    return ["## 1. QLIKE по окнам", "", md([[s] + [f(mt.get((s, w))) for w in PW] + [f(dg.get((s, w))) for w in DWIN] for s in sers], ["ряд"] + list(PW) + list(DWIN)), "",
            "## 2. Снижение исторической ошибки (точка, 90 % интервал; Δ в % QLIKE базы) — ещё не статистический вывод", ""] + [x for y in ZS for x in (
                "### %s" % y, "", md([[b, w, f(sel(pr, y, b, w)["delta_pct"], 2), f(sel(pr, y, b, w)["lo90"], 2), f(sel(pr, y, b, w)["hi90"], 2)] for b in BASES for w in PW], ["база", "окно", "Δ %", "lo90", "hi90"]), "")]


def stat(t: dict, num: dict) -> list:
    pr = t["PAIRS"]
    rows = lambda y, bs: [[b, w, f(sel(pr, y, b, w)["p_one"], 3), f(sel(pr, y, b, w)["p_holm_N"], 3), f(sel(pr, y, b, w)["p_holm_Nwide"], 3), conf(float(sel(pr, y, b, w)["p_holm_N"]))] for b in bs for w in PW]
    out = ["## 3. Статистический вывод (порог: p Холма N = %d < 0,05); DM лаг 5, односторонний «ряд лучше базы»" % num["n_holm"], ""]
    for y, bs in (("ZE", BASES + ("ZEp",)), ("ZEp", BASES)):
        out += ["### %s" % y, "", md(rows(y, bs), ["база", "окно", "DM p", "p Холм N", "p Холм N_wide", "вывод по Холму"]), ""]
    return out


def tables(t: dict) -> list:
    r62, dg = t["RULE62"], t["DIAG"]
    return ["## 4. Правило 6.2 против X1 (2018–2021: Δ < 0, DM p < 0,05, ≥ 3 из 4 лет)", "", md([[r.y, f(r.delta_q, 6), f(r.p_dm_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in r62.itertuples()], ["ряд", "Δ QLIKE", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 5. По годам: Δ к X1, K2 и ZB (в % QLIKE базы)", "", md([[r.y, r.base, r.year, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2)] for r in t["YEARS"].itertuples()], ["ряд", "база", "год t⁺", "n", "QLIKE", "QLIKE базы", "Δ %"]), "",
            "## 6. Состояния E (e_d = 1) и N (e_d = 0): Δ к X1 и к ZB (в % QLIKE базы состояния), средний g", "",
            md([[r.y, r.state, r.window, r.base, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.mean_g, 4)] for r in t["STATES"].itertuples()], ["ряд", "состояние", "окно", "база", "n", "QLIKE", "QLIKE базы", "Δ %", "lo90", "hi90", "средний g"]), "",
            "## 7. Диагностика (не критерий): окна «без 2022» и 2023–2025", "", md([[r.y, r.base, r.window, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3)] for r in dg.itertuples()], ["ряд", "база", "окно", "n", "QLIKE", "QLIKE базы", "Δ %", "lo90", "hi90", "DM p"]), ""]


def crit(t: dict) -> list:
    rows = [[r.y, r.crit, r.name, f(r.value, 3), f(r.threshold, 2), "да" if r.passed else "нет"] for r in t["CRIT"].itertuples()]
    return ["## 8. Критерии K1–K6 по ZE (записаны до счёта; ZEp критериев не получает)", "", md(rows, ["гипотеза", "критерий", "условие / итог", "значение", "порог", "выполнено"]), "", DIAG, ""]


def rest(t: dict) -> list:
    pa = t["PARAMS"].assign(year=lambda x: x["month"].str[:4]).groupby(["y", "year"])[["lam", "b", "a", "c", "n_ext"]].mean().reset_index()
    return ["## 9. MCS 90 % (все ряды)", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples()], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), "",
            "## 10. Стресс (без 1 % крайних, без 2018)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3, 5)] for r in t["STRESS"].itertuples()], ["ряд", "стресс", "n", "QLIKE", "разность к лучшему из K2/K3"]), "",
            "## 11. Выученные параметры по годам (среднее по месяцам переоценки)", "", md([[r.y, r.year] + [f(getattr(r, c), 3) for c in ("lam", "b", "a", "c", "n_ext")] for r in pa.itertuples()], ["ряд", "год", "λ", "b", "a", "c", "закрытых e-пар"]), ""]


def notes(num: dict) -> list:
    return ["## 12. Трактовки и отступления", ""] + ["- " + x for x in num["notes"]] + [""]


def build(t: dict, num: dict) -> str:
    return "\n".join(head(t, num) + proto(num) + qlike(t) + stat(t, num) + tables(t) + crit(t) + rest(t) + notes(num))
