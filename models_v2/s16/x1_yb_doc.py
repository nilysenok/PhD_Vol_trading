"""Y-B: `YB_REPORT.md` — генератор из таблиц (числа не вписываются руками; `--check` сверяет файл с `build`). «Снижение исторической ошибки» (точка, интервал) и «подтверждено» (p Холма N < 0,05) — РАЗДЕЛЬНО."""
from __future__ import annotations

import pandas as pd

from models_v2.s16.x1_ya_doc import PW, conf, f, md

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-243, лист Y-B (SBER, h = 1; израсходовано 4 конфигурации из 4 утверждённых: Y1, Y2 — лист Y-A, Y3, Y4 — этот лист)."
SERS = ("Y3", "Y4", "Y1", "Y2", "X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1")
BS = ("X1", "K2", "K3", "V4:NTP2", "V7:cfg8", "V6:A2", "V7:cfg6", "K1", "Y1", "Y2")
WARN = "Y5 и Y6 не запускаются без отдельного разрешения владельца (Ж-243)."


def build(t: dict, num: dict) -> str:
    pr, mt = t["PAIRS"], t["METRICS"].set_index(["series", "window"])["qlike"]
    sel = lambda y, b, w: pr[(pr["y"] == y) & (pr["base"] == b) & (pr["window"] == w)].iloc[0]
    r62, cd = t["RULE62"].set_index("y"), t["CONDITION"].set_index("item")["rule62"]
    head = ["**%s: правило 6.2 против X1 на 2018–2021 — %s** (Δ QLIKE = %s, DM p = %s, лет Δ<0: %d из %d; p Холма N = %s). Вывод по Холму (N = 744, порог 0,05) против X1 на 2018–2021: %s." % (
        y, r62.at[y, "rule62"], f(r62.at[y, "delta_q"], 6), f(r62.at[y, "p_dm_one"], 3), r62.at[y, "years_neg"], r62.at[y, "n_years"], f(r62.at[y, "p_holm_N"], 3), conf(float(r62.at[y, "p_holm_N"]))) for y in ("Y3", "Y4")]
    head.append("**Условие Y5/Y6 («правило 6.2 против X1 = да у Y1 или у Y3»): %s** (Y1: %s, Y3: %s). %s" % ("ВЫПОЛНЕНО" if cd["условие Y5/Y6"] == "да" else "НЕ ВЫПОЛНЕНО", cd["Y1"], cd["Y3"], WARN))
    out = head + ["", "# Y-B — Y3 и Y4: обучаемая агрегация M10 под остаток X1 (М2)", "", STATUS, "",
                  "Даты %s … %s, n = %d; включение с 2018-07-02 (закрытых пар на эту дату: %d, порог %d); до включения F = X1 побитно (%d дней, active = 0). Холм N = 744 / N_wide = 3193. Y3: seeds %s, итог F = exp(среднее ln F_s)." % (
                      num["first"], num["last"], num["n_dates"], num["pairs_at_activation"], num["min_pairs"], num["n_inactive"], num["seeds"]), "",
                  "## 1. Признаки (PIT) и проверки", "",
                  "- `lvol` склада M10 V7 — СЫРОЙ ln объёма бара (паспорт `v7_0_seq.bar_table`: `np.log(volume)`; exp(lvol) — целые объёмы %s…%s); канал `r` = 100·r. Медиана объёма слота — по 20 прошлым сессиям < d, считается по сырому объёму." % (num["vol_min"], num["vol_max"]),
                  "- φ: положение k/52; ln max(|r|; 1e-8) минус медиана слота за 20 прошлых; sign r; ln(объём / медиана объёма слота за 20 прошлых). Присутствующих баров слота без истории (φ2 = φ4 = 0) за весь склад: %d." % num["n_nohist"],
                  "- Y4: профиль p̄_k — среднее доли r²_k/RV по 250 прошлым сессиям < d с баром слота; строк без профиля (w ≡ 1): %d; дней с RV_d = 0: %d; максимум |среднее w − 1| (Y4): %.2e." % (num["n_prof_bad"], num["n_rv_zero"], num["w4_dev"]),
                  "- Тождества: w ≡ 1 ⇒ B_d = RV_d побитно; среднее w = 1 по присутствующим барам (тесты `test_v2_x1_yb_*`).", "",
                  "## 2. QLIKE по окнам", "", md([[s] + [f(mt.get((s, w))) for w in PW] for s in SERS], ["ряд"] + list(PW)), "",
                  "## 3. Снижение исторической ошибки (точка и 90 % интервал; Δ в % QLIKE базы) — это ещё не статистический вывод", ""]
    for y in ("Y3", "Y4"):
        out += ["### %s" % y, "", md([[b, w, f(sel(y, b, w)["delta_pct"], 2), f(sel(y, b, w)["lo90"], 2), f(sel(y, b, w)["hi90"], 2)] for b in BS for w in PW], ["база", "окно", "Δ %", "lo90", "hi90"]), ""]
    out += ["## 4. Статистический вывод (порог: p Холма N = 744 < 0,05); DM лаг 5, односторонний «Y лучше базы»", ""]
    for y in ("Y3", "Y4"):
        out += ["### %s" % y, "", md([[b, w, f(sel(y, b, w)["p_one"], 3), f(sel(y, b, w)["p_holm_N"], 3), f(sel(y, b, w)["p_holm_Nwide"], 3), conf(float(sel(y, b, w)["p_holm_N"]))] for b in BS for w in PW], ["база", "окно", "DM p", "p Холм N", "p Холм N_wide", "вывод по Холму"]), ""]
    out += ["## 5. Y3 против Y4 (ценность обучения весов; X = Y3, база = Y4)", "", md([[w, f(sel("Y3", "Y4", w)["delta_pct"], 2), f(sel("Y3", "Y4", w)["lo90"], 2), f(sel("Y3", "Y4", w)["hi90"], 2), f(sel("Y3", "Y4", w)["p_one"], 3), f(sel("Y3", "Y4", w)["p_holm_N"], 3), sel("Y3", "Y4", w)["years_neg"]] for w in pr["window"].unique()], ["окно", "Δ %", "lo90", "hi90", "DM p", "p Холм N", "лет Δ<0"]), "",
            "## 6. Правило 6.2 против X1 (2018–2021: Δ < 0, DM p < 0,05, ≥ 3 из 4 лет)", "", md([[r.y, f(r.delta_q, 6), f(r.p_dm_one, 3), r.years_neg, r.rule62, f(r.p_holm_N, 3)] for r in t["RULE62"].itertuples()], ["ряд", "Δ QLIKE", "DM p", "лет Δ<0", "6.2", "p Холм N"]), "",
            "## 7. По годам: Δ к X1 и K2 (в % QLIKE базы)", "", md([[r.y, r.base, r.year, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_pct, 2)] for r in t["YEARS"].itertuples()], ["ряд", "база", "год t⁺", "n", "QLIKE Y", "QLIKE базы", "Δ %"]), "",
            "## 8. Клетки остатка X1 (описательно; p без поправки)", "", md([[r.cell, r.y, r.base, r.n, f(r.y_qlike), f(r.base_qlike), f(r.delta_q, 5), f(r.delta_pct, 2), f(r.lo90, 2), f(r.hi90, 2), f(r.p_one, 3)] for r in t["CELLS"].itertuples()], ["клетка", "ряд", "база", "n", "QLIKE", "QLIKE базы", "Δ QLIKE", "Δ %", "lo90", "hi90", "DM p"]), "",
            "## 9. «Go» 6.1 (база min(K2, K3, M, K8, NTP2, A2, X1))", "", md([[r.y, int(r.cond), r.name, f(r.value, 5), f(r.threshold, 4), "да" if r.passed else "нет"] for r in t["GO"].itertuples()], ["ряд", "№", "условие", "значение", "порог", "выполнено"]), "",
            "## 10. MCS 90 % (с Y1–Y4 вместе)", "", md([[r.window, r.series, f(r.mean_qlike), f(r.p, 3), "да" if r.in_mcs else "нет"] for r in t["MCS"].itertuples()], ["окно", "ряд", "QLIKE", "p MCS", "в MCS"]), "",
            "## 11. Стресс (без 1 % крайних, без 2018)", "", md([[r.series, r.stress, r.n, f(r.qlike), f(r.gap_best_k2k3, 5)] for r in t["STRESS"].itertuples() if r.series in ("X1", "Y1", "Y2", "Y3", "Y4", "K2")], ["ряд", "стресс", "n", "QLIKE", "разность к лучшему из K2/K3"]), "",
            "## 12. Параметры по годам (среднее по месяцам переоценки; Y3 — среднее по seeds)", ""]
    pa = t["PARAMS"].assign(year=lambda x: x["month"].str[:4]).groupby(["y", "year"])[["theta", "g1", "g2", "g3", "g4", "lam"]].mean().reset_index()
    out += [md([[r.y, r.year, f(r.theta), f(r.g1), f(r.g2), f(r.g3), f(r.g4), f(r.lam, 3)] for r in pa.itertuples()], ["ряд", "год", "θ", "γ1 (положение)", "γ2 (ln\\|r\\|)", "γ3 (знак)", "γ4 (объём)", "λ"]), "",
            "## 13. Условие для Y5/Y6 (только отчёт; не запуск)", "", md([[r.item, r.rule62] for r in t["CONDITION"].itertuples()], ["пункт: правило 6.2 против X1", "результат"]), "",
            "Условие: «правило 6.2 против X1 = да у Y1 или у Y3» — %s. Y1 — по `YA_RULE62.csv` (лист Y-A), Y3 — по этому листу. %s" % ("выполнено" if cd["условие Y5/Y6"] == "да" else "не выполнено", WARN), ""]
    return "\n".join(out)
