"""V4-DX-CS, G28: CS_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» по трём условиям (Ж-225 п. 10; `v4_e_go.rules`); все числа берутся из таблиц стадии, не вводятся вручную.
Модуль — вне путей прогноза (использует диагностику `v4_dx_cold`)."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_b_report as BR

DEVIATIONS = [
    "База — C8 из набора C-b как есть (признак med_lrv, κ = λ = 250·s², старт a = 0, b = −1, месячная переоценка); оценщик CS — копия целевой функции `v4_cb_fit.fit_regime` с вектором κ_i (`v4_cs_fit`); при расписании ≡ 250 совпадает с C8 до 1e-8 (тест).",
    "n в CS1 — число закрытых пар панели на d0 (t⁺ < d0, без исключённой бумаги в LOTO); n_i в CS2 — закрытые пары бумаги i на d0. Бумага без пар в обучении получает панельные параметры (отклонения 0), как в C8.",
    "«При n → 0 CS равны C8 на первом месяце оценки» проверено как тождество на расписании (sched(0) = 250) и совпадение приорных месяцев (до 60 пар) побитно; на первом месяце с оценкой n > 0 (≥ 60 пар), поэтому там CS ≠ C8 по построению.",
    "Холодный старт (а), (б): ЗАДНИМ ЧИСЛОМ — параметры/вес видят цели 2017, поэтому это верхняя граница потерь, не достижимый результат; «лучший допустимый приор по данным до 2017» — нет данных (OOF K2/K3 за 2016 нет, псевдоистория не создавалась).",
    "«Механичность»: сравниваются Δ crit_panel к C8 целиком и без целей 2017 (условие 2 «go»), а также Δ среднего QLIKE по годам; вердикт по правилу «вклад без 2017 ≥ половины общего» (описательно).",
]


def _crit_lines(crit: pd.Series, go: dict, thr: float) -> list:
    L = ["| ряд | crit_panel | без 2017 | без 1 % дат |", "|---|---|---|---|"]
    for s in ("k1", "k2", "k3", "d2", "C1", "C4", "C8", "CS1", "CS2"):
        g = go["ref"].get(s) or go["cfg"][s]
        L.append("| %s | %.5f | %.5f | %.5f |" % (s, crit[s], g["crit_no2017"], g["crit_trim1"]))
    return L + ["", "Порог go (1): %.5f; (2), (3): min(K2, K3, D2) − 0,0005 по тому же критерию." % thr]


def _year_lines(yt: pd.DataFrame) -> list:
    L = ["| год | K1 | C8 | CS1 | CS2 | CS1 − C8 | CS2 − C8 |", "|---|---|---|---|---|---|---|"]
    t = yt.pivot_table(index="year", columns="series", values="ann_qlike")
    for y in t.index:
        L.append("| %d | %.4f | %.4f | %.4f | %.4f | %+.4f | %+.4f |" % (y, t.loc[y, "k1"], t.loc[y, "C8"], t.loc[y, "CS1"], t.loc[y, "CS2"], t.loc[y, "CS1"] - t.loc[y, "C8"], t.loc[y, "CS2"] - t.loc[y, "C8"]))
    return L


def _mech(go: dict, crit: pd.Series, best: str) -> str:
    tot, n17 = float(crit["C8"] - crit[best]), float(go["cfg"]["C8"]["crit_no2017"] - go["cfg"][best]["crit_no2017"])
    if tot <= 0:
        return "%s не лучше C8 по crit_panel (Δ = %+.5f): улучшения нет, вопрос о механическом устранении 2017 не возникает." % (best, -tot)
    share = n17 / tot
    verdict = "не сводится к одному 2017" if share >= 0.5 else "в основном обеспечено 2017 (механическое устранение одного года)"
    return "%s: улучшение crit_panel к C8 = %.5f; без целей 2017 — %.5f (%.0f %% общего) → улучшение %s." % (best, tot, n17, 100 * share, verdict)


def first_line(r: dict, d: pd.DataFrame) -> str:
    best, crit, go, cp = r["best"], r["crit"].set_index("series")["crit_panel"], r["go"], r["cp"]
    g, comp = go["cfg"][best], r["comp"].set_index(["series", "window"])["qlike"]
    k1 = lambda w: comp[("k1", w)]
    dc = lambda w: 100.0 * (comp[(best, w)] - comp[("C8", w)]) / k1(w)
    yt = r["years"].pivot_table(index="year", columns="series", values="ann_qlike")
    dy = lambda y: 100.0 * (yt.loc[y, best] - yt.loc[y, "C8"]) / yt.loc[y, "k1"]
    loss = d[(d["group"] == "M16") & d["item"].str.startswith("loss_cold")].set_index("item")["qlike_2017"]
    return ("%s — Cold Start: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s; (2) без 2017 %.5f: %s; (3) без 1 %% дат %.5f: %s; Δ QLIKE к C8 (%% уровня K1, M16): вся история %.2f, 2017 %.2f, 2018 %.2f, 2022–2025 %.2f; "
            "потери холодного старта 2017 (потолок, задним числом): (а) %.4f, (б) %.4f QLIKE" % (
                "go" if g["go"] else "stop", best, crit[best], cp["go_threshold_B"], "да" if g["c1"] else "нет", g["crit_no2017"], "да" if g["c2"] else "нет", g["crit_trim1"], "да" if g["c3"] else "нет",
                dc("all"), dy(2017), dy(2018), dc("2022_2025"), loss["loss_cold_start_a"], loss["loss_cold_start_b"]))


def build(r: dict, d: pd.DataFrame) -> str:
    cp, crit, best, go = r["cp"], r["crit"].set_index("series")["crit_panel"], r["best"], r["go"]
    L = [first_line(r, d), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; диагностика — задним числом, описательно; окна 2022+ в выборе не участвуют.", "",
         "## Отступления и толкования (до выводов)"] + ["- " + x for x in DEVIATIONS]
    L += ["", "## crit_panel и три условия «go» (M16 × цели 2017–2021, `hp_stats_sel.crit_panel`)", ""] + _crit_lines(crit, go, cp["go_threshold_B"])
    L += ["", "Условия лучшей (%s): (1) %s, (2) %s, (3) %s." % (best, *("да" if go["cfg"][best][k] else "нет" for k in ("c1", "c2", "c3")))]
    L += ["", "## Холодный старт 2017 (CS_DIAG.csv; M16 / SBER, QLIKE по целям 2017)", "", "| ряд | M16 | SBER |", "|---|---|---|"]
    q = d[d["item"].isin(["k1", "k2", "k3", "d2", "C8", "CS1", "CS2", "a_C8_params_end2017", "b_best_const_weight_2017", "loss_cold_start_a", "loss_cold_start_b"])]
    for it in q["item"].unique():
        m, s = (q[(q["item"] == it) & (q["group"] == g)]["qlike_2017"].iloc[0] for g in ("M16", "SBER"))
        L.append("| %s | %.4f | %.4f |" % (it, m, s))
    L += ["", "Лучший постоянный вес на M за 2017 (б): %.3f. Приор до 2017: нет данных (OOF K2/K3 за 2016 нет; псевдоистория не создавалась)." % d["w_const"].dropna().iloc[0]]
    L += ["", "## Средний QLIKE по годам цели (M16; 2017–2021 выбор, 2022–2025 описательно)", ""] + _year_lines(r["years"][r["years"]["year"] <= 2025])
    L += ["", "## Не является ли улучшение механическим устранением одного года", "", _mech(go, crit, best)]
    L += ["", "## Холм (лучшая; основное семейство M16 × {2022_2025, all} × CS1–CS2, N=%d; N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"]), ""] + BR.holm_lines({**r, "best": best, "tests": r["tests"][r["tests"]["comparator"].isin(["k1", "d2", "C8"])]})
    L += ["", "(Полная таблица — `CS_TESTS.csv`; здесь K1, D2, C8.)"]
    L += ["", "## SBER (отдельно, в выбор не входит; 2271 дата; ориентир 0,2081 — потолок)", ""] + BR.sber_lines(r)
    lt = r["loto"][r["loto"]["held_out"] == "M16"].set_index("window")
    L += ["", "## LOTO лучшей (%s; M16: LOTO / пул / K1)" % best, ""] + ["- %s: %.4f / %.4f / %.4f" % (w, lt.loc[w, "q_loto"], lt.loc[w, "q_pool"], lt.loc[w, "q_k1"]) for w in ("2017_2021", "2022_2025", "all")]
    L += ["", "Израсходовано конфигураций листа: 2 (CS1, CS2) из 60; после листа %d из 60 (CHECKPOINT.json, `used_after_CS`)." % (int(cp["used_after_E"]) + 2)]
    return "\n".join(L) + "\n"
