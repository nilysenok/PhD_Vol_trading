"""V4-RQC, G37: RQC_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» RQC по трём условиям (Ж-225 п. 10; `v4_e_go.rules`) и позиция K8 (crit_panel и простое среднее QLIKE 2017–2021 против K1, K2, K3, D2 и лучших B, C, NTP, CLD);
все числа — из таблиц стадии, пересчитываются проверкой. Модуль вне путей прогноза."""
from __future__ import annotations

import pandas as pd

from models_v2.s11 import v4_rqc_stage as ST

DEVIATIONS = [
    "До MIN_PAIRS = 60 закрытых пар (панели у RQC1/RQC2, бумаги у RQC3) a = 1 — в задании порог не назван, взят единый для программы. Год пар RQC2 — по дате цели t⁺ (пара декабря с t⁺ в январе относится к новому году).",
    "RQC3: «заморозка на 5 торговых дней» = дни d…d+4 (день-скачок включён); заморожен только выход a, EWMA продолжает принимать пары; вне заморозки шаг к цели ≤ 0,02. Панельная часть — EWMA по дням (полураспад 60) по winsorized парам панели; "
    "процентили скачков и обрезки — по прошлым дням/парам бумаги (≥ 60), без текущей; ln a — EWMA winsorized ln x буквально по заданию. Это геометрическое среднее, не mean(RV/K8): смещено вниз по Йенсену (средний ln a ≈ −0,25), поэтому RQC3 структурно хуже K8. Вариант «EWMA winsorized x в уровне, ln — только для стягивания и шага» не считался: конфигурации сверх трёх запрещены, правка после просмотра — решение дирижёра.",
    "K8 считается с прежними K1, K2, K3 без обучения; конфигураций израсходовано 4 (K8, RQC1–RQC3), всего 35 из 60. Лучшая RQC выбрана по crit_panel (окна 2022+ описательно).",
    "Правки прежнего: `v4_trials.verify`, `v4_e_check`, `v4_dx_cs_check`, `v4_ntpd_check`, `v4_cldd_check` — допустимы строки `Ж-222:K8`, `Ж-222:RQC*`; `run_v4`: стадия RQC в таблице LATE.",
]


def _q(o):
    return o["comp"].set_index(["series", "window"])["qlike"]


def _best_of(o, names):
    cs = o["crit"].set_index("series")["crit_panel"]
    return min(names, key=lambda x: float(cs[x]))


def refs(o) -> list:
    cp = o["cp"]
    return [("K1", "k1"), ("K2", "k2"), ("K3", "k3"), ("D2", "d2")] + [(cp[k], cp[k]) for k in ("B_best", "C_best_overall", "NTP_best", "CLD_best")]


def first_line(o: dict) -> str:
    cs, cp, q, best = o["crit"].set_index("series")["crit_panel"], o["cp"], _q(o), o["best"]
    g, yn = o["go"]["cfg"][best], (lambda x: "да" if x else "нет")
    mn = lambda k: min(x[k] for x in o["go"]["ref"].values())
    pos = "; ".join("%s %.5f/%.5f" % (nm, cs[s], q[(s, "2017_2021")]) for nm, s in refs(o))
    return ("%s — RQC: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s; (2) без 2017 %.5f ≤ min(K2,K3,D2) %.5f − 0,0005: %s; (3) без 1 %% дат %.5f ≤ %.5f − 0,0005: %s; "
            "K8 (контроль) crit_panel %.5f / простое среднее QLIKE 2017–2021 %.5f против (crit_panel/среднее): %s" % (
                "go" if g["go"] else "stop", best, cs[best], cp["go_threshold_B"], yn(g["c1"]), g["crit_no2017"], mn("crit_no2017"), yn(g["c2"]), g["crit_trim1"], mn("crit_trim1"), yn(g["c3"]), cs["K8"], q[("K8", "2017_2021")], pos))


def _delta(o, s: str) -> str:
    q, y = _q(o), o["years"].set_index(["series", "year"])["ann_qlike"]
    d = lambda a, b, k: 100.0 * (a - b) / k
    parts = []
    for lab, f in (("вся", lambda r: q[(r, "all")]), ("2017", lambda r: y[(r, 2017)]), ("2018", lambda r: y[(r, 2018)]), ("2022–2025", lambda r: q[(r, "2022_2025")])):
        parts.append("%s %s" % (lab, "/".join("%.2f" % d(f(s), f(r), f("k1")) for r in ("k1", "k2", "k3", "d2"))))
    return "%s: %s" % (s, "; ".join(parts))


def position(o) -> list:
    cs, q = o["crit"].set_index("series")["crit_panel"], _q(o)
    w = [x for x in ST.ALLW]
    bet = [x for x in w if cs[x] < cs["K8"]]
    bq = [x for x in w if q[(x, "2017_2021")] < q[("K8", "2017_2021")]]
    st = ", ".join("%s %+.5f" % (b, cs[b] - cs["K8"]) for b in (_best_of(o, v) for v in ST.STAGE_W.values()))
    g, mn, yn = o["go"]["cfg"]["K8"], (lambda k: min(x[k] for x in o["go"]["ref"].values())), (lambda x: "да" if x else "нет")
    k8 = "K8 по трём условиям (контроль, не кандидат на «go»): (1) crit_panel %.5f ≤ %.5f: %s; (2) без 2017 %.5f ≤ %.5f − 0,0005: %s; (3) без 1 %% дат %.5f ≤ %.5f − 0,0005: %s; итог %s." % (
        cs["K8"], o["cp"]["go_threshold_B"], yn(g["c1"]), g["crit_no2017"], mn("crit_no2017"), yn(g["c2"]), g["crit_trim1"], mn("crit_trim1"), yn(g["c3"]), "go" if g["go"] else "stop")
    return [k8, "Оценённых весов B–CLD: %d. Лучше K8 по crit_panel: %d (%s); по простому среднему QLIKE 2017–2021: %d. Лучший вес каждого этапа минус K8 по crit_panel (минус = лучше равных): %s." % (
        len(w), len(bet), ", ".join(sorted(bet, key=lambda x: cs[x])[:8]) or "нет", len(bq), st)]


def a_lines(o) -> list:
    a = o["a"][o["a"]["kind"] == "window"].pivot(index="config", columns="month", values="log_a")
    f = lambda c: "; ".join("%s %+.4f" % (m, a.loc[c, m]) for m in a.columns)
    mx = lambda cs_: max(abs(a.loc[c, m]) for c in cs_ for m in ("2018-03..04 (санкции)", "2022-02-24..03-31"))
    return ["Среднее ln a в окнах шоков (описательно): " + " | ".join("%s: %s" % (c, f(c)) for c in ("RQC1", "RQC2", "RQC3")), "E1–E5 (ln-множитель b к своей базе): " + " | ".join("%s: %s" % (c, f(c)) for c in ("E1", "E2", "E3", "E4", "E5")),
            "Максимум |средний ln| в двух окнах шоков: RQC3 %.4f, RQC1 %.4f; E1–E5 %.4f." % (mx(["RQC3"]), mx(["RQC1"]), mx(["E1", "E2", "E3", "E4", "E5"]))]


def stable(o) -> tuple:
    """Калибровка после скачков устойчива: RQC3 не хуже K8 по crit_panel и по среднему QLIKE M16 в 2018 и в 2022 (правило задано до счёта в этом отчёте)."""
    cs, y = o["crit"].set_index("series")["crit_panel"], o["years"].set_index(["series", "year"])["ann_qlike"]
    d = (cs["RQC3"] - cs["K8"], y[("RQC3", 2018)] - y[("K8", 2018)], y[("RQC3", 2022)] - y[("K8", 2022)])
    return all(x <= 0 for x in d), d


def build(o: dict) -> str:
    cs, q, cp, best = o["crit"].set_index("series")["crit_panel"], _q(o), o["cp"], o["best"]
    sb, lo = o["sber"][o["sber"]["window"] == "all"].set_index("series")["qlike"], o["loto"][(o["loto"]["held_out"] == "M16") & (o["loto"]["window"].isin(["all", "2017_2021"]))].set_index("window")
    ok, d = stable(o)
    ts = o["tests"][(o["tests"]["config"] == best) & (o["tests"]["comparator"] == "K8") & (o["tests"]["window"].isin(["2022_2025", "all"]))]
    L = [first_line(o), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; выбор — только crit_panel (2017–2021); окна 2022+ — описательно (Ж-225).", "",
         "## Отступления и толкования (до выводов)"] + ["- " + x for x in DEVIATIONS] + ["",
         "## Δ QLIKE к K1/K2/K3/D2, % уровня K1 (M16; минус = лучше)", _delta(o, "K8"), _delta(o, best), "", "## Позиция K8 против оценённых весов B–CLD"] + position(o) + ["",
         "## Поведение a в 2018 и 2022"] + a_lines(o) + ["Доля дней с заморозкой у RQC3: %.3f; RQC1 на границах [0,6; 1,6]: %.3f." % (
             float(o["fc"]["RQC3"]["frozen"].mean()), float(o["fc"]["RQC1"]["a"].isin([0.6, 1.6]).mean())),
         "Устойчива ли калибровка после скачков: %s (RQC3 − K8: crit_panel %+.5f, 2018 %+.5f, 2022 %+.5f; правило — все три ≤ 0)." % ("да" if ok else "нет", *d), "",
         "## Статистика лучшей RQC к K8 (M16; Δ, DM p, p_holm N=%d / N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"]),
         "; ".join("%s: %+.5f, p %.3f, p_holm %.3f, p_holm_wide %.3f" % (r.window, r.delta_mean, r.dm_p, r.p_holm, r.p_holm_wide) for r in ts.itertuples()), "",
         "## SBER (2271 дата; ориентир 0,2081 — потолок: c, λ выбраны на всей истории)", "; ".join("%s %.4f" % (s, sb[s]) for s in ("k1", "k2", "k3", "d2", "K8") + tuple(ST.CFGS[1:])), "",
         "## LOTO лучшей RQC (%s; M16: LOTO / пул / K1)" % best] + ["- %s: %.4f / %.4f / %.4f" % (w, lo.loc[w, "q_loto"], lo.loc[w, "q_pool"], lo.loc[w, "q_k1"]) for w in ("2017_2021", "all")] + ["",
         "## Реестр", "В ALL_TRIALS строки Ж-222:K8, RQC1–RQC3 (`v4_trials.replace_rows`); израсходовано 4 конфигурации, всего 35 из 60.", "",
         "Файлы: RQC_COMPARISON, RQC_PER_TICKER, RQC_TESTS, RQC_YEARS, RQC_SBER, RQC_LOTO, RQC_A (.csv); %s." % o["note"]]
    return "\n".join(L) + "\n"
