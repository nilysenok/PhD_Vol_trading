"""V4-H, G41: H_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» этапа H по трём условиям (Ж-225 п. 10; `v4_e_go.rules`), лучшая H, её crit_panel против порога 0,26095 и против K8, простое среднее QLIKE 2017–2021;
все числа — из таблиц стадии, пересчитываются проверкой. Модуль вне путей прогноза."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_hd_value as V

DEVIATIONS = [
    "Обрезка g ∈ [−0,5; 0,5] применяется при прогнозе, в обучении цель гладкая (без обрезки). s² для штрафа = mean z̃² на окне оценки (z̃ стандартизован, s² ≈ 1). Гребень проекции z на базу — λ = 250 в единицах стандартизованных величин.",
    "Порог 60 закрытых пар (g = 0 до него) — единый для программы, в задании не назван. H6: «статистика CUSUM» = brk = ln(1 + дни с последнего срабатывания) из BRK (`v4_cld_brk`, пересчитан по набору бумаг; равен BRK.parquet на 17). "
    "Доля первого часа — rv_open/rv из `daily_<T>` (M10, Ж-214), в Z этапа C её не было. Процентили скачка — окно 252 строк бумаги до d (≥ 60).",
    "Правки прежнего (G38): `run_v4 --stage DX --check` теперь только читает выходы с диска (раньше пересчитывал стадию и перезаписывал DX_runtime_sec); `--check` стадий audit, A, B, Ca, Cb, E, CS, NTP, CLD, RQC выполняется в песочнице (`v4_sandbox`: копия выходов, "
    "пересчёт и проверка там, проект не меняется) — прямое чтение невозможно: стадия Ca при `--check` затирала таблицы Cb. Допустимы строки `Ж-222:H*` в `v4_trials.verify`, `v4_e_check`, `v4_dx_cs_check`, `v4_ntpd_check`, `v4_cldd_check`, `v4_rqc_check`; "
    "`v4_rqc_cache.prior_sha` не считает файлы этапа H; `run_v4`: стадия H в таблице LATE. `v4_e_check` входит в хэш кода E: ключ `E_cache_key` сменился, таблицы E пересчитаны (прогнозы E побитно прежние, `E_runtime_sec` возвращён к 86,0).",
    "`--stage H` записывает, `--stage H --check` только читает (кэш и таблицы пересчитываются в памяти и сверяются с файлами). Гиперпараметры (κ, λ проекции, обрезка, порог пар) не подбирались; результат — единственный объявленный счёт. Израсходовано 8 конфигураций, всего 43 из 60.",
]


def _q(o):
    return o["comp"].set_index(["series", "window"])["qlike"]


def first_line(o: dict) -> str:
    cs, cp, q, best = o["crit"].set_index("series")["crit_panel"], o["cp"], _q(o), o["best"]
    g, yn = o["go"]["cfg"][best], (lambda x: "да" if x else "нет")
    mn = lambda k: min(x[k] for x in o["go"]["ref"].values())
    return ("%s — H: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s (K8 %.5f, Δ к K8 %+.5f); (2) без 2017 %.5f ≤ min(K2,K3,D2) %.5f − 0,0005: %s; (3) без 1 %% дат %.5f ≤ %.5f − 0,0005: %s; "
            "простое среднее QLIKE 2017–2021: %s %.5f, K8 %.5f, K1 %.5f, D2 %.5f" % (
                "go" if g["go"] else "stop", best, cs[best], cp["go_threshold_B"], yn(g["c1"]), cs["K8"], cs[best] - cs["K8"], g["crit_no2017"], mn("crit_no2017"), yn(g["c2"]), g["crit_trim1"], mn("crit_trim1"), yn(g["c3"]),
                best, q[(best, "2017_2021")], q[("K8", "2017_2021")], q[("k1", "2017_2021")], q[("d2", "2017_2021")]))


def _delta(o, s: str) -> str:
    q, y = _q(o), o["years"].set_index(["series", "year"])["ann_qlike"]
    parts = []
    for lab, f in (("вся", lambda r: q[(r, "all")]), ("2017", lambda r: y[(r, 2017)]), ("2018", lambda r: y[(r, 2018)]), ("2022–2025", lambda r: q[(r, "2022_2025")])):
        parts.append("%s %s" % (lab, "/".join("%.2f" % (100.0 * (f(s) - f(r)) / f("k1")) for r in ("k1", "k2", "k3", "d2", "K8"))))
    return "%s к K1/K2/K3/D2/K8: %s" % (s, "; ".join(parts))


def has_value(o: dict, cfg: str) -> bool:
    """Источник даёт условную ценность сверх K8 (описательное правило отчёта): crit_panel ниже K8 и Δ QLIKE к K8 на всей истории < 0 при DM p < 0,05."""
    cs, t = o["crit"].set_index("series")["crit_panel"], o["tests"]
    r = t[(t["config"] == cfg) & (t["comparator"] == "K8") & (t["window"] == "all")].iloc[0]
    return bool(cs[cfg] < cs["K8"] and r["delta_mean"] < 0 and r["dm_p"] < 0.05)


def source_lines(o: dict) -> list:
    cs, q, t = o["crit"].set_index("series")["crit_panel"], _q(o), o["tests"]
    out = []
    for c in o["cfgs"]:
        r = {w: t[(t["config"] == c) & (t["comparator"] == "K8") & (t["window"] == w)].iloc[0] for w in ("2022_2025", "all")}
        out.append("- %s (%s): Δ crit_panel к K8 %+.5f; Δ QLIKE к K8 2017–2021 %+.5f, 2022–2025 %+.5f (DM p %.3f, p_holm %.3f), вся %+.5f (p %.3f, p_holm %.3f); ценность: %s" % (
            c, V.SOURCE[c], cs[c] - cs["K8"], q[(c, "2017_2021")] - q[("K8", "2017_2021")], r["2022_2025"]["delta_mean"], r["2022_2025"]["dm_p"], r["2022_2025"]["p_holm"], r["all"]["delta_mean"], r["all"]["dm_p"],
            r["all"]["p_holm"], "да" if has_value(o, c) else "нет"))
    return out


def out_lines(o: dict) -> list:
    v = o["value"]
    a = v[(v["config"] == o["best"]) & (v["window"] == "all")].iloc[0]
    return ["Строк M16 с RV вне [min(H, M), max(H, M)] (вся история): %.1f %%. Лучшая %s: выигрыш к K8 на строку вне диапазона %+.6f QLIKE, внутри %+.6f; доля общего выигрыша с этих строк %.2f; потеря K8 на строку вне диапазона %.4f." % (
        100.0 * a["share_rows_out_of_range"], o["best"], a["gain_per_row_out"], a["gain_per_row_in"], a["share_of_gain_out"], a["loss_k8_out"])]


def build(o: dict) -> str:
    cs, q, cp, best = o["crit"].set_index("series")["crit_panel"], _q(o), o["cp"], o["best"]
    sb = o["sber"][o["sber"]["window"] == "all"].set_index("series")["qlike"]
    lo = o["loto"][(o["loto"]["held_out"] == "M16") & (o["loto"]["window"].isin(["all", "2017_2021"]))].set_index("window")
    ts = o["tests"][(o["tests"]["config"] == best) & (o["tests"]["window"].isin(["2022_2025", "all"]))]
    L = [first_line(o), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; выбор — только crit_panel (2017–2021); окна 2022+ — описательно (Ж-225, Ж-222).", "",
         "## Отступления и толкования (до выводов)"] + ["- " + x for x in DEVIATIONS] + ["", "## Условная ценность источников сверх K8 (M16; минус = лучше K8)"] + source_lines(o) + [
         "Правило «ценность» (описательное): crit_panel ниже K8, Δ QLIKE к K8 на всей истории < 0, DM p < 0,05.", "", "## Лучшая H: Δ QLIKE, % уровня K1 (M16; минус = лучше)", _delta(o, best),
         "Контроли crit_panel: K1 %.5f, K2 %.5f, K3 %.5f, D2 %.5f, K8 %.5f, C8 %.5f, CLD1 %.5f." % tuple(cs[x] for x in ("k1", "k2", "k3", "d2", "K8", "C8", "CLD1")), "",
         "## Дни вне диапазона [H, M]"] + out_lines(o) + ["", "## Статистика лучшей H к K8 (M16; Δ, DM p, p_holm N=%d / N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"])] + [
         "- %s: %+.5f, p %.3f, p_holm %.3f, p_holm_wide %.3f" % (r.window, r.delta_mean, r.dm_p, r.p_holm, r.p_holm_wide) for r in ts[ts["comparator"] == "K8"].itertuples()] + ["",
         "## SBER (2271 дата; ориентир 0,2081 — потолок: c, λ выбраны на всей истории)", "; ".join("%s %.4f" % (s, sb[s]) for s in ("k1", "k2", "k3", "d2", "K8", best)), "",
         "## LOTO лучшей H (%s; M16: LOTO / пул / K1)" % best] + ["- %s: %.4f / %.4f / %.4f" % (w, lo.loc[w, "q_loto"], lo.loc[w, "q_pool"], lo.loc[w, "q_k1"]) for w in ("2017_2021", "all")] + ["",
         "## Реестр", "В ALL_TRIALS строки Ж-222:H1…H8 (`v4_trials.replace_rows`); израсходовано 8 конфигураций, всего 43 из 60.", "",
         "Файлы: H_COMPARISON, H_PER_TICKER, H_TESTS, H_YEARS, H_SBER, H_LOTO, H_VALUE, H_COEF (.csv); %s." % o["note"]]
    return "\n".join(L) + "\n"
