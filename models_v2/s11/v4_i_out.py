"""V4-I, G43–G44: точность компонент (I_COMP.csv) и отчёт I_REPORT.md (≤ 1000 слов). Первая строка — «go»/«stop» этапа I по трём условиям (Ж-225 п. 10; `v4_e_go.rules`), лучшая I, её crit_panel против порога 0,26095 и против K8,
простое среднее QLIKE 2017–2021; все числа — из таблиц стадии, пересчитываются проверкой. Модуль вне путей прогноза."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_b_stats as ST, v4_io as IO

DEVIATIONS = [
    "C + J = RV: в плавающей арифметике побитно равна только C = RV − J (100 % строк); сумма C + J отличается от RV не более чем на один ulp (≤ 2,2e-16 отн.; побитно совпадает ~95 % строк). Тождество проверяется так (`v4_i_comp.identity_ok`), мутация «C := BV» ловится.",
    "Событие «скачок» J > 0: в данных J = max(RV − BV, 0) уже обнулён ниже порога, событие = J > 0 (≈ 78 % дней, поэтому p̂ близко к частоте, а точность «верно предсказанных скачков» высока тривиально — ориентир: частота события и Брайер константы).",
    "Порог 60 закрытых пар (S = K8 до него и в строках с нефинитным признаком) — единый для программы. Стандартизация признаков и гребень: λ = κ = 250 на стандартизованных признаках окна; свободные члены бумаг — κ = 250 (b0 не штрафуется). m̂ = EWMA₆₀ по дням J > 0 (по событиям), стянутая к панельному среднему J при J > 0 по закрытым парам.",
    "θ I3 обучается на прогнозах S прошлых месяцев (вне выборки; строки до 60 пар не участвуют); гребень λ = 250·mean(u²), как объявлено. Гиперпараметры не подбирались; результат — единственный объявленный счёт: 3 конфигурации, всего 46 из 60.",
    "Правки прежнего (минимальные, строкой): допустимы строки `Ж-222:I*` в `v4_trials.verify`, `v4_e_check`, `v4_dx_cs_check`, `v4_ntpd_check`, `v4_cldd_check`, `v4_rqc_check`, `v4_hd_check`; `prior_sha` стадий H и RQC не считает файлы этапа I; `run_v4`: стадия I в таблице LATE.",
]


def _ql(rv, f):
    r = np.asarray(rv, float) / np.asarray(f, float)
    return r - np.log(r) - 1.0


def _one(g: pd.DataFrame) -> dict:
    ev, p, c, j = g["ev_t"].to_numpy(float), g["phat"].to_numpy(float), g["c_t"].to_numpy(float), g["j_t"].to_numpy(float)
    ch, jh = g["chat"].to_numpy(float), g["jhat"].to_numpy(float)
    rv = g["rv"].to_numpy(float)
    pred = p >= 0.5
    return {"n": len(g), "qlike_c": float(_ql(c, ch).mean()), "bias_c": float(ch.mean() / c.mean() - 1.0), "rmse_c": float(np.sqrt(np.mean((ch - c) ** 2))),
            "base_rate": float(ev.mean()), "acc_jump": float(np.mean(pred == (ev > 0))), "recall_jump": float(pred[ev > 0].mean()) if (ev > 0).any() else np.nan,
            "brier_p": float(np.mean((p - ev) ** 2)), "brier_const": float(np.mean((ev.mean() - ev) ** 2)), "mean_jhat": float(jh.mean()), "mean_j": float(j.mean()),
            "rmse_jhat": float(np.sqrt(np.mean((jh - j) ** 2))), "rmse_j_zero": float(np.sqrt(np.mean(j ** 2))),
            "qlike_rv_chat": float(_ql(rv, ch).mean()), "qlike_rv_i1": float(_ql(rv, ch + jh).mean()), "qlike_rv_k1": float(_ql(rv, g["h"].to_numpy(float)).mean()),
            "jshare_hat": float(np.mean(jh / (ch + jh))), "jshare_true": float(np.mean(j / rv))}


def acc_table(cmp: pd.DataFrame, Fe: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    """Точность Ĉ (QLIKE к C, смещение, RMSE), p̂ (доля верных при пороге 0,5, полнота скачков, Брайер p̂ против константы-частоты окна), Ĵ (среднее, RMSE против нулевого прогноза), вклад Ĵ в RV (QLIKE(RV; Ĉ) → QLIKE(RV; Ĉ + Ĵ)).
    Только строки с компонентным прогнозом и общие даты основного сравнения; M16 — среднее по бумагам, SBER — отдельно."""
    m = cmp.merge(Fe[["ticker", "date", "c_t", "j_t", "ev_t"]], on=["ticker", "date"]).merge(c[["ticker", "date", "group"]], on=["ticker", "date"])
    m = m[np.isfinite(m[["chat", "jhat", "phat", "c_t"]].to_numpy(float)).all(1) & (m["c_t"] > 0)]
    rows = []
    for w in ST.WINS:
        s = m[ST.win_mask(m, w)]
        per = pd.DataFrame([{"ticker": t, "group": g["group"].iloc[0], **_one(g)} for t, g in s.groupby("ticker")])
        if per.empty:
            continue
        for scope, sub in (("M16", per[per["group"] == "M16"]), ("SBER", per[per["ticker"] == "SBER"])):
            if len(sub):
                rows.append({"scope": scope, "window": w, "n_tickers": len(sub), **sub.drop(columns=["ticker", "group"]).mean().to_dict()})
    return pd.DataFrame(rows)


def _q(o):
    return o["comp"].set_index(["series", "window"])["qlike"]


def first_line(o: dict) -> str:
    cs, cp, q, best = o["crit"].set_index("series")["crit_panel"], o["cp"], _q(o), o["best"]
    g, yn = o["go"]["cfg"][best], (lambda x: "да" if x else "нет")
    mn = lambda k: min(x[k] for x in o["go"]["ref"].values())
    return ("%s — I: лучшая %s, (1) crit_panel %.5f ≤ порога %.5f: %s (K8 %.5f, Δ к K8 %+.5f); (2) без 2017 %.5f ≤ min(K2,K3,D2) %.5f − 0,0005: %s; (3) без 1 %% дат %.5f ≤ %.5f − 0,0005: %s; "
            "простое среднее QLIKE 2017–2021: %s %.5f, K8 %.5f, K1 %.5f, D2 %.5f" % (
                "go" if g["go"] else "stop", best, cs[best], cp["go_threshold_B"], yn(g["c1"]), cs["K8"], cs[best] - cs["K8"], g["crit_no2017"], mn("crit_no2017"), yn(g["c2"]), g["crit_trim1"], mn("crit_trim1"), yn(g["c3"]),
                best, q[(best, "2017_2021")], q[("K8", "2017_2021")], q[("k1", "2017_2021")], q[("d2", "2017_2021")]))


def _delta(o, s: str) -> str:
    q, y = _q(o), o["years"].set_index(["series", "year"])["ann_qlike"]
    parts = []
    for lab, f in (("вся", lambda r: q[(r, "all")]), ("2017", lambda r: y[(r, 2017)]), ("2018", lambda r: y[(r, 2018)]), ("2022–2025", lambda r: q[(r, "2022_2025")])):
        parts.append("%s %s" % (lab, "/".join("%.2f" % (100.0 * (f(s) - f(r)) / f("k1")) for r in ("k1", "k2", "k3", "d2", "K8"))))
    return "%s к K1/K2/K3/D2/K8: %s" % (s, "; ".join(parts))


def hajj_verdict(o: dict) -> list:
    """«Учитывает ли HAR-J уже ожидаемый вклад скачков»: I1 против K1 (все окна, % уровня K1) и вклад Ĵ (QLIKE(RV; Ĉ) − QLIKE(RV; Ĉ + Ĵ), M16, вся история)."""
    q, a = _q(o), o["acc"].set_index(["scope", "window"])
    d1 = 100.0 * (q[("I1", "all")] - q[("k1", "all")]) / q[("k1", "all")]
    d17 = 100.0 * (q[("I1", "2017_2021")] - q[("k1", "2017_2021")]) / q[("k1", "2017_2021")]
    r = a.loc[("M16", "all")]
    gain = float(r["qlike_rv_chat"] - r["qlike_rv_i1"])
    ans = "да (I1 не лучше K1)" if d1 >= 0 else "частично: I1 лучше K1 (компонентный разбор даёт информацию сверх ln(1+J_d) в HAR-J)"
    return ["I1 против K1 (M16, QLIKE, %% уровня K1): вся %+.2f, 2017–2021 %+.2f; вклад Ĵ (QLIKE(RV; Ĉ) − QLIKE(RV; Ĉ + Ĵ), M16, вся) %+.5f (плюс = Ĵ помогает); доля скачка в прогнозе %.3f против факта %.3f." % (
        d1, d17, gain, r["jshare_hat"], r["jshare_true"]), "Учитывает ли HAR-J уже ожидаемый вклад скачков (правило: I1 − K1 ≥ 0 на всей истории — да): %s." % ans]


def comp_lines(o: dict) -> list:
    a = o["acc"][(o["acc"]["scope"] == "M16") & o["acc"]["window"].isin(["2017_2021", "all"])].set_index("window")
    return ["- %s (M16): QLIKE(C; Ĉ) %.4f, смещение Ĉ %+.3f; скачок: частота %.3f, доля верных %.3f, полнота %.3f, Брайер p̂ %.4f против константы %.4f; Ĵ: среднее %.3f против %.3f, RMSE %.3f против нулевого %.3f" % (
        w, a.loc[w, "qlike_c"], a.loc[w, "bias_c"], a.loc[w, "base_rate"], a.loc[w, "acc_jump"], a.loc[w, "recall_jump"], a.loc[w, "brier_p"], a.loc[w, "brier_const"], a.loc[w, "mean_jhat"], a.loc[w, "mean_j"],
        a.loc[w, "rmse_jhat"], a.loc[w, "rmse_j_zero"]) for w in ("2017_2021", "all")]


def build(o: dict) -> str:
    cs, q, cp, best = o["crit"].set_index("series")["crit_panel"], _q(o), o["cp"], o["best"]
    sb = o["sber"][o["sber"]["window"] == "all"].set_index("series")["qlike"]
    lo = o["loto"][(o["loto"]["held_out"] == "M16") & (o["loto"]["window"].isin(["all", "2017_2021"]))].set_index("window")
    ts = o["tests"][(o["tests"]["config"] == best) & (o["tests"]["window"].isin(["2022_2025", "all"])) & (o["tests"]["comparator"] == "K8")]
    L = [first_line(o), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок; выбор — только crit_panel (2017–2021); окна 2022+ — описательно (Ж-225, Ж-222).", "", "## Отступления и толкования (до выводов)"] + [
        "- " + x for x in DEVIATIONS] + ["", "## Лучшая I: Δ QLIKE, % уровня K1 (M16; минус = лучше)", _delta(o, best), "Остальные: " + "; ".join(_delta(o, c) for c in o["cfgs"] if c != best),
         "crit_panel: K1 %.5f, K2 %.5f, K3 %.5f, D2 %.5f, K8 %.5f, C8 %.5f, CLD1 %.5f, H2 %.5f; I1 %.5f, I2 %.5f, I3 %.5f." % (tuple(cs[x] for x in ("k1", "k2", "k3", "d2", "K8", "C8", "CLD1", "H2")) + tuple(cs[x] for x in o["cfgs"])),
         "", "## Точность компонент"] + comp_lines(o) + ["", "## HAR-J и вклад скачков"] + hajj_verdict(o) + ["", "## Статистика лучшей I к K8 (M16; Δ, DM p, p_holm N=%d / N_wide=%d)" % (cp["N_declared_before_stats"], cp["N_wide"])] + [
        "- %s: %+.5f, p %.3f, p_holm %.3f, p_holm_wide %.3f" % (r.window, r.delta_mean, r.dm_p, r.p_holm, r.p_holm_wide) for r in ts.itertuples()] + ["",
        "## SBER (2271 дата; ориентир 0,2081 — потолок: c, λ выбраны на всей истории; сравнение с Ж-214 по SBER — только описательно: 2271 дата и определение RV те же, что в SBER_BENCH (аудит G3: rv_def_same, 0 расхождений); Ж-214 — другая модель и обучение)", "; ".join("%s %.4f" % (s, sb[s]) for s in ("k1", "k2", "k3", "d2", "K8", best)), "",
        "## LOTO лучшей I (%s; M16: LOTO / пул / K1)" % best] + ["- %s: %.4f / %.4f / %.4f" % (w, lo.loc[w, "q_loto"], lo.loc[w, "q_pool"], lo.loc[w, "q_k1"]) for w in ("2017_2021", "all")] + ["",
        "## Реестр", "В ALL_TRIALS строки Ж-222:I1…I3 (`v4_trials.replace_rows`); израсходовано 3 конфигурации, всего 46 из 60.", "",
        "Файлы: I_COMPARISON, I_PER_TICKER, I_TESTS, I_YEARS, I_SBER, I_LOTO, I_COMP (.csv); %s." % o["note"]]
    return "\n".join(L) + "\n"
