"""V4-DX, G26: DX_REPORT.md (≤ 1500 слов) — ответы на семь вопросов Ж-225 п. 7 по годам. Все числа берутся из DX_YEARS/DX_FEATURES/DX_2018 (не вводятся вручную); диагностика задним числом, описательно, без подбора."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEVIATIONS = [
    "Вес HAR-J — в геометрической смеси на пуле строк M16 (SBER — свои строки); интервал 90 % — блочный бутстрап дат (блок 20, 300 выборок вместо 5000 в тестах: решатель веса считается на каждой выборке); w ∈ [−1; 2], чтобы интервал не упирался в границу [0; 1]; «ML охватывает HAR-J» = нижняя граница ≤ 0; значение −1,00 или 2,00 — упор в границу решателя (2022, апрель 2018).",
    "O1 и O3 считаются по бумагам и усредняются (как в `A_ORACLE`); вес и доли дней — по пулу строк группы. Год = год цели t⁺.",
    "Квинтили признаков — по строкам M16 с датой до 1 января года среза; 2017 квинтилей не имеет (прошлого нет). «Дни после излома» = 28 календарных дней от 06.04.2018, 09.03.2020, 24.02.2022.",
    "Прогнозы C8, C1 — прежние (только чтение); 2022–2025 — только описание, подбора нет.",
]


def _p(x, k=1):
    return ("%." + str(k) + "f") % x


def first_line(Y: pd.DataFrame) -> str:
    kg = Y[(Y["group"] == "M16") & (Y["m"] == "KG")].set_index("slice")
    yr = ["Y%d" % y for y in range(2017, 2027)]
    a = kg.loc[yr]
    al = kg.loc["all"]
    return ("Гибрид не даёт устойчивого преимущества, потому что в %d из 10 лет HAR-J лучше √(K2·K3) (D̄ < 0), а в %d из 10 лет ML охватывает HAR-J (90 %% интервал веса HAR-J включает 0): выпуклая смесь даёт на всей истории лишь "
            "%s %% (постоянный вес, O1) и %s %% (условный вес внутри [min, max], O2) от QLIKE лучшей одиночной, тогда как %s %% дней RV лежит вне диапазона [H; M] и несёт %s %% потерь (потолок выхода за диапазон O3: %s %%); "
            "диагностика задним числом, M16" % (int((a["d_mean"] < 0).sum()), int(a["ml_encompasses_har"].astype(bool).sum()), _p(al["gain_o1_pct"]), _p(al["gain_o2_pct"]), _p(100 * al["share_outside"]),
                                                  _p(100 * al["loss_share_outside"]), _p(al["gain_o3_pct"])))


def year_table(Y: pd.DataFrame, group="M16", m="KG") -> list:
    L = ["| срез | D̄ | дней D<0, % | бумаг D̄<0 | вес HAR-J [90 %] | ML охв. | оба ниже, % | оба выше, % | в диапазоне, % | потерь вне, % | O1 / O2 / O3, % q |", "|" + "---|" * 11]
    for r in Y[(Y["group"] == group) & (Y["m"] == m)].itertuples():
        L.append("| %s | %+.4f | %s | %s | %.2f [%.2f; %.2f] | %s | %s | %s | %s | %s | %s / %s / %s |" % (
            r.slice, r.d_mean, _p(100 * r.share_d_neg), "%d/%d" % (r.n_tickers_har_better, r.n_tickers), r.w_har, r.w_har_lo90, r.w_har_hi90, "да" if r.ml_encompasses_har else "нет",
            _p(100 * r.share_both_under), _p(100 * r.share_both_over), _p(100 * r.share_in_range), _p(100 * r.loss_share_outside), _p(r.gain_o1_pct), _p(r.gain_o2_pct), _p(r.gain_o3_pct)))
    return L


def other_m(Y: pd.DataFrame) -> list:
    L = []
    for g in ("M16", "SBER"):
        for m in ("KG", "K2", "K3"):
            a = Y[(Y["group"] == g) & (Y["m"] == m) & Y["slice"].str.startswith("Y")]
            L.append("- %s, M = %s: HAR-J лучше в %d из %d лет; ML охватывает HAR-J в %d; HAR-J охватывает ML в %d." % (g, m, int((a["d_mean"] < 0).sum()), len(a), int(a["ml_encompasses_har"].astype(bool).sum()),
                                                                                int(a["har_encompasses_ml"].astype(bool).sum())))
    return L


def feat_lines(F: pd.DataFrame) -> list:
    L = []
    for w in ("2018_2021", "2022_2025"):
        a = F[(F["slice"] == w) & (F["quintile"] == "all")].copy()
        a["abs"] = a["spearman_d"].abs()
        for r in a.sort_values("abs", ascending=False).head(3).itertuples():
            q = F[(F["slice"] == w) & (F["feature"] == r.feature) & F["quintile"].isin(["1", "5"])].set_index("quintile")
            L.append("- %s, %s: ρ(D) = %+.3f, ρ(в диапазоне) = %+.3f; D̄ в 1-м квинтиле %+.4f, в 5-м %+.4f." % (w, r.feature, r.spearman_d, r.spearman_in_range, q.loc["1", "mean_d"], q.loc["5", "mean_d"]))
    for b, s in (("2018-04-06", "Y2018"), ("2020-03-09", "Y2020"), ("2022-02-24", "Y2022")):
        g = F[(F["slice"] == s) & (F["feature"] == "post_" + b)].set_index("quintile")
        if {"0", "1"} <= set(g.index):
            L.append("- после излома %s (28 дн.): D̄ %+.4f (n = %d) против %+.4f вне окна; в диапазоне %s %% против %s %%." % (b, g.loc["1", "mean_d"], g.loc["1", "n"], g.loc["0", "mean_d"], _p(100 * g.loc["1", "share_in_range"]), _p(100 * g.loc["0", "share_in_range"])))
    return L


def d2018_lines(T: pd.DataFrame) -> list:
    L = []
    for s in ("C8", "C1"):
        t = T[T["series"] == s]
        y = t[t["kind"] == "year"].iloc[0]
        mo = t[t["kind"] == "month"].sort_values("contrib_to_year", ascending=False).head(3)
        tk = t[t["kind"] == "ticker"].sort_values("d_vs_k1", ascending=False).head(3)
        L.append("- %s: Δ QLIKE к K1 за 2018 = %+.4f; средний вес на M %.2f против лучшего постоянного веса года %.2f. Больше всего проигрыш вносят месяцы %s; бумаги %s." % (
            s, y["d_vs_k1"], y["mean_w"], y["w_best_year"], ", ".join("%s (%+.4f; вес %.2f, лучший %.2f)" % (r.key, r.contrib_to_year, r.mean_w, r.w_best_period) for r in mo.itertuples()),
            ", ".join("%s (%+.3f)" % (r.key, r.d_vs_k1) for r in tk.itertuples())))
    return L


def answers(Y: pd.DataFrame) -> list:
    kg = Y[(Y["group"] == "M16") & (Y["m"] == "KG") & Y["slice"].str.startswith("Y")].set_index("slice")
    ys = lambda mask: ", ".join(i[1:] for i in kg.index[mask]) or "нет"
    both = (kg["share_both_under"] + kg["share_both_over"])
    return ["1. HAR-J лучше √(K2·K3) (D̄ < 0) в годы: %s." % ys((kg["d_mean"] < 0).to_numpy()),
            "2. ML охватывает HAR-J (интервал веса HAR-J включает 0) в годы: %s; HAR-J охватывает ML: %s." % (ys(kg["ml_encompasses_har"].astype(bool).to_numpy()), ys(kg["har_encompasses_ml"].astype(bool).to_numpy())),
            "3. Обе модели ошибаются в одну сторону (оба ниже + оба выше) чаще всего в %s (%s %% дней); меньше всего — в %s (%s %%)." % (both.idxmax()[1:], _p(100 * both.max()), both.idxmin()[1:], _p(100 * both.min())),
            "4. Выпуклая смесь может помочь больше всего в %s (O1 %s %%, O2 %s %% q лучшей одиночной); меньше всего — в %s (O1 %s %%)." % (kg["gain_o2_pct"].idxmax()[1:], _p(kg.loc[kg["gain_o2_pct"].idxmax(), "gain_o1_pct"]),
                                                                                                                  _p(kg["gain_o2_pct"].max()), kg["gain_o1_pct"].idxmin()[1:], _p(kg["gain_o1_pct"].min())),
            "5. Нужен выход за диапазон: доля дней вне [H; M] — от %s %% (%s) до %s %% (%s); выигрыш O3 к O2 до %s п.п. q." % (_p(100 * kg["share_outside"].min()), kg["share_outside"].idxmin()[1:], _p(100 * kg["share_outside"].max()),
                                                                                                                 kg["share_outside"].idxmax()[1:], _p(((kg["gain_o3_pct"] - kg["gain_o2_pct"]).max())))]


def build(Y: pd.DataFrame, F: pd.DataFrame, T: pd.DataFrame) -> str:
    L = [first_line(Y), "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; диагностика задним числом, описательно; в прогноз и выбор не входит; первая цифра — потолок.", "", "## Отступления и толкования (до выводов)"]
    L += ["- " + x for x in DEVIATIONS] + ["", "## Ответы на вопросы 1–5 (M16, M = √(K2·K3), по годам цели)", ""] + answers(Y)
    L += ["", "## Пункты 1–6 по годам и окнам (M16, M = √(K2·K3); D = QLIKE(H) − QLIKE(M), минус — HAR-J лучше; O1–O3 — выигрыш в % QLIKE лучшей одиночной)", ""] + year_table(Y)
    L += ["", "## Другие M и SBER (годы цели)", ""] + other_m(Y)
    sb = Y[(Y["group"] == "SBER") & (Y["m"] == "KG") & Y["slice"].isin(["Y%d" % y for y in range(2017, 2027)])]
    L += ["", "SBER, M = √(K2·K3): " + "; ".join("%s D̄ %+.4f, вес HAR-J %.2f [%.2f; %.2f]" % (r.slice[1:], r.d_mean, r.w_har, r.w_har_lo90, r.w_har_hi90) for r in sb.itertuples())]
    L += ["", "## 2022–2025 (диагностика, без подбора)", ""] + ["- " + x for x in _span(Y)]
    L += ["", "## Вопрос 6: признаки (DX_FEATURES.csv; квинтили по прошлому окну)", ""] + feat_lines(F)
    L += ["", "## Вопрос 7: разбор 2018 (DX_2018.csv)", ""] + d2018_lines(T)
    return "\n".join(L) + "\n"


def _span(Y: pd.DataFrame) -> list:
    kg = Y[(Y["group"] == "M16") & (Y["m"] == "KG")].set_index("slice")
    out = []
    for s in ("Y2022", "Y2023", "Y2024", "Y2025", "2022_2025"):
        r = kg.loc[s]
        out.append("%s: D̄ %+.4f, вес HAR-J %.2f [%.2f; %.2f], вне диапазона %s %% дней (%s %% потерь), O1/O2/O3 %s/%s/%s %%." % (
            s, r["d_mean"], r["w_har"], r["w_har_lo90"], r["w_har_hi90"], _p(100 * r["share_outside"]), _p(100 * r["loss_share_outside"]), _p(r["gain_o1_pct"]), _p(r["gain_o2_pct"]), _p(r["gain_o3_pct"])))
    return out
