"""T0: текст `P0_TABLE.md` — генератор из таблиц (числа не вписываются руками). Ответы А / Б / В — по правилам, записанным в коде."""
from __future__ import annotations

import pandas as pd

from models_v2.s16 import x1_t0 as X

STATUS = "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок. Ж-242 пп. 1–2, лист T0. Обучения нет, использованы сохранённые прогнозы."


def tab(head, rows) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f6(x) -> str:
    return "%.6f" % x


def pct(x) -> str:
    return "%+.2f %%" % x


def avail(t: pd.DataFrame) -> list:
    r0 = t.iloc[0]
    has, no = list(t.loc[t["has_2017"], "series"]), list(t.loc[~t["has_2017"], "series"])
    L = ["## 1. Доступность рядов (строки с конечными RV и F > 0)", "",
         tab(["ряд", "d с", "d по", "t⁺ с", "t⁺ по", "дат", "строк в файле", "t⁺ в 2017"], [(r.series, r.avail_first_d, r.avail_last_d, r.avail_first_tplus, r.avail_last_tplus, r.avail_n, r.raw_rows, r.n_2017) for r in t.itertuples()]), "",
         "- 2017 (по году t⁺) есть у: %s." % ", ".join(has), "- 2017 нет у: %s." % (", ".join(no) or "—"),
         "- **Общий период** (пересечение дат d всех рядов): d с %s по %s; t⁺ с %s по %s, %d дат; он %s2017. Период 2018–2025 нельзя называть периодом 2017–2025." % (
             r0.common_first_d, r0.common_last_d, r0.first_all, r0.last_all, r0.n_all, "включает " if r0.first_all[:4] == "2017" else "не включает "),
         "- Строк в файле больше, чем дат с конечными RV и F > 0, там, где в файле есть нечисловая строка (не заполняется).",
         "- Цель (RV дня t⁺) и t⁺ сверены: на общих датах одинаковы у всех файлов (иначе генератор останавливается).", ""]
    return L


def qlike_part(t: pd.DataFrame) -> list:
    L = ["## 2. QLIKE (среднее дневных l = RV/F − ln(RV/F) − 1 на общих датах; меньше — лучше)", ""]
    for w, name in (("all", "Полный общий период"), ("2018_2021", "2018–2021 (окно разработки)"), ("2022_2025", "2022–2025 (историческая проверка, не OOS)"), ("2026", "2026 (доступная часть)")):
        L += ["**%s**" % name, "", tab(["ряд", "QLIKE", "t⁺ с", "t⁺ по", "n"], [(r.label, f6(getattr(r, "q_" + w)), getattr(r, "first_" + w), getattr(r, "last_" + w), getattr(r, "n_" + w)) for r in t.itertuples()]), ""]
    L += ["**По годам t⁺**", "", tab(["ряд"] + ["%d (n=%d)" % (y, t["n_%d" % y].iloc[0]) for y in X.YEARS], [[r.series] + [f6(getattr(r, "q_%d" % y)) for y in X.YEARS] for r in t.itertuples()]), "",
          "MCS 90 %% на полном периоде: %s." % ", ".join("%s (p=%.3f)" % (r.series, r.mcs_p_all) for r in t.itertuples() if r.mcs90_all), ""]
    return L


def year_rows(T, best: str, b: str) -> list:
    nall = T.at[best, "n_all"]
    return [(y, T.at[best, "q_%d" % y] - T.at[b, "q_%d" % y], 100 * (T.at[best, "q_%d" % y] - T.at[b, "q_%d" % y]) / T.at[b, "q_%d" % y], T.at[best, "n_%d" % y] * (T.at[best, "q_%d" % y] - T.at[b, "q_%d" % y]) / nall) for y in range(2018, 2026)]


def answers(t: pd.DataFrame, s: pd.DataFrame) -> list:
    T = t.set_index("series")
    order = T.loc[list(X.HYB), "q_all"].sort_values()
    best = order.index[0]
    L = ["## 3. Ответы", "", "**А. Численно лучший гибрид на полном общем периоде (t⁺ %s – %s, n = %d).** %s, QLIKE %s. Порядок гибридов: %s." % (
        T.at[best, "first_all"], T.at[best, "last_all"], T.at[best, "n_all"], T.at[best, "label"], f6(order.iloc[0]), "; ".join("%s %s" % (k, f6(v)) for k, v in order.items()))]
    if best in ("V7:cfg8", "X1"):
        L.append("Оговорка: cfg8 выбрана после просмотра по QLIKE 2018–2021 (X1 построен на ней), поэтому цифра — потолок.")
    bases = [b for b in ("K2", "K3", "V4:NTP2") if b != best]
    L += ["", "**Б. Устойчивость по годам (%s; Δ = QLIKE гибрида − QLIKE базы, в %% QLIKE базы того же года).**" % best, ""]
    for b in bases:
        yr, full = year_rows(T, best, b), T.at[best, "q_all"] - T.at[b, "q_all"]
        worst = max(yr, key=lambda r: r[1])
        d26 = T.at[best, "q_2026"] - T.at[b, "q_2026"]
        L.append("- против %s: лучше в %d из 8 полных лет 2018–2025; худший год %d: Δ %s (%s QLIKE), его вклад в полный Δ %+.6f при полном Δ %+.6f; 2026 (n=%d, неполный): Δ %+.6f." % (
            b, sum(r[1] < 0 for r in yr), worst[0], pct(worst[2]), f6(worst[1]), worst[3], full, T.at[best, "n_2026"], d26))
    L += ["", "**В. Статистика с учётом поиска (полный период; Холм N = %d, N_wide = %d; DM лаг 5, односторонний «гибрид лучше»; 90 %% бутстрап-интервал блок 20, 2000, seed 0).**" % (X.N_HOLM, X.N_WIDE), ""]
    S = s[(s["window"] == "all") & (s["hybrid"] == best)].set_index("base")
    L.append(tab(["база", "QLIKE базы (знаменатель)", "Δ % базы", "90 % интервал", "p DM", "p Холма N", "p Холма N_wide"],
                 [(b, f6(S.at[b, "base_qlike"]), pct(S.at[b, "delta_pct_of_base"]), "[%+.2f; %+.2f]" % (S.at[b, "lo90"], S.at[b, "hi90"]), "%.4f" % S.at[b, "p_one"], "%.4f" % S.at[b, "p_holm_N"], "%.4f" % S.at[b, "p_holm_Nwide"]) for b in S.index]))
    ok = all(S.at[b, "p_holm_N"] < 0.05 for b in bases)
    L += ["", "Δ < 0 — гибрид лучше базы. p DM без поправки против %s: %s." % (", ".join(bases), ", ".join("%.4f" % S.at[b, "p_one"] for b in bases)), ("Преимущество %s над %s значимо с поправкой Холма (N = %d)." if ok else "Преимущество %s над %s НЕ подтверждено с поправкой Холма (N = %d): это отсутствие доказательства при поиске, а не отсутствие численного улучшения; численное улучшение — не доказанное превосходство.") % (best, ", ".join(bases), X.N_HOLM), ""]
    return L


def build(t: pd.DataFrame, s: pd.DataFrame) -> str:
    L = ["# P0_TABLE — единая сравнительная таблица QLIKE, SBER, h = 1 (лист T0, Ж-242)", "", STATUS, "",
         "Таблицы: `P0_TABLE.csv` (ряды × окна, годы, MCS), `P0_STATS.csv` (Δ с бутстрапом, DM, Холм по каждой паре «гибрид — база»). Генератор: `models_v2/s16/x1_t0.py`.", ""]
    return "\n".join(L + avail(t) + qlike_part(t) + answers(t, s)) + "\n"
