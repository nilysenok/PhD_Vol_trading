"""V7-1b-C: текст `LEAF1B_REPORT.md`. Каждое число берётся из таблиц оценки (`t`), E0 и CHECKPOINT; в шаблоне чисел нет (тест: изменение ячейки таблицы меняет текст; подмена числа в файле ≠ render)."""
from __future__ import annotations

import pandas as pd

WIN4 = ("2018_2021", "2019_2021", "2022_2025", "2026")
SHOW = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g", "cfg1", "cfg2", "cfg3")
YEAR_SHOW = ("cfg1", "cfg2", "cfg3", "K1", "K2", "K3", "V6:A2", "V4:NTP2")
DBASE = ("V6:A0", "V6:A0g", "K2", "K3", "V4:NTP2", "V6:A2")
NAME = {"cfg1": "1 (TCN-w5)", "cfg2": "2 (CNN-w5)", "cfg3": "3 (w1)"}


def verdict(is_go: bool, confirmed: bool) -> str:
    if not is_go:
        return "go не выполнен"
    return "go, подтверждено (p Холма < 0,05)" if confirmed else "статистически неподтверждённое историческое улучшение"


def f(x, nd=4) -> str:
    return ("%." + str(nd) + "f") % x


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def q_table(m: pd.DataFrame, wins, series=SHOW) -> pd.DataFrame:
    p = m.pivot(index="series", columns="window", values="qlike")
    return pd.DataFrame([[s] + [f(p.loc[s, w], 6) for w in wins] for s in series], columns=["ряд"] + list(wins))


def delta_table(tt: pd.DataFrame, cfg: str) -> pd.DataFrame:
    rows = []
    for w in ("2018_2021", "2022_2025"):
        for b in DBASE:
            r = tt[(tt["config"] == cfg) & (tt["window"] == w) & (tt["base"] == b)].iloc[0]
            rows.append([w, b, f(r["delta_pct"], 2), "[%s; %s]" % (f(r["lo90"], 2), f(r["hi90"], 2)), f(r["p_one"], 4), f(r["p_holm_N"], 4), f(r["p_holm_Nwide"], 4)])
    return pd.DataFrame(rows, columns=["окно", "база", "Δ, % K1", "90 % интервал", "DM p", "Холм N=733", "Холм N=3166"])


def render(t: dict, e0: pd.DataFrame, inp: dict, cp: dict) -> str:
    e = e0.groupby(["kind", "component"])["r2_weighted"].mean().unstack()
    ch, ni = cp["choice3"], t["NEWINFO"]
    L = ["# LEAF1B — лист 1b V7: конфигурации 1–3 (SBER, h = 1)", "",
         "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; все цифры — потолок (первый счёт, дизайн K2/K3 выбирался на SBER 2017–2021). Расходуется 3 конфигурации из 11. Каждое число — из таблиц `1b_*.csv` и CHECKPOINT ключ `1b`.", "",
         "Одинаковые даты: %d, %s … %s." % (len(inp["idx"]), inp["idx"][0].date(), inp["idx"][-1].date()),
         "Выбор энкодера для 3 (записан до счёта 3): %s; QLIKE 2018–2021 конфигураций 1 и 2 — %s и %s." % (ch["choice"], f(ch["qlike"]["tcn"], 6), f(ch["qlike"]["cnn"], 6)), "",
         "## 1. E0 (частный R² компонент z, среднее по seeds; 2018–2021)", "",
         md(pd.DataFrame([[k] + [f(e.loc[k, c], 3) for c in e.columns] for k in e.index], columns=["энкодер"] + ["z%d" % c for c in e.columns])), "",
         "Новая информация есть у обоих энкодеров (порог 0,9): %s." % ("нет, стоп" if cp["E0"]["both_no_new_info"] else "да, конфигурации 4–11 допущены"), "",
         "## 2. QLIKE по окнам (одинаковые даты)", "", md(q_table(t["METRICS"], WIN4)), ""]
    for c in ("cfg1", "cfg2", "cfg3"):
        L += ["## 3.%s. Конфигурация %s: Δ (QLIKE конфигурации − базы) в %% уровня K1 окна" % (c[-1], NAME[c]), "", md(delta_table(t["TESTS"], c)), ""]
    L += ["## 4. Правило 6.2 «новая информация» (2018–2021)", "", md(ni.assign(delta=ni["delta"].map(lambda x: f(x, 6)), p_dm_one=ni["p_dm_one"].map(f))[["config", "base", "delta", "p_dm_one", "years_neg", "new_information"]]), ""]
    g = t["GO"]
    L += ["## 5. «Go» 6.1 (шесть условий)", "", md(g.assign(value=g["value"].map(lambda x: f(x, 6)), threshold=g["threshold"].map(lambda x: f(x, 4)))[["config", "cond", "name", "value", "threshold", "passed"]]), ""]
    L += ["Значение условия 5 «inf» означает: суммарный выигрыш над лучшим из K2/K3 за 2018–2021 не положителен, условие не выполнено.", "",
          "Статусы: " + "; ".join("%s — %s" % (NAME[c], cp["rules"][c]["status"]) for c in ("cfg1", "cfg2", "cfg3")), "",
          "## 6. QLIKE по годам", "", md(q_table(t["METRICS"], tuple(str(y) for y in range(2018, 2026)) + ("2026",), YEAR_SHOW)), "",
          "## 7. Стресс", "", md(t["STRESS"][t["STRESS"]["series"].isin(["K2", "K3", "V6:A2", "cfg1", "cfg2", "cfg3"])].assign(qlike=lambda d: d["qlike"].map(lambda x: f(x, 6)), gap_best_k2k3=lambda d: d["gap_best_k2k3"].map(lambda x: f(x, 6)))), "",
          "## 8. MCS 90 %", "", md(t["MCS"][t["MCS"]["in_mcs"]].assign(mean_qlike=lambda d: d["mean_qlike"].map(lambda x: f(x, 6)), p=lambda d: d["p"].map(f))[["window", "series", "mean_qlike", "p"]]), "",
          "Холм с N = 733 и N_wide = 3166. Полные таблицы — `1b_METRICS.csv`, `1b_TESTS.csv`, `1b_GO.csv`, `1b_NEWINFO.csv`, `1b_MCS.csv`, `1b_STRESS.csv`, `1b_YEARS.csv`."]
    return "\n".join(L) + "\n"
