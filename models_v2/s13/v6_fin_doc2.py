"""V6 лист FIN, G52: блоки итогового документа 5–9 (конфигурации и метрики, QLIKE по годам, сравнения и p, абляции, стресс). Все числа — из таблиц на диске."""
from __future__ import annotations

import pandas as pd

from models_v2.s13 import v6_fin_data as D
from models_v2.s13.v6_fin_doc import BASES, WN, f, hdr, iv, m, nm, s, yn

STATUS_CFG = {"A3": "по правилу (A2 выполнила п. 2.2)", "C4": "по правилу (C2 выполнила п. 2.2)", "E3": "по правилу (E2 выполнила п. 2.2)", "A0g": "диагностический контроль (Ж-230); ОТСТУПЛЕНИЕ от предрегистрации: на месте CB2 (вопрос 1)"}


def configs_block(t: dict) -> str:
    tr = t["trials"].set_index("key")
    rows = []
    for c in D.CFG:
        r = tr.loc["Ж-227:" + c]
        rows.append("| %s | %s | %s | %s | %s |" % (c, r["direction"], r["config"].split('"features": ')[1].split(', "learner"')[0], r["config"].split('"learner": ')[1].split(", ")[0], STATUS_CFG.get(c, "посчитана")))
    return hdr(["id", "направление", "признаки", "обучатель", "статус"]) + "\n".join(rows) + "\n"


def metrics_block(t: dict, win: str) -> str:
    a = m(t)
    rows = ["| %s | %s |" % (nm(x), " | ".join(f(a.loc[(x, win), c]) for c in ("qlike", "rmse", "mae", "r2_rv", "r2_ln", "bias_ln"))) for x in D.SER]
    return hdr(["ряд (n = %d)" % a.loc[("K1", win), "n"], "QLIKE", "RMSE", "MAE", "R² (RV)", "R² (ln)", "bias (ln F/RV)"]) + "\n".join(rows) + "\n"


def go_block(t: dict) -> str:
    g = t["go"]
    rows = ["| %s | %s | %d из 5 |" % (c, " | ".join(yn(g[(g["config"] == c) & (g["cond"] == i)]["passed"].iloc[0]) for i in range(1, 6)), int(g[g["config"] == c]["passed"].astype(str).str.lower().eq("true").sum())) for c in D.CFG]
    return hdr(["конфигурация", "(1) crit_SBER", "(2) без 2017", "(3) без 2017–2018", "(4) без 1 %", "(5) лет лучше K2/K3", "выполнено"]) + "\n".join(rows) + "\n"


def years_block(t: dict) -> str:
    y = t["years"].set_index("year")[D.SER]
    return hdr(["ряд"] + [str(i) for i in y.index]) + "".join("| %s | %s |\n" % (nm(c), " | ".join(f(v) for v in y[c])) for c in D.SER)


def delta_win(t: dict, win: str) -> str:
    d = t["tests"]
    d = d[(d["section"] == "delta_dm") & (d["window"] == win)].set_index(["config", "base"])
    rows = ["| %s | %s |" % (c, " | ".join(iv(d.loc[(c, b)]) for b in BASES)) for c in D.CFG]
    return hdr(["конфигурация (%s)" % WN[win]] + ["Δ к %s, %%" % b for b in BASES]) + "\n".join(rows) + "\n"


def p_win(t: dict, win: str) -> str:
    d = t["tests"]
    d = d[(d["section"] == "delta_dm") & (d["window"] == win)].set_index(["config", "base"])
    cell = lambda r: "%s / %s / %s" % (f(r.p_one, 3), f(r.p_holm_N, 3), f(r.p_holm_Nwide, 3))
    rows = ["| %s | %s |" % (c, " | ".join(cell(d.loc[(c, b)]) for b in BASES)) for c in D.CFG]
    return hdr(["конфигурация (%s)" % WN[win]] + ["p DM / Холм N = 722 / Холм N = 3122: %s" % b for b in BASES]) + "\n".join(rows) + "\n"


def info_block(t: dict, src: list) -> str:
    d = t["info"]
    d = d[(d["window"] == "2017_2021") & d["src"].isin(src)]
    rows = ["| %s | %s | %s | %s | %d из %d | %s %s %s | %s | %s |" % (r.pair, f(r.delta_qlike, 5), iv(r), f(r.dm_p_one, 3), r.years_neg, r.years_n, yn(r.cond_i), yn(r.cond_ii), yn(r.cond_iii), yn(r.new_information), f(r.p_holm_N, 3)) for r in d.itertuples()]
    return hdr(["пара x против y", "Δ QLIKE", "Δ, % уровня K1 [90 %]", "DM p (односторонний)", "лет с Δ < 0", "(i) (ii) (iii)", "несёт новую информацию", "p Холма N = 722"]) + "\n".join(rows) + "\n"


def versions_block(t: dict) -> str:
    d = t["info"]
    d = d[(d["window"] == "2017_2021") & d["pair"].str.match(r"^(A2|C2|C4|E2) vs A0g?$")]
    rows = ["| %s | %s | %s | %d из %d | %s |" % (r.pair, iv(r), f(r.dm_p_one, 3), r.years_neg, r.years_n, yn(r.new_information)) for r in d.sort_values(["x", "y"]).itertuples()]
    return hdr(["пара (обе версии: против A0 и против A0g)", "Δ, % уровня K1 [90 %]", "DM p", "лет с Δ < 0", "несёт новую информацию"]) + "\n".join(rows) + "\n"


def decomp_block(t: dict) -> str:
    d = t["abl"]
    d = d[d["part"] != "DX-A"]
    rows = ["| %s | %s − %s | %s | %s | %s |" % (r.part, r.x, r.y, WN[r.window], iv(r), f(r.dm_p_one, 3)) for r in d.itertuples()]
    return hdr(["часть", "x − y", "окно", "Δ, % уровня K1 [90 %]", "DM p"]) + "\n".join(rows) + "\n"


def dxa_block(t: dict) -> str:
    d = t["abl"]
    d = d[d["part"] == "DX-A"]
    return hdr(["модель", "тест", "df", "статистика", "p"]) + "".join("| %s | %s | %s | %s | %s |\n" % (r.model, r.test, "—" if pd.isna(r.df) else int(r.df), f(r.stat, 3), f(r.p, 4)) for r in d.itertuples())


def stress_block(t: dict, col: str) -> str:
    d = t["stress"]
    d = d[d["kind"] == "stress"]
    sc = list(dict.fromkeys(d["scenario"]))
    p = d.pivot(index="config", columns="scenario", values=col).reindex(D.CFG)[sc]
    return hdr(["конфигурация"] + sc) + "".join("| %s | %s |\n" % (i, " | ".join(s(v) for v in r)) for i, r in p.iterrows())


def sens_block(t: dict) -> str:
    d = t["stress"]
    d = d[d["kind"] == "sens"]
    return hdr(["конфигурация", "соседняя точка", "QLIKE вся", "Δ к K1 вся, %", "QLIKE 2017–2021", "Δ к K1 2017–2021, %"]) + "".join(
        "| %s | %s | %s | %s | %s | %s |\n" % (r.config, r.scenario, f(r.qlike_all), s(r.d_pct_K1_all), f(r.qlike_2017_2021), s(r.d_pct_K1_2017_2021)) for r in d.itertuples())


def unst_block(t: dict) -> str:
    d = t["unst"]
    d = d[(d["window"] == "2017_2021") & (d["base"] == "A0")]
    return hdr(["конфигурация против A0", "группа дней", "дней", "Δ, % уровня K1 [90 %]"]) + "".join(
        "| %s | %s | %d | %s [%s; %s] |\n" % (r.config, r.group, r.n_days, s(r.delta_pct), s(r.lo90), s(r.hi90)) for r in d.itertuples())
