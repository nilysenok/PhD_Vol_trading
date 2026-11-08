"""V5 лист FIN, G24: итоговый документ — блоки, ЦЕЛИКОМ собираемые из таблиц на диске (`FIN_*.csv`, `*_STRESS`, `*_LOTO`, `CRIT_*`, `DX1_*`, `CHECKPOINT.json`, `ALL_TRIALS.csv`). Ни одно результатное число не вписано руками:
десятичные числа текста — только из этих функций (проверка `v5_fin_docheck`). Формат: десятичная запятая, минус «−»; Δ — разность средних QLIKE по M16 в процентах уровня QLIKE K1 того же окна (минус = лучше)."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s12 import v5_0_io as IO, v5_fin_data as D

WN = {"2017": "2017", "2018_2021": "2018–2021", "2022": "2022", "2023_2025": "2023–2025", "2026": "2026", "2022_2025": "2022–2025", "2017_2021": "2017–2021", "all": "вся"}
REFS = ["k1", "k2", "k3", "d2", "K8"]
LOAD = {"all": "FIN_ALL", "tickers": "FIN_TICKERS", "years": "FIN_YEARS", "sber": "FIN_SBER", "tests": "FIN_TESTS", "mcs": "FIN_MCS", "joint": "FIN_JOINT", "enc": "FIN_ENCOMPASS", "abl": "FIN_ABLATION", "cost": "FIN_COST",
        "gs_stress": "GS_STRESS", "sp_stress": "SP_STRESS", "gs_loto": "GS_LOTO", "sp_loto": "SP_LOTO", "eqy": "CRIT_EQY", "audit": "CRIT_AUDIT", "mom": "DX1_MOMENTS", "jump": "DX1_JUMPS"}


def load() -> dict:
    t = {k: IO.read_csv(v + ".csv") for k, v in LOAD.items()}
    t["trials"] = pd.read_csv(IO.REP / "ALL_TRIALS.csv")
    t["cp"] = json.loads((IO.REP / "CHECKPOINT.json").read_text())
    t["gs"], t["sp"] = t["cp"]["FIN"]["best_gs"], t["cp"]["FIN"]["best_sp"]
    return t


def f(x: float, nd: int = 5) -> str:
    return ("%.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def s(x: float, nd: int = 2) -> str:
    return ("%+.*f" % (nd, x)).replace(".", ",").replace("-", "−")


def yn(v) -> str:
    return "—" if pd.isna(v) else ("да" if str(v).lower() == "true" else "нет")


def word(x: float) -> str:
    return "лучше" if x < 0 else "хуже"


def hdr(cols: list) -> str:
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"


def nm(x: str) -> str:
    return D.nm(x)


def m(t: dict) -> pd.DataFrame:
    return t["all"][t["all"]["section"] == "metrics"].set_index(["series", "window"])


def go_row(t: dict) -> pd.DataFrame:
    return t["all"][t["all"]["section"] == "go"].set_index("series")


def n_go(t: dict) -> int:
    g = go_row(t)
    return int(sum(str(g.loc[x, "go"]).lower() == "true" for x in D.GS + D.SP))


def crit_block(t: dict) -> str:
    g, a, rows = go_row(t), m(t), []
    for x in D.SER:
        w = "—" if x not in g.index else None
        c = [f(g.loc[x, "crit_panel"]), f(g.loc[x, "crit_no2017"]), f(g.loc[x, "crit_trim1"])] if w is None else [f(a.loc[(x, "all"), "crit_panel"]), "—", "—"]
        cond = [yn(g.loc[x, k]) for k in ("c1", "c2", "c3", "go")] if w is None and x in D.GS + D.SP else ["—"] * 4
        rows.append("| %s | %s | %s | %s |" % (nm(x), " | ".join(c), f(a.loc[(x, "all"), "EqY"]), " | ".join(cond)))
    thr = g.iloc[0]
    return hdr(["ряд", "crit_panel (≤ %s)" % f(thr["thr1"]), "без 2017 (≤ %s)" % f(thr["thr2"]), "без 1 %% (≤ %s)" % f(thr["thr3"]), "EqY (диагностика)", "(1)", "(2)", "(3)", "go"]) + "\n".join(rows) + "\n"


def first_line(t: dict) -> str:
    g, b1, b2 = go_row(t), t["gs"], t["sp"]
    if n_go(t) != 0:
        raise ValueError("первая строка документа написана для результата «0 go»")
    v = lambda x, c: f(g.loc[x, c])
    return ("%d из %d конфигураций прошли «go» (три условия против порогов с K8 в планке); устойчивого преимущества над XGBoost (K2), LightGBM (K3) и над K8 нет. Лучшие по crit_panel: %s %s и %s %s против K8 %s, D2 %s, K1 %s, K2 %s, K3 %s (порог %s); "
            "у %s без 1 %% крайних дат %s против порога %s (условие 3 нет), у %s без 2017 %s против порога %s и без 1 %% %s против %s; выигрыш %s держится на 2018, выигрыш %s — на 2022."
            % (n_go(t), len(D.GS + D.SP), b2, v(b2, "crit_panel"), b1, v(b1, "crit_panel"), v("K8", "crit_panel"), v("d2", "crit_panel"), v("k1", "crit_panel"), v("k2", "crit_panel"), v("k3", "crit_panel"), v(b2, "thr1"),
               b2, v(b2, "crit_trim1"), v(b2, "thr3"), b1, v(b1, "crit_no2017"), v(b1, "thr2"), v(b1, "crit_trim1"), v(b1, "thr3"), b2, b1))


def delta_block(t: dict, best: str) -> str:
    a = m(t)
    rows = ["| %s | %s | %s |" % (WN[w], f(a.loc[(best, w), "qlike"], 4), " | ".join(s(a.loc[(best, w), "d_qlike_%s_pct_of_k1" % r]) for r in REFS)) for w in ("all", "2017", "2018_2021", "2022", "2023_2025", "2022_2025", "2026")]
    return hdr(["окно", "QLIKE %s (M16)" % best] + ["Δ к %s, %%" % nm(r) for r in REFS]) + "\n".join(rows) + "\n"


def levels_block(t: dict) -> str:
    a, cols = m(t), ["k1", "k2", "k3", "M", "d2", "K8", "GS4", "SP1"]
    rows = ["| %s | %s |" % (WN[w], " | ".join(f(a.loc[(x, w), "qlike"], 4) for x in cols)) for w in ("2017", "2018_2021", "2022", "2023_2025", "2026", "2017_2021", "2022_2025", "all")]
    return hdr(["окно (QLIKE M16)"] + [nm(x) for x in cols]) + "\n".join(rows) + "\n"


def tickers_block(t: dict, w: str) -> str:
    d = t["tickers"]
    d = d[(d["window"] == w) & (d["ticker"] != "COUNT")]
    p = d.pivot(index="ticker", columns="series", values="qlike")[D.SER]
    p.loc["M16 (среднее)"] = d[d["group"] == "M16"].pivot(index="ticker", columns="series", values="qlike")[D.SER].mean()
    return hdr(["бумага"] + [nm(x) for x in D.SER]) + "".join("| %s | %s |\n" % (i, " | ".join(f(v, 4) for v in r)) for i, r in p.iterrows())


def counts_block(t: dict) -> str:
    d = t["tickers"]
    d = d[d["ticker"] == "COUNT"].set_index(["series", "window"])
    ws = ("2017_2021", "2022_2025", "all")
    rows = ["| %s | %s |" % (x, " | ".join("%d / %d" % (d.loc[(x, w), "n_improve_vs_K8"], d.loc[(x, w), "n_improve_vs_best_k2k3"]) for w in ws)) for x in D.GS + D.SP]
    return hdr(["конфигурация"] + ["%s: к K8 / к лучшему из K2, K3 (из %d)" % (WN[w], d.loc[("GS4", w), "n_tickers"]) for w in ws]) + "\n".join(rows) + "\n"


def sber_block(t: dict) -> str:
    b = t["sber"].set_index(["series", "window"])
    ref = float(b.loc[("k1", "all"), "ref_a1_roll_126"])
    order = sorted(D.SER, key=lambda x: (float(b.loc[(x, "all"), "qlike"]), x))
    return hdr(["ряд", "QLIKE вся (%d дат)" % b.loc[("k1", "all"), "n"], "Δ к ориентиру %s" % f(ref, 4), "QLIKE 2017–2021", "QLIKE 2022–2025"]) + "".join(
        "| %s | %s | %s | %s | %s |\n" % (nm(x), f(b.loc[(x, "all"), "qlike"], 4), s(b.loc[(x, "all"), "qlike"] - ref, 4), f(b.loc[(x, "2017_2021"), "qlike"], 4), f(b.loc[(x, "2022_2025"), "qlike"], 4)) for x in order)


def years_block(t: dict) -> str:
    p = t["years"].pivot(index="year", columns="series", values="qlike_m16")[D.SER]
    return hdr(["год цели"] + [nm(x) for x in D.SER]) + "".join("| %d | %s |\n" % (i, " | ".join(f(v, 4) for v in r)) for i, r in p.iterrows())
