"""V7 лист FIN: пересчитываемые таблицы FIN_ALL, FIN_YEARS, FIN_TESTS, FIN_MCS, FIN_STRESS, FIN_AFTER — из СОХРАНЁННЫХ прогнозов 18 рядов на одинаковых датах. Определения прежние (функции листов 1b–4 без правок):
метрики `v7_1b_ev.metrics_table`, Δ в % уровня K1 с 90 % интервалом (блок 20, 2000, seed 0), DM (НУ 5), Холм N = 733 / 3166, MCS 90 %, стресс. FIN_AFTER — окна после 2021 (Ж-232 п. 11).
Кризисные дни: 5 % дат окна 2022–2025 с наибольшим прогнозом K1 (известен заранее) — правило объявлено в этом листе до просмотра, не подбиралось (отступление FIN, раздел 13); календарное окно февраль–апрель 2022 не годится: торги акциями были остановлены."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q
from models_v2.s14 import v7_1b_ev as EV, v7_fin_data as D

YEAR_W = [str(y) for y in range(2018, 2027)]
CRISIS_SHARE = 0.05
SHORT = {"K2": "K2", "K3": "K3", "V4:NTP2": "NTP2", "V6:A2": "A2"}


def load() -> dict:
    from models_v2.s14 import v7_4_evrun as R
    inp = R.load()
    return dict(inp=inp, s={k: inp["series"][k] for k in D.SER}, rv=inp["rv"], tp=inp["tp"])


def metrics(d: dict) -> pd.DataFrame:
    return EV.metrics_table(d["rv"], d["s"], d["tp"])


def years(allm: pd.DataFrame) -> pd.DataFrame:
    y = allm[allm["window"].isin(YEAR_W)].pivot(index="window", columns="series", values="qlike")[D.SER]
    y.index = y.index.astype(int)
    return y.rename_axis("year").rename_axis(columns=None).reset_index()


def tests(d: dict) -> pd.DataFrame:
    refs = {k: d["s"][k] for k in EV.REFS}
    return pd.concat([EV.tests_table(d["rv"], d["s"][c], refs, d["tp"]).assign(config=c) for c in D.CFG], ignore_index=True)


def after(d: dict) -> pd.DataFrame:
    rv, ti = d["rv"], pd.DatetimeIndex(d["tp"])
    q = {k: Q.ql(rv, v) for k, v in d["s"].items()}
    m = {"2018_2021": (ti.year >= 2018) & (ti.year <= 2021), "2022_2025": (ti.year >= 2022) & (ti.year <= 2025), "2022": ti.year == 2022, "2023_2025": (ti.year >= 2023) & (ti.year <= 2025)}
    k1 = np.asarray(d["s"]["K1"], float)
    m["crisis"] = m["2022_2025"] & (k1 >= np.quantile(k1[m["2022_2025"]], 1 - CRISIS_SHARE))
    rows = []
    for k in D.SER:
        r = dict(series=k, n_2018_2021=int(m["2018_2021"].sum()), n_2022_2025=int(m["2022_2025"].sum()), n_crisis=int(m["crisis"].sum()))
        r.update({"qlike_" + w: float(q[k][mm].mean()) for w, mm in m.items()})
        r["ratio_2022_2025_to_2018_2021"] = r["qlike_2022_2025"] / r["qlike_2018_2021"]
        for b, sb in SHORT.items():
            for w in ("2018_2021", "2022_2025"):
                r["share_better_%s_%s" % (sb, w)] = float((q[k][m[w]] < q[b][m[w]]).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def tables(d: dict) -> dict:
    allm = metrics(d)
    return {"FIN_ALL": allm, "FIN_YEARS": years(allm), "FIN_TESTS": tests(d), "FIN_MCS": EV.mcs_table(d["rv"], d["s"], d["tp"]), "FIN_STRESS": EV.stress_table(d["rv"], d["s"], d["tp"]), "FIN_AFTER": after(d)}


def all_tables(d: dict) -> dict:
    return {**tables(d), **D.sheets()}


def best(t: dict, win: str) -> str:
    a = t["FIN_ALL"]
    a = a[(a["window"] == win) & a["series"].isin(D.CFG)]
    return str(a.loc[a["qlike"].idxmin(), "series"])


def advantage(t: dict) -> list:
    """Конфигурации, у которых на 2018–2021 интервал Δ против K2, K3, NTP2 и A2 целиком ниже нуля (hi90 < 0)."""
    x = t["FIN_TESTS"]
    x = x[(x["window"] == "2018_2021") & x["base"].isin(D.BASES)]
    return sorted(c for c, g in x.groupby("config") if len(g) == len(D.BASES) and bool((g["hi90"] < 0).all()))
