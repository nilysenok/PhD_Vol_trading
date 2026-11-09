"""V7-2b, G28–G29: оценка конфигураций 1–7 версии 2 (чистые функции от массивов; файлы не читаются, ничего не пишется). Метрики, Δ, DM, Холм, «go», стресс, MCS — `v7_1b_ev` (импорт, без копий).
Ряды на ОДИНАКОВЫХ датах: K1, K2, K3, M, K8, V4:NTP2, V6:A2, V6:A0, V6:A0g, конфигурации 1–7. Парные сравнения по гипотезам — `pairs_table`: Δ в % уровня K1 с 90 % интервалом, DM лаг 5 (p односторонний «X лучше Y»),
правило 6.2 (`v7_1m_stat.new_information`), стресс — без 1 % крайних дат пары (по |q_X − q_Y| на 2018–2021) и без 2018. Холм N = 733 / N_wide = 3166 — по окну, семейство = все пары таблицы."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EVX
from models_v2.s14 import v7_1b_ev as EV, v7_1m_stat as ST

CFGS = tuple("cfg%d" % i for i in range(1, 8))
CTRL = ("K1", "K2", "K3", "M", "K8", "V4:NTP2", "V6:A2", "V6:A0", "V6:A0g")
PAIRS = tuple([("H1", c, b) for c in CFGS[:3] for b in ("V6:A0", "V6:A0g")] + [("H2", "cfg6", "cfg4"), ("H3", "cfg5", "cfg6"), ("H3", "cfg7", "cfg6"), ("H3", "cfg5", "cfg7")]
              + [("4–7 против баз", c, b) for c in CFGS[3:] for b in ("V6:A0", "V6:A0g", "cfg1")])
PAIR_WIN = ("2018_2021", "2018_2021 без 1 % крайних", "2019_2021", "2022_2025")
YEARS = range(2018, 2027)
V1_WIN = ("2018_2021", "2019_2021", "2022_2025", "2026", "all")


def _stat(rv, fx, fy, k1, tp) -> dict:
    qx, qy = Q.ql(rv, fx), Q.ql(rv, fy)
    ax, ay = Q.annual(qx, tp, YEARS), Q.annual(qy, tp, YEARS)
    return dict(n=len(rv), delta_q=float((qx - qy).mean()), **EVX.delta_ci(qx, qy, float(Q.ql(rv, k1).mean()), b=ST.B, block=ST.BLOCK, seed=ST.SEED),
                p_one=EVX.dm_one(rv, fy, fx)["p_one"], years_neg=int(sum(ax[y] - ay[y] < 0 for y in ax)), n_years=len(ax))


def _masks(rv, fx, fy, tp) -> dict:
    """Окна пары: 2018–2021; то же без 1 % дат с наибольшим |q_X − q_Y|; 2019–2021; 2022–2025."""
    m1, q = ST.wmask(tp, "2018_2021"), Q.ql(rv, fx) - Q.ql(rv, fy)
    keep, idx = m1.copy(), np.flatnonzero(m1)
    keep[idx[np.argsort(-np.abs(q[idx]), kind="stable")[: int(round(EV.TRIM * len(idx)))]]] = False
    return {PAIR_WIN[0]: m1, PAIR_WIN[1]: keep, PAIR_WIN[2]: ST.wmask(tp, "2019_2021"), PAIR_WIN[3]: ST.wmask(tp, "2022_2025")}


def rule62(r: dict, win: str) -> str:
    """Правило 6.2 на окне разработки: Δ < 0, DM p < 0,05, Δ < 0 не менее чем в 3 годах (в окнах 2022+ — «—», правило к ним не применяется)."""
    if win == PAIR_WIN[3]:
        return "—"
    return "да" if (r["delta_q"] < 0 and r["p_one"] < 0.05 and r["years_neg"] >= 3) else "нет"


def pairs_table(rv, s: dict, tp, pairs=PAIRS) -> pd.DataFrame:
    rv, tp, rows = np.asarray(rv, float), np.asarray(tp), []
    for hyp, x, y in pairs:
        for win, m in _masks(rv, s[x], s[y], tp).items():
            r = _stat(rv[m], s[x][m], s[y][m], s["K1"][m], tp[m])
            rows.append(dict(hyp=hyp, x=x, y=y, window=win, rule62=rule62(r, win), **r))
    t = pd.DataFrame(rows)
    t["p_holm_N"], t["p_holm_Nwide"] = np.nan, np.nan
    for w in PAIR_WIN:
        k = t["window"] == w
        t.loc[k, "p_holm_N"], t.loc[k, "p_holm_Nwide"] = EVX.holm_both(t.loc[k, "p_one"].to_numpy(), EV.N_CP)
    return t


def v1_table(rv, s2: dict, s1: dict, tp) -> pd.DataFrame:
    """Версия 1 рядом с версией 2: QLIKE по окнам и Δ к K3 в % уровня K1 (интервал 90 %) у обеих версий."""
    rv, rows = np.asarray(rv, float), []
    for c in CFGS:
        for w in V1_WIN:
            m = ST.wmask(tp, w)
            q3, k1 = Q.ql(rv[m], s2["K3"][m]), float(Q.ql(rv[m], s2["K1"][m]).mean())
            row = dict(config=c, window=w)
            for tag, f in (("v1", s1[c]), ("v2", s2[c])):
                q = Q.ql(rv[m], f[m])
                ci = EVX.delta_ci(q, q3, k1, b=ST.B, block=ST.BLOCK, seed=ST.SEED)
                row.update({"qlike_" + tag: float(q.mean()), "d_k3_" + tag: ci["delta_pct"], "lo90_" + tag: ci["lo90"], "hi90_" + tag: ci["hi90"]})
            rows.append(row)
    return pd.DataFrame(rows)


def main_tables(rv, s: dict, tp) -> dict:
    """Таблицы оценки (кроме специализации): METRICS, MCS, TESTS, GO, YEARS, STRESS, PAIRS."""
    refs = {k: s[k] for k in EV.REFS}
    t = {"METRICS": EV.metrics_table(rv, s, tp), "MCS": EV.mcs_table(rv, s, tp), "PAIRS": pairs_table(rv, s, tp)}
    parts = {c: dict(tests=EV.tests_table(rv, s[c], refs, tp), go=EV.go_table(rv, s[c], refs, tp), yr=EV.years_table(rv, s[c], refs, tp)) for c in CFGS}
    for key, part in (("TESTS", "tests"), ("GO", "go"), ("YEARS", "yr")):
        t[key] = pd.concat([p[part].assign(config=c) for c, p in parts.items()], ignore_index=True)
    t["STRESS"] = EV.stress_table(rv, {k: s[k] for k in CTRL + CFGS}, tp)
    return t
