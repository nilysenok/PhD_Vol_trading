"""V6-4, G38: проверка «не сводится к режиму» (план п. 6). Δ D2 − K9 по терцилям H_t (пороги — по H закрытых пар пула дня d, `v6_4_an.tercile`), в % уровня K1 подмножества, 90 % интервал блокового бутстрапа, DM; корреляция ln M_D2 с ln H_t.
Выигрыш в терциле — Δ < 0 с 90 % интервалом целиком ниже 0 (формализация «выигрыша», окно 2017–2021). Вывод: выигрыш только в одном терциле — «эквивалент калибровки по уровню»."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_1_eval as EV

WINDOWS = ("2017_2021", "all")
EQ = "эквивалент калибровки по уровню"


def table(rv, f_d2, f_k9, tp, terc, h, m_d2) -> pd.DataFrame:
    """Строки: терцили 1…3 и «все» по окнам (Δ D2 − K9, DM «D2 лучше K9»), корреляция ln M_D2 и ln H_t по окнам."""
    rv, f_d2, f_k9, h, terc = (np.asarray(x, float) for x in (rv, f_d2, f_k9, h, terc))
    rows = []
    for win in WINDOWS:
        mw = Q.wmask(tp, win)
        for g in (1, 2, 3, 0):
            m = mw & ((terc == g) if g else True)
            qx, qy = Q.ql(rv[m], f_d2[m]), Q.ql(rv[m], f_k9[m])
            k1 = float(qy.mean())
            dm = EV.dm_one(rv[m], f_k9[m], f_d2[m])
            ci = EV.delta_ci(qx, qy, k1)
            rows.append(dict(section="терциль" if g else "все", window=win, group=g if g else "все", n=int(m.sum()), qlike_d2=float(qx.mean()), qlike_k9=k1, delta_pct_of_K9=ci["delta_pct"], lo90=ci["lo90"], hi90=ci["hi90"],
                             dm_p_one=dm["p_one"], gain=bool(ci["delta_pct"] < 0 and ci["hi90"] < 0), corr_lnM_lnH=np.nan, verdict=""))
        rows.append(dict(section="корреляция", window=win, group="все", n=int(mw.sum()), corr_lnM_lnH=float(np.corrcoef(np.log(np.asarray(m_d2, float)[mw]), np.log(h[mw]))[0, 1]), verdict=""))
    return pd.DataFrame(rows)


def verdict(t: pd.DataFrame) -> str:
    """Вывод по окну 2017–2021: Δ D2 − K9 ≥ 0 — проверка неприменима; выигрыш ровно в одном терциле — «эквивалент калибровки по уровню»; иначе — «нет»."""
    s = t[(t["window"] == "2017_2021")]
    allr = s[s["section"] == "все"].iloc[0]
    g = s[s["section"] == "терциль"]
    won = [int(x) for x in g.loc[g["gain"].astype(bool), "group"]]
    if allr["delta_pct_of_K9"] >= 0:
        return "D2 не лучше K9 в 2017–2021 (Δ %+.2f %% уровня K9): выигрыша нет, проверка терцилей неприменима" % allr["delta_pct_of_K9"]
    if len(won) == 1:
        return "%s: выигрыш только в терциле %d" % (EQ, won[0])
    if not won:
        return "выигрыш D2 против K9 не подтверждён интервалом ни в одном терциле (общий Δ %+.2f %% уровня K9): эквивалентом калибровки по уровню не признаётся, выигрыша нет" % allr["delta_pct_of_K9"]
    return "нет: выигрыш подтверждён интервалом в терцилях %s (не в одном)" % won


def with_verdict(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t.loc[t.index[-1], "verdict"] = verdict(t)
    return t
