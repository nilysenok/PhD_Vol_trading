"""V4-E: ужесточённое правило «go» (Ж-224 п. 10; GATES_V4.md, «Ужесточение go»). «go» — одновременно: (1) crit_panel ≤ порога B (0,26095); (2) crit_panel без целей 2017 ≤ min(K2, K3, D2 по нему же) − 0,0005;
(3) crit_panel без 1 % строк (бумага, дата) окна выбора M16 × 2017–2021 с наибольшей |QLIKE(ряд) − QLIKE(лучший из K2/K3 в этой строке)| ≤ min(K2, K3, D2 по нему же) − 0,0005 (отбрасывание — своё для каждого ряда).
Все ряды — на общих датах (S.common)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S

REF = ("k2", "k3", "d2")
TRIM = 0.01


def _ql(c: pd.DataFrame, s: str) -> np.ndarray:
    r = c["rv_tplus"].to_numpy(float) / c[s].to_numpy(float)
    return r - np.log(r) - 1.0


def crit_no2017(k: pd.DataFrame, cols: list) -> pd.Series:
    kk = k[k["t_plus"].dt.year >= 2018]
    return S.crit_panel(kk, cols).set_index("series")["crit_panel"]


def crit_trim(k: pd.DataFrame, cols: list, frac: float = TRIM) -> pd.Series:
    """crit_panel каждого ряда без доли `frac` строк окна выбора с наибольшей |QLIKE(ряд) − QLIKE(лучший из K2/K3)|."""
    c = S.common(k, list(cols))
    w = S.crit_window(c).reset_index(drop=True)
    best = np.minimum(_ql(w, "k2"), _ql(w, "k3"))
    out = {}
    for s in cols:
        d = np.abs(_ql(w, s) - best)
        drop = np.argsort(-d, kind="stable")[: int(round(frac * len(w)))]
        keep = np.ones(len(w), bool)
        keep[drop] = False
        out[s] = float(S.crit_panel(w[keep], [s]).iloc[0]["crit_panel"])
    return pd.Series(out)


def rules(k: pd.DataFrame, cols_e: list, crit: pd.Series, thr: float) -> dict:
    """-> {cfg: {'c1', 'c2', 'c3', 'go'}, 'best': лучшая E по crit_panel, 'tables': ...}. c2, c3 — с запасом MARGIN к min(K2, K3, D2) по тому же критерию."""
    allc = list(REF) + list(cols_e)
    n17, tr = crit_no2017(k, allc), crit_trim(k, allc)
    res = {}
    for e in cols_e:
        c1 = bool(crit[e] <= thr)
        c2 = bool(n17[e] <= min(n17[r] for r in REF) - S.MARGIN)
        c3 = bool(tr[e] <= min(tr[r] for r in REF) - S.MARGIN)
        res[e] = {"c1": c1, "c2": c2, "c3": c3, "go": c1 and c2 and c3, "crit_no2017": float(n17[e]), "crit_trim1": float(tr[e])}
    ref = {r: {"crit_no2017": float(n17[r]), "crit_trim1": float(tr[r])} for r in REF}
    return {"cfg": res, "ref": ref, "best": min(cols_e, key=lambda x: float(crit[x]))}
