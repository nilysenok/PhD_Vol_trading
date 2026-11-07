"""P3c, G13 (R15): численные ядра диагностики — γ (охват), ICC, R² поправки, корреляции, Спирмен, DM по датам. Чистые функции, без чтения файлов.
γ бумаги — `s8.hs_enc.wald` (Ж-212 этап A, не копия); панельная γ — тот же QLIKE-фит `hs_qfit.fit`, скоры суммируются по датам, ковариация — Ньюи–Уэст `hs_enc.nw_cov` по датам."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from models_v2.s8 import hs_enc, hs_qfit


def delta_x(s2_har, s2_base) -> np.ndarray:
    """X = [1, δ], δ = ln HAR − ln base (Ж-212 «const + δ»)."""
    d = np.log(np.asarray(s2_har, float)) - np.log(np.asarray(s2_base, float))
    return np.column_stack([np.ones(len(d)), d])


def gamma_ticker(rv, s2_har, s2_base) -> dict:
    """γ_δ, t, p бумаги (HAC лаг 5, строки упорядочены по дате)."""
    w = hs_enc.wald(rv, s2_base, delta_x(s2_har, s2_base))
    return {"gamma": float(w["gamma"][1]), "t": float(w["t"][1]), "p": float(w["p"][1]), "n": int(w["n"])}


def gamma_panel(rv, s2_har, s2_base, dates) -> dict:
    """Панельная γ_δ: общий QLIKE-фит по всем (бумага, дата); сэндвич: скоры по датам, V = A⁻¹ (T·Ω/n²) A⁻¹."""
    rv, base = np.asarray(rv, float), np.asarray(s2_base, float)
    X, off = delta_x(s2_har, s2_base), np.log(base)
    gam = hs_qfit.fit(X, rv, off)
    r = rv * np.exp(-(off + hs_qfit.lin(X, gam)))
    sc = pd.DataFrame(X * (1.0 - r)[:, None]).groupby(np.asarray(dates)).sum().to_numpy()
    n, T = len(rv), len(sc)
    A = np.einsum("ij,i,ik->jk", X, r, X) / n
    Ai = np.linalg.inv(A)
    V = Ai @ (hs_enc.nw_cov(sc) * T / n ** 2) @ Ai
    t = gam[1] / np.sqrt(V[1, 1])
    return {"gamma": float(gam[1]), "t": float(t), "p": float(2 * sps.norm.sf(abs(t))), "n": n}


def icc1(r, groups) -> dict:
    """ICC(1) по одностороннему ANOVA (неравные группы) и η² (доля дисперсии, объяснённая сдвигом группы)."""
    r, g = np.asarray(r, float), np.asarray(groups)
    s = pd.Series(r).groupby(g)
    n_i, m_i = s.size().to_numpy(float), s.mean().to_numpy()
    N, k, m = len(r), len(n_i), r.mean()
    if k < 2 or N <= k:
        return {"icc": float("nan"), "eta2": float("nan")}
    ssb = float((n_i * (m_i - m) ** 2).sum())
    ssw = float(((r - pd.Series(r).groupby(g).transform("mean").to_numpy()) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - float((n_i ** 2).sum()) / N) / (k - 1)
    return {"icc": float((msb - msw) / (msb + (n0 - 1) * msw)), "eta2": ssb / (ssb + ssw)}


def r2_correction(e, ghat) -> dict:
    """e = ln RV − ln K1; ĝ — применённая поправка. r2_mz = corr² (инвариантна к уровню); r2_oos = 1 − Σ(e − ĝ)²/Σe² (база — K1, т.е. ĝ = 0)."""
    e, g = np.asarray(e, float), np.asarray(ghat, float)
    c = float(np.corrcoef(e, g)[0, 1]) if np.std(g) > 0 and np.std(e) > 0 else float("nan")
    return {"r2_mz": c * c if np.isfinite(c) else float("nan"), "r2_oos": float(1 - ((e - g) ** 2).sum() / (e ** 2).sum())}


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


def spearman(x, y) -> float:
    return float(sps.spearmanr(x, y)[0]) if len(x) > 2 else float("nan")


def date_mean_t(d, dates) -> dict:
    """Среднее по бумагам за дату, затем среднее по датам и t (Ньюи–Уэст лаг 5); d — построчная разность потерь."""
    s = pd.Series(np.asarray(d, float)).groupby(np.asarray(dates)).mean().sort_index().to_numpy()
    if len(s) < 10:
        return {"mean": float("nan"), "t": float("nan"), "n_dates": len(s)}
    return {"mean": float(s.mean()), "t": float(s.mean() / np.sqrt(hs_enc.nw_cov(s[:, None])[0, 0] / len(s))), "n_dates": len(s)}


def qlike_rows(rv, s2) -> np.ndarray:
    r = np.asarray(rv, float) / np.asarray(s2, float)
    return r - np.log(r) - 1.0
