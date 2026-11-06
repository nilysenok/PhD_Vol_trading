"""Ж-214, V7, J3 `cj_logit`: ПРОГНОЗ будущего скачка P(J_{t+1} > 0) логистикой с ridge (У14 GATES_V2). Признаки строки d (известны к 18:50 d):
[ln(1+J_d), ln(1+J_w), ln(1+J_m), ln RV_d, ln RV_w, ln RV_m], стандартизация по обучающим парам, ridge λ = 1,0 (свободный член не штрафуется), Ньютон <= 25 итераций;
пары t+1 <= d, один сегмент, расширяющееся окно. Метрики — к базовой частоте (доля скачков среди закрытых пар). Диагностика СОСТОЯВШИХСЯ скачков — отдельная функция `diagnostic`, в прогноз не входит."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s9 import hv2_dec as DC, hv2_world as W

LAM, ITERS = 1.0, 25


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_logit(Z: np.ndarray, y: np.ndarray, lam: float = LAM, iters: int = ITERS) -> np.ndarray:
    """Логистическая регрессия с ridge на столбцы Z (без константы; константа первой и не штрафуется). -> [b0, b...]."""
    A = np.column_stack([np.ones(len(y)), Z])
    pen = np.r_[0.0, np.full(Z.shape[1], lam)]
    b = np.zeros(A.shape[1])
    b[0] = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6) / (1 - np.clip(y.mean(), 1e-6, 1 - 1e-6)))
    for _ in range(iters):
        p = sigmoid(np.einsum("ij,j->i", A, b))
        g = np.einsum("ij,i->j", A, p - y) + pen * b
        Hs = np.einsum("ij,i,ik->jk", A, p * (1 - p), A) + np.diag(pen) + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(Hs, g)
        b -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b


def features(w: DC.DWorld) -> np.ndarray:
    return np.column_stack([DC.own(w.comp["j"], w.p.seg, True), w.p.X[:, :3]])


def run(w: DC.DWorld, extra: int = 0) -> tuple:
    """-> (p̂ на сетке, базовая частота по закрытым парам, число пар). `extra` > 0 — МУТАНТ «пары, закрытые +extra» (будущие цели). NaN, если признаки d нефинитны."""
    F, j = features(w), w.comp["j"]
    ind = np.where(np.isfinite(j), (j > 0).astype(float), np.nan)
    ph, base, npair = np.full(len(w.el), np.nan), np.full(len(w.el), np.nan), np.zeros(len(w.el), int)
    for a, i in enumerate(w.el):
        if not np.isfinite(F[i]).all():
            continue
        idx = W.pairs_idx(w.p.seg, F, min(int(i) + extra, len(w.p.rv) - 1))
        idx = idx[np.isfinite(ind[idx + 1])]
        if len(idx) < 60:
            continue
        mu, sd = F[idx].mean(axis=0), F[idx].std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)
        b = fit_logit((F[idx] - mu) / sd, ind[idx + 1])
        ph[a] = float(sigmoid(b[0] + np.einsum("j,j->", b[1:], (F[i] - mu) / sd)))
        base[a], npair[a] = float(ind[idx + 1].mean()), len(idx)
    return ph, base, npair


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Площадь под ROC (Манн–Уитни, средние ранги); NaN, если один класс."""
    y = np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def metrics(w: DC.DWorld, ph: np.ndarray, base: np.ndarray) -> pd.DataFrame:
    """AUC и Brier к базовой частоте: всего, по годам, без 24.02.2022; контроль — наивный прогноз 1[J_d > 0] (персистентность)."""
    y = (w.comp["j"][w.el + 1] > 0).astype(float)
    naive = (w.comp["j"][w.el] > 0).astype(float)
    ok = np.isfinite(ph)
    yr, td = pd.DatetimeIndex(w.td).year.to_numpy(), pd.DatetimeIndex(w.td)
    rows = []
    for lab, m in [("all", ok), ("ex_240222", ok & (td != pd.Timestamp("2022-02-24")))] + [(str(Y), ok & (yr == Y)) for Y in sorted(set(yr))]:
        b, b0 = float(np.mean((ph[m] - y[m]) ** 2)), float(np.mean((base[m] - y[m]) ** 2))
        rows.append({"period": lab, "n": int(m.sum()), "freq": float(y[m].mean()), "auc": auc(ph[m], y[m]), "auc_naive_persist": auc(naive[m], y[m]),
                     "brier": b, "brier_base": b0, "brier_skill": 1 - b / b0})
    return pd.DataFrame(rows)


def diagnostic(m10: pd.DataFrame) -> pd.DataFrame:
    """ДИАГНОСТИКА состоявшихся скачков (не прогноз): по годам доля дней J > 0 и средняя доля J в RV (на сетке дат)."""
    g = m10[m10["grid_d"]]
    return pd.DataFrame({"share_days_J_pos": (g["j"] > 0).groupby(g.index.year).mean(), "mean_J_share_of_rv": (g["j"] / g["rv"]).groupby(g.index.year).mean(),
                         "n_days": g["j"].groupby(g.index.year).size()}).reset_index(names="year")
