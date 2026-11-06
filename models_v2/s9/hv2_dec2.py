"""Ж-214, лист 3 (остаток V7, схема T; У22 GATES_V2, принято дирижёром): T2 `tod_cross`, T4 `tod_qlike`, P1 `tod_edges_w`, P2 `tod_edges_qlike`.
T2: [ln x_d, ln x_w, ln x_m своей части] + ln(1+J_d) + ln d-лаг двух других частей. T4: признаки T1, QLIKE-подгонка по каждой части отдельно (σ̂²_k = exp(Xγ) без s²/2, старт OLS, как У3).
P1: T3 с краевой частью по w-среднему (ln CLOSE_w в OPEN, MIDDLE; ln OPEN_w в CLOSE). P2: признаки T3, QLIKE-подгонка. Сумма частей, fallback на A0 — как в hv2_dec (У10)."""
from __future__ import annotations

import numpy as np

from models_v2.s9 import hv2_dec as DC, hv2_qlog as QG, hv2_world as W

PARTS = ("rv_open", "rv_mid", "rv_close")
CONFIGS = {"T2": False, "T4": True, "P1": False, "P2": True}     # конфигурация -> QLIKE-подгонка
EDGE = {"rv_open": "rv_close", "rv_mid": "rv_close", "rv_close": "rv_open"}


def spec(w: DC.DWorld, cfg: str) -> list:
    """[(часть, признаки, цель-уровень)] конфигурации."""
    seg, c = w.p.seg, w.comp
    J = w.p.X[:, 3:4]
    eqs = []
    for k in PARTS:
        base = [DC.own(c[k], seg), J]
        if cfg == "T2":
            base += [DC.lnx(c[o]) for o in PARTS if o != k]
        elif cfg == "P1":
            base += [DC.lnx(DC.wmean(c[EDGE[k]], seg, 5))]
        elif cfg == "P2":
            base += [DC.lnx(c[EDGE[k]])]
        eqs.append((k, np.column_stack(base), c[k]))
    return eqs


def forecast_q(w: W.World, F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """σ̂² = exp(γ'(1, F_d)), γ — минимум QLIKE на парах t+1 <= d (цель > 0); NaN при нефинитных признаках строки d или < k+2 парах."""
    out = np.full(len(w.el), np.nan)
    for a, i in enumerate(w.el):
        if not np.isfinite(F[i]).all():
            continue
        idx = W.pairs_idx(w.p.seg, F, int(i))
        idx = idx[np.isfinite(y[idx + 1]) & (y[idx + 1] > 0)]
        if len(idx) < F.shape[1] + 2:
            continue
        gam = QG.wfit(np.column_stack([np.ones(len(idx)), F[idx]]), y[idx + 1], np.ones(len(idx)))
        out[a] = float(np.exp(gam[0] + np.einsum("j,j->", gam[1:], F[i])))
    return out


def run(w: DC.DWorld, cfg: str, a0: np.ndarray | None = None) -> tuple:
    """-> (σ̂² сетки с подменой на A0, {часть: σ̂²_k}, маска fallback)."""
    comps = {}
    for k, F, y in spec(w, cfg):
        comps[k] = forecast_q(w, F, y) if CONFIGS[cfg] else DC.forecast(w, F, DC.lnx(y))
    tot = np.sum([comps[n] for n in comps], axis=0)
    fb = ~np.isfinite(tot) | (tot <= 0)
    a0 = DC.a0_forecast(w) if a0 is None else a0
    return np.where(fb, a0, tot), comps, fb


def sigma(w: DC.DWorld, cfg: str) -> np.ndarray:
    return run(w, cfg)[0]
