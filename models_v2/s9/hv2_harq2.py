"""Ж-214, V6, остаток (У9 GATES_V2): H2/H3/H5/H6/H8 — расширения H1 (спецификация spec), H9 — BPQ в уровнях, H10/H11 — плато.
spec: q ('rq'|'rel'), qwin (None|окно z-оценки), react (подмножество {'J','w5'}: дополнительные реакции q·ln(1+J), q̄₅·ln RV_w), wins ((lo, hi) процентили ln RV_d),
stress (процентиль порога S_t), qlike (bool). Порядок колонок: [ln RV_d, q·ln RV_d, ln RV_w, ln RV_m, ln(1+J)] + extras (q·ln(1+J); q̄₅·ln RV_w; S·q·ln RV_d)."""
from __future__ import annotations

import numpy as np

from models_v2.s9 import hv2_harq as HQ, hv2_qlev as QL, hv2_world as W

EXTRA = {"J": "b_JQ", "w5": "b_wQ5", "stress": "b_dQS"}


def winsor(x: np.ndarray, lo: float, hi: float, min_obs: int = 60) -> np.ndarray:
    """x_t обрезается по процентилям lo/hi значений s < t (расширяющееся, причинно); при < min_obs прошлых значений — без изменений."""
    x = np.asarray(x, float)
    out = x.copy()
    for t in range(min_obs, len(x)):
        past = x[:t][np.isfinite(x[:t])]
        if len(past) >= min_obs:
            out[t] = np.clip(x[t], *np.percentile(past, [lo, hi]))
    return out


def stress(mkt: np.ndarray, pct: float, min_obs: int = 60, full: bool = False) -> np.ndarray:
    """S_t = 1[mkt_t > pct-й процентиль mkt_s, s < t]; 0 там, где не определено. `full` — МУТАНТ «порог по всей выборке» (ловится тестом обрезки)."""
    m = np.asarray(mkt, float)
    out = np.zeros(len(m))
    for t in range(len(m)):
        past = m[:] if full else m[:t]
        past = past[np.isfinite(past)]
        if len(past) >= min_obs and np.isfinite(m[t]):
            out[t] = float(m[t] > np.percentile(past, pct))
    return out


def qbar5(q: np.ndarray) -> np.ndarray:
    """Среднее q за дни t−4…t (все пять определены)."""
    out = np.full(len(q), np.nan)
    if len(q) >= 5:
        out[4:] = np.lib.stride_tricks.sliding_window_view(q, 5).mean(axis=1)
    return out


def design(w: W.World, spec: dict) -> tuple:
    """-> (n x k матрица, имена коэффициентов (без b0))."""
    q = W.zscore(_raw(w, spec["q"]), win=spec.get("qwin"))
    X = w.p.X
    xd = winsor(X[:, 0], *spec["wins"]) if spec.get("wins") else X[:, 0]
    cols = [xd, q * xd, X[:, 1], X[:, 2], X[:, 3]]
    names = ["b_d", "b_dQ", "b_w", "b_m", "b_J"]
    if "J" in spec.get("react", ()):
        cols.append(q * X[:, 3]); names.append(EXTRA["J"])
    if "w5" in spec.get("react", ()):
        cols.append(qbar5(q) * X[:, 1]); names.append(EXTRA["w5"])
    if spec.get("stress"):
        cols.append(stress(w.mkt, spec["stress"]) * q * xd); names.append(EXTRA["stress"])
    return np.column_stack(cols), names


def _raw(w: W.World, kind: str) -> np.ndarray:
    return 0.5 * np.log(np.where(w.rq > 0, w.rq, np.nan)) if kind == "rq" else np.log(np.where(w.rel > 0, w.rel, np.nan))


def run(w: W.World, spec: dict) -> tuple:
    """-> (σ̂² сетки, коэффициенты n x (1+k), имена, число пар)."""
    Xd, names = design(w, spec)
    s2, B, npair = HQ.fit_design(w, Xd, bool(spec.get("qlike")))
    return s2, B, ["b0"] + names, npair


def run_h9(w: W.World) -> tuple:
    """H9: σ̂² = β0 + (β1 + β1Q·x)·RV_d + β2 RV_w + β3 RV_m + β_J J, x = √RQ, реакция r(x) >= 0 на [a, b] (min/max x по обучающим парам):
    u = r(a), v = r(b) >= 0, колонки RV_d·λ1, RV_d·λ2 (λ1 = (b−x)/(b−a), λ2 = (x−a)/(b−a)); QLIKE-подгонка (границы), x дня прогноза обрезается в [a, b].
    -> (σ̂², n x 7 [b0, u, v, b_w, b_m, b_J, β1Q], [a, b] по датам)."""
    Z, x = QL.levels(w, True), np.sqrt(np.where(w.rq >= 0, w.rq, np.nan))
    F = np.column_stack([Z, x])
    s2, B, ab = np.empty(len(w.el)), np.empty((len(w.el), 7)), np.empty((len(w.el), 2))
    for k, i in enumerate(w.el):
        idx = W.pairs_idx(w.p.seg, F, int(i))
        a, b = float(x[idx].min()), float(x[idx].max())
        lam = lambda xx: np.column_stack([(b - xx) / (b - a), (xx - a) / (b - a)])
        mk = lambda ii, xx: np.column_stack([Z[ii, 0], Z[ii, 1:2] * lam(xx), Z[ii, 2:5]])
        th = QL.fit_levels(mk(idx, x[idx]), w.p.rv[idx + 1])
        xi = np.clip(x[[i]], a, b)
        s2[k] = float(np.einsum("j,j->", th, mk(np.array([i]), xi)[0]))
        B[k], ab[k] = [*th, (th[2] - th[1]) / (b - a)], (a, b)
    return s2, B, ab
