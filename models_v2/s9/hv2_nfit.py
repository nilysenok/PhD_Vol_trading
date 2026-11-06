"""Ж-214, лист 4 (V9): движок структурной QLIKE-коррекции. σ̂² = base·exp(λ·g), g = c0 + g_tree, QLIKE-объектив по m = ln f: grad = 1 − RV/f, hess = RV/f (+ штраф μ), HP зафиксированы
(Ж-212 HS6 Ca: `hs_cfit.HP`), ежедневное переобучение, расширяющееся окно, пары (t, t+1), t+1 <= d, база — PIT-прогноз строки t (offset = ln base_t + c0), seed 42. Переиспользован `hs_cfit.fit_day`.
λ_wf ∈ {0; 0,25; 0,5; 0,75; 1} — по закрытым прогнозам (`hs_rules.causal_pick`, >= 252 закрытых, до этого λ = 0 ≡ база побитно). N2 — те же g_k по компонентам, σ̂² = Σ_k σ̂²_k·exp(λ g_k)."""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from models_v2.s8 import hs_cfit, hs_rules
from models_v2.s9 import hv2_msfeat as MS

LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROCS = 4
_S: dict = {}


def target(x) -> np.ndarray:
    """y_t = x_{t+1} (последняя строка — NaN)."""
    y = np.full(len(x), np.nan)
    y[:-1] = np.asarray(x, float)[1:]
    return y


def pair_rows(seg, gate, lb, y, i: int, extra: int = 0) -> np.ndarray:
    """Строки t с t+1 <= i (+extra — МУТАНТ «будущие пары»), одним сегментом, конечными воротами, базой и положительной целью."""
    t = np.arange(0, min(i + extra, len(y) - 1))
    ok = (seg[t] == seg[t + 1]) & np.isfinite(gate[t]).all(axis=1) & np.isfinite(lb[t]) & np.isfinite(y[t]) & (y[t] > 0)
    return t[ok]


def fit_one(S: dict, i: int) -> tuple:
    """-> (g на строке i, fallback). Fallback (g = 0 ≡ база): нет базы/ворот в строке i или < min_pairs пар."""
    if not (np.isfinite(S["lb"][i]) and np.isfinite(S["gate"][i]).all()):
        return 0.0, True
    idx = pair_rows(S["seg"], S["gate"], S["lb"], S["y"], i, S.get("extra", 0))
    if len(idx) < S["min_pairs"]:
        return 0.0, True
    gt, c0 = hs_cfit.fit_day(S["X"], S["y"], S["lb"], idx, i, S["lib"], S["mu"], (0,) * S["X"].shape[1], 42)
    return gt + c0, False


def _init(st: dict) -> None:
    _S.update(st)


def day_job(i: int) -> tuple:
    return fit_one(_S, i)


def g_days(X, y, lb, seg, rows, lib: str, mu: float = 0.0, procs: int = PROCS, min_pairs: int = MS.MIN_PAIRS, extra: int = 0) -> tuple:
    """g и fallback на строках rows. procs <= 1 — в этом процессе (тесты, обрезка), иначе пул spawn."""
    st = dict(X=X, y=y, lb=lb, seg=seg, gate=X[:, :4], lib=lib, mu=mu, min_pairs=min_pairs, extra=extra)
    rows = [int(i) for i in rows]
    if procs <= 1:
        old = dict(_S)
        _S.clear()
        _S.update(st)
        try:
            res = [day_job(i) for i in rows]
        finally:
            _S.clear()
            _S.update(old)
    else:
        with mp.get_context("spawn").Pool(min(procs, PROCS), initializer=_init, initargs=(st,)) as p:
            res = p.map(day_job, rows, chunksize=8)
    return np.array([r[0] for r in res], float), np.array([r[1] for r in res], bool)


def deriv_check(y, m, off, mu: float = 0.0, gh=hs_cfit.gh, e: float = 1e-6) -> dict:
    """Численная проверка производных QLIKE по m = ln f (центральные разности `hs_cfit.loss`): макс. ошибка градиента и гессиана, min гессиана, min f = e^m. `gh` — проверяемая функция (мутации)."""
    y, m, off = (np.asarray(a, float) for a in (y, m, off))
    g, h = gh(y, m, off, mu)
    ng = (hs_cfit.loss(y, m + e, off, mu) - hs_cfit.loss(y, m - e, off, mu)) / (2 * e)
    nh = (gh(y, m + e, off, mu)[0] - gh(y, m - e, off, mu)[0]) / (2 * e)
    return {"max_err_grad": float(np.max(np.abs(g - ng))), "max_err_hess": float(np.max(np.abs(h - nh))), "min_hess": float(h.min()), "min_f": float(np.exp(m).min())}


def lam_pick(rv, S, d, td, min_closed: int = hs_rules.MIN_CLOSED, extra: int = 0) -> tuple:
    """S — кандидаты по λ (столбцы, порядок LAMBDAS; n×5). -> (σ̂² выбранного λ, λ). Выбор на дату d — по закрытым (target_date <= d); `extra` — МУТАНТ «закрыто + extra»."""
    ic = hs_rules.causal_pick(hs_rules.qlike_matrix(rv, S), d, td, base=0, min_closed=min_closed, extra=extra)
    return hs_rules.pick_series(S, ic), np.array(LAMBDAS)[ic]


def n1_candidates(base, g) -> np.ndarray:
    return np.asarray(base, float)[:, None] * np.exp(np.array(LAMBDAS)[None, :] * np.asarray(g, float)[:, None])


def n2_candidates(comps: list, gs: list, fb, tot) -> np.ndarray:
    """Кандидаты N2 по λ: в днях fallback T3 — прогноз T3 (= A0), иначе Σ_k σ̂²_k·exp(λ g_k) (порядок суммирования как в `hv2_dec.run`)."""
    return np.column_stack([np.where(fb, tot, np.sum([c * np.exp(lam * g) for c, g in zip(comps, gs)], axis=0)) for lam in LAMBDAS])


def run_n1(dw, base: np.ndarray, groups, lib: str, mu: float = 0.0, procs: int = PROCS, min_pairs: int = MS.MIN_PAIRS, min_closed: int = hs_rules.MIN_CLOSED, extra: int = 0, pair_extra: int = 0) -> dict:
    """N1 на сетке dw.el: base — σ̂² базы на всех строках (NaN вне расширенных). -> s2 (λ_wf), full (λ = 1), g, lam, fb."""
    rows = dw.el
    X, _ = MS.feature_matrix(dw, np.log(base), groups)
    g, fb = g_days(X, target(dw.p.rv), np.log(base), dw.p.seg, rows, lib, mu, procs, min_pairs, pair_extra)
    S = n1_candidates(base[rows], g)
    s2, lam = lam_pick(dw.p.rv[rows + 1], S, dw.p.dates[rows], dw.p.dates[rows + 1], min_closed, extra)
    return {"s2": s2, "full": S[:, -1], "g": g, "lam": lam, "fb": fb}


def run_n2(dw, t3: dict, groups, lib: str, mu: float = 0.0, procs: int = PROCS, min_pairs: int = MS.MIN_PAIRS, min_closed: int = hs_rules.MIN_CLOSED, extra: int = 0) -> dict:
    """N2 по компонентам T3 (OPEN, MIDDLE, CLOSE): свой бустер на компоненту, база — ПРОГНОЗ T3 компоненты. -> s2, full, g_<k>, lam, fb (fallback T3)."""
    rows, names = dw.el, list(t3["comps"])
    gs, fbs = [], []
    for k in names:
        base = t3["comps"][k]
        X, _ = MS.feature_matrix(dw, np.log(base), groups)
        g, fb = g_days(X, target(dw.comp[k]), np.log(base), dw.p.seg, rows, lib, mu, procs, min_pairs)
        gs.append(g)
        fbs.append(fb)
    S = n2_candidates([t3["comps"][k][rows] for k in names], gs, t3["fb"][rows] > 0, t3["s2"][rows])
    s2, lam = lam_pick(dw.p.rv[rows + 1], S, dw.p.dates[rows], dw.p.dates[rows + 1], min_closed, extra)
    return {"s2": s2, "full": S[:, -1], "lam": lam, "fb": np.any(fbs, axis=0), **{"g_" + k: g for k, g in zip(names, gs)}}
