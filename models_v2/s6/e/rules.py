"""Правила весов ансамблей и гибридов (волна E, GATES_E R3–R5). Вес на дате d — функция закрытых прогнозов (target_date <= d) и RV до d.
S — (n, K) σ̂² членов на сетке; m[k] — число закрытых прогнозов на k (`ens.closed`); extra — мутация «закрыто + extra» (должна краснить тесты)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2 import metrics
from models_v2.s5.d import ens

MIN_CLOSED, WIN, CUT = ens.MIN_CLOSED, ens.WIN, np.datetime64("2018-12-31")
SIMPLE = ("mean_lvl", "mean_log", "median")


def w_equal(S: np.ndarray) -> np.ndarray:
    return np.full(S.shape, 1.0 / S.shape[1])


def n_cut(td: np.ndarray, cut=CUT) -> int:
    """Число прогнозов с target_date <= cut (первые n_cut: сетка отсортирована)."""
    return int((np.asarray(td).astype("datetime64[D]") <= cut).sum())


def w_fixed(rv, S, m, c: int, step: float = 0.1) -> np.ndarray:
    """fixed_2018: оптимум симплекса по первым c прогнозам (цели <= 31.12.2018), затем фиксирован; на k, где закрыто < c, — равные веса."""
    out = w_equal(S)
    if c < 1 or c > len(rv):
        return out
    G = ens.simplex(S.shape[1], step)
    wf = ens.pick(ens.cum_qlike(rv, S, G)[c - 1], G)
    out[m >= c] = wf
    return out


def w_inv_rmse(rv, S, m, win: int = WIN) -> np.ndarray:
    """w_i ∝ 1/RMSE (уровень σ̂² против RV) по последним win закрытым; до max(win, MIN_CLOSED) закрытых — равные веса."""
    n, K = S.shape
    cs = np.cumsum((rv[:, None] - S) ** 2, axis=0)
    out = w_equal(S)
    for k in range(n):
        if m[k] >= max(win, MIN_CLOSED):
            r = np.sqrt((cs[m[k] - 1] - (cs[m[k] - 1 - win] if m[k] > win else 0.0)) / win)
            out[k] = (1.0 / r) / (1.0 / r).sum()
    return out


def w_roll_con(rv, S, m, win: int = WIN, lo: float = 0.1, step: float = 0.1) -> np.ndarray:
    """roll_252_con: оптимум симплекса по последним win закрытым при весах каждого члена >= lo."""
    n, K = S.shape
    G = ens.simplex(K, step)
    G = G[G.min(axis=1) >= lo - 1e-12]
    C = ens.cum_qlike(rv, S, G)
    out = w_equal(S)
    for k in range(n):
        if m[k] >= MIN_CLOSED:
            out[k] = ens.pick(C[m[k] - 1] - (C[m[k] - 1 - win] if m[k] > win else 0.0), G)
    return out


def _fit_stack(rv, S, x0) -> np.ndarray:
    """Минимум среднего QLIKE смеси Σ w_i S_i по w >= 1e-6 (сумма свободна); выпуклая задача, L-BFGS-B."""
    def f(w):
        s2 = S @ w
        r = rv / s2
        return float((r - np.log(r) - 1).mean()), ((1.0 / s2 - rv / s2 ** 2)[:, None] * S).mean(axis=0)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(1e-6, 10.0)] * S.shape[1])
    return res.x


def w_stack(rv, S, m, refit: int = 21) -> np.ndarray:
    """stack_nnls: неотрицательные веса по ВСЕМ закрытым прогнозам (временной OOF); переоценка каждые refit дат, между — прежние. До MIN_CLOSED — равные."""
    n, K = S.shape
    out, w = w_equal(S), None
    for k in range(n):
        if m[k] < MIN_CLOSED:
            continue
        if w is None or k % refit == 0:
            w = _fit_stack(rv[:m[k]], S[:m[k]], np.full(K, 1.0 / K) if w is None else w)
        out[k] = w
    return out


def weights(rule: str, S, ctx: dict, extra: int = 0, c: int | None = None) -> np.ndarray:
    """Веса (n, K) правила rule. extra — мутация MD1 (закрыто + extra). c — число прогнозов до 31.12.2018 (для fixed_2018)."""
    rv, m = ctx["rv"], ens.closed(ctx, extra)
    if rule in ("fix", "mean_lvl"):
        return w_equal(S)
    if rule == "fixed_2018":
        return w_fixed(rv, S, m, n_cut(ctx["td"]) if c is None else c)
    if rule == "inv_qlike_252":
        return ens.w_inverse(rv, S, m, 252)
    if rule == "inv_rmse_252":
        return w_inv_rmse(rv, S, m, 252)
    if rule.startswith("dyn_") or rule == "roll_252":
        return ens.w_window(rv, S, m, int(rule.split("_")[1]))
    if rule == "roll_252_con":
        return w_roll_con(rv, S, m, 252)
    if rule == "stack_nnls":
        return w_stack(rv, S, m)
    if rule == "reg_g4":
        return ens.w_regime(rv, S, m, ctx["st_g4"])
    raise KeyError(rule)


def sigma2(rule: str, S, ctx: dict, extra: int = 0, c: int | None = None) -> np.ndarray:
    """σ̂² правила rule над членами S; простые правила (контроли) не требуют ctx."""
    if rule == "mean_lvl":
        return S.mean(axis=1)
    if rule == "mean_log":
        return np.exp(np.log(S).mean(axis=1))
    if rule == "median":
        return np.median(S, axis=1)
    return np.einsum("nk,nk->n", weights(rule, S, ctx, extra, c), S)


def qlike_mean(rv, s2) -> float:
    return float(metrics.qlike(rv, s2).mean())
