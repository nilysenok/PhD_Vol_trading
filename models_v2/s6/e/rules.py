"""Правила весов ансамблей и гибридов (волна E, GATES_E R3, R6). Вес на дате d — функция закрытых прогнозов (target_date <= d) и RV до d.
S — (n, K) σ̂² членов на сетке; m[k] — число закрытых прогнозов на k (`ens.closed`); extra — мутация «закрыто + extra» (должна краснить тесты)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from models_v2 import metrics
from models_v2.s5.d import ens

MIN_CLOSED, WIN = ens.MIN_CLOSED, ens.WIN
SIMPLE = ("mean_lvl", "mean_log", "median")
S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def w_equal(S: np.ndarray) -> np.ndarray:
    return np.full(S.shape, 1.0 / S.shape[1])


def w_fixed_loyo(rv, S, td, d, step: float = 0.1, extra_year: int = 0, min_n: int = 60) -> np.ndarray:
    """fixed_loyo: на дате d года Y вес = оптимум симплекса по прогнозам с target < 01.01.Y, фиксирован в течение года Y. Если таких < min_n (2017) — равные.
    extra_year — мутация «target < 01.01.(Y+extra_year)»: вес года Y видит прогнозы самого года Y."""
    out = w_equal(S)
    G = ens.simplex(S.shape[1], step)
    C = ens.cum_qlike(rv, S, G)
    td = np.asarray(td).astype("datetime64[D]")
    yr = np.asarray(d).astype("datetime64[Y]").astype(int) + 1970
    for Y in np.unique(yr):
        c = int((td < np.datetime64("%d-01-01" % (Y + extra_year))).sum())
        if c >= min_n:
            out[yr == Y] = ens.pick(C[c - 1], G)
    return out


def w_shrink(rv, S, m, win: int = 252, sel: int = 126, grid=S_GRID, extra_sel: int = 0) -> np.ndarray:
    """shrink_252: w = (1−s)·w_roll + s·w_eq; s — минимум среднего QLIKE этих же правил на последних sel закрытых датах j < m[k] (вес даты j посчитан на j).
    До sel закрытых — s = 1 (равные веса). extra_sel — мутация: окно выбора s захватывает extra_sel ещё не закрытых дат."""
    n = S.shape[0]
    wr, we = ens.w_window(rv, S, m, win), w_equal(S)
    Q = np.column_stack([metrics.qlike(rv, np.einsum("nk,nk->n", (1 - s) * wr + s * we, S)) for s in grid])
    cq = np.vstack([np.zeros((1, len(grid))), np.cumsum(Q, axis=0)])
    out = we.copy()
    for k in range(n):
        mk = min(int(m[k]) + extra_sel, n)
        if mk >= sel:
            s = grid[int(np.argmin(cq[mk] - cq[mk - sel]))]
            out[k] = (1 - s) * wr[k] + s * we[k]
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


def w_group(rv, S, m, grp, step: float = 0.1, min_group: int = 60) -> np.ndarray:
    """C2/C3: веса по закрытым прогнозам той же группы (grp на d: 0…ng−1, −1 — нет); меньше min_group закрытых в группе или группа −1 — веса exp (все закрытые); до 252 — равные."""
    n, K = S.shape
    G = ens.simplex(K, step)
    Q = metrics.qlike(rv[:, None], np.einsum("nk,pk->np", S, G))
    out = ens.w_window(rv, S, m, 0, step)
    for g in sorted(set(np.asarray(grp).tolist()) - {-1}):
        mem = grp == g
        Cg, cnt = np.cumsum(Q * mem[:, None], axis=0), np.cumsum(mem)
        for k in np.flatnonzero(mem):
            if m[k] >= MIN_CLOSED and cnt[m[k] - 1] >= min_group:
                out[k] = ens.pick(Cg[m[k] - 1], G)
    return out


def w_switch(rv, S, m, win: int = 63) -> np.ndarray:
    """C6: S = (HAR, ML). ML, если сумма потерь HAR за последние win закрытых больше суммы потерь ML (гистерезис 0), иначе HAR; до win закрытых — HAR."""
    cq = np.vstack([np.zeros((1, 2)), np.cumsum(metrics.qlike(rv[:, None], S), axis=0)])
    out = np.tile([1.0, 0.0], (S.shape[0], 1))
    for k in range(S.shape[0]):
        if m[k] >= win:
            d = (cq[m[k]] - cq[m[k] - win])
            out[k] = [0.0, 1.0] if d[0] > d[1] else [1.0, 0.0]
    return out


def c_shrink(rv, h, sb, m, win: int = 252) -> np.ndarray:
    """C5: c_t = clip((f − 0,5)/0,5, 0, 1), f — доля последних win закрытых дней, где QLIKE гибрида sb лучше QLIKE HAR h; до win закрытых c = 0,5."""
    imp = (metrics.qlike(rv, sb) < metrics.qlike(rv, h)).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(imp)])
    c = np.full(len(rv), 0.5)
    for k in range(len(rv)):
        if m[k] >= win:
            c[k] = np.clip((cs[m[k]] - cs[m[k] - win]) / win - 0.5, 0.0, 0.5) / 0.5
    return c


def weights(rule: str, S, ctx: dict, extra: int = 0, extra_year: int = 0) -> np.ndarray:
    """Веса (n, K) правила rule. extra — мутация «закрыто + extra»; extra_year — мутация fixed_loyo («target < 01.01.Y+1»)."""
    rv, m = ctx["rv"], ens.closed(ctx, extra)
    if rule in ("fix", "mean_lvl"):
        return w_equal(S)
    if rule == "fixed_loyo":
        return w_fixed_loyo(rv, S, ctx["td"], ctx["d"], extra_year=extra_year)
    if rule == "inv_qlike_252":
        return ens.w_inverse(rv, S, m, 252)
    if rule == "inv_rmse_252":
        return w_inv_rmse(rv, S, m, 252)
    if rule == "shrink_252":
        return w_shrink(rv, S, m, 252, extra_sel=extra)
    if rule == "roll_252_con":
        return w_roll_con(rv, S, m, 252)
    if rule.startswith("dyn_") or rule.startswith("roll_"):
        return ens.w_window(rv, S, m, int(rule.split("_")[1]))
    if rule == "stack_nnls":
        return w_stack(rv, S, m)
    if rule == "reg_g4":
        return ens.w_regime(rv, S, m, ctx["st_g4"])
    raise KeyError(rule)


def sigma2(rule: str, S, ctx: dict, extra: int = 0, extra_year: int = 0) -> np.ndarray:
    """σ̂² правила rule над членами S; простые правила (контроли) не требуют ctx."""
    if rule == "mean_lvl":
        return S.mean(axis=1)
    if rule == "mean_log":
        return np.exp(np.log(S).mean(axis=1))
    if rule == "median":
        return np.median(S, axis=1)
    return np.einsum("nk,nk->n", weights(rule, S, ctx, extra, extra_year), S)


def qlike_mean(rv, s2) -> float:
    return float(metrics.qlike(rv, s2).mean())
