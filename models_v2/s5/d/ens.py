"""D1: правила ансамблей E1–E7 над готовыми прогнозами σ̂² членов. Веса — только по закрытым прогнозам (target_date <= d) и по режиму на d.
Члены: H = HAR-J(б), X = b2_f4, L = LGBM F4, Xr = XGB F4 rolling 1008, Xh = XGB F4 веса 2 года. Сетка симплекса шаг 0,1; ничья <= 1e-12 — первый в лексикографическом порядке."""
from __future__ import annotations

import itertools

import numpy as np

from models_v2 import metrics
from models_v2.s3 import post

TIE, MIN_CLOSED, WIN, MIN_GROUP = post.TIE, 252, 252, 60
SPEC = {"e1_roll": (("X", "L"), "roll"), "e1_exp": (("X", "L"), "exp"), "e3_roll": (("H", "L"), "roll"), "e3_exp": (("H", "L"), "exp"),
        "e4_roll": (("H", "X", "L"), "roll"), "e4_exp": (("H", "X", "L"), "exp"), "e5_roll": (("X", "Xr", "Xh"), "roll"),
        "e5_exp": (("X", "Xr", "Xh"), "exp"), "e6_g4": (("H", "X", "L"), "g4"), "e6_rv": (("H", "X", "L"), "rv"),
        "e7_hx": (("H", "X"), "inv"), "e7_hxl": (("H", "X", "L"), "inv"), "e7_xl": (("X", "L"), "inv")}
CONTROL_SETS = {"xl": ("X", "L"), "hx": ("H", "X"), "hl": ("H", "L"), "hxl": ("H", "X", "L"), "xrh": ("X", "Xr", "Xh")}
for _k, _m in CONTROL_SETS.items():
    SPEC["mean_lvl_" + _k], SPEC["mean_log_" + _k] = (_m, "mean_lvl"), (_m, "mean_log")
TRIALS = tuple(k for k in SPEC if not k.startswith("mean_"))


def simplex(K: int, step: float = 0.1) -> np.ndarray:
    """Все векторы неотрицательных весов с суммой 1 на сетке step, лексикографически по возрастанию."""
    n = int(round(1 / step))
    g = [c for c in itertools.product(range(n + 1), repeat=K) if sum(c) == n]
    return np.array(sorted(g), float) / n


def pick(tot: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Веса с минимальной суммой потерь; ничья <= TIE — первый индекс (лексикографически меньший вес первого члена)."""
    return G[int(np.flatnonzero(tot <= tot.min() + TIE)[0])]


def cum_qlike(rv, S, G) -> np.ndarray:
    """C[j, p] = Σ_{k<=j} QLIKE(rv_k, Σ_i G[p, i]·S[k, i])."""
    mix = np.einsum("nk,pk->np", S, G)
    return np.cumsum(metrics.qlike(rv[:, None], mix), axis=0)


def closed(ctx: dict, extra: int = 0) -> np.ndarray:
    n = len(ctx["rv"])
    return np.minimum(post.closed_count(np.asarray(ctx["d"]), np.asarray(ctx["td"])) + extra, n)


def w_window(rv, S, m, win, step: float = 0.1) -> np.ndarray:
    """roll (win > 0): последние win закрытых; exp (win = 0): все закрытые; до 252 закрытых — равные веса."""
    n, K = S.shape
    G = simplex(K, step)
    C = cum_qlike(rv, S, G)
    out = np.full((n, K), 1.0 / K)
    for k in range(n):
        if m[k] >= MIN_CLOSED:
            lo = C[m[k] - 1 - win] if win and m[k] > win else 0.0
            out[k] = pick(C[m[k] - 1] - lo, G)
    return out


def w_regime(rv, S, m, grp, step: float = 0.1) -> np.ndarray:
    """E6: веса по закрытым прогнозам той же группы режима (группа на d); меньше MIN_GROUP закрытых в группе — веса exp; до 252 — равные."""
    n, K = S.shape
    G = simplex(K, step)
    Q = metrics.qlike(rv[:, None], np.einsum("nk,pk->np", S, G))
    exp = w_window(rv, S, m, 0, step)
    out = exp.copy()
    for g in (False, True):
        mem = (grp == g)
        Cg, cnt = np.cumsum(Q * mem[:, None], axis=0), np.cumsum(mem)
        for k in np.flatnonzero(mem):
            if m[k] >= MIN_CLOSED and cnt[m[k] - 1] >= MIN_GROUP:
                out[k] = pick(Cg[m[k] - 1], G)
    return out


def w_inverse(rv, S, m, win: int = WIN) -> np.ndarray:
    """E7: w_i ∝ 1/Q_i, Q_i — средний QLIKE члена i по последним win (252) закрытым; до 252 закрытых — равные веса."""
    n, K = S.shape
    cq = np.cumsum(metrics.qlike(rv[:, None], S), axis=0)
    out = np.full((n, K), 1.0 / K)
    for k in range(n):
        if m[k] >= MIN_CLOSED and m[k] >= win:
            q = (cq[m[k] - 1] - (cq[m[k] - 1 - win] if m[k] > win else 0.0)) / win
            out[k] = (1.0 / q) / (1.0 / q).sum()
    return out


def stack(members: tuple, ctx: dict, M: dict) -> np.ndarray:
    return np.column_stack([ctx["h"] if k == "H" else M[k] for k in members]).astype(float)


def build(row: str, ctx: dict, M: dict, extra_closed: int = 0, step: float = 0.1, win: int = WIN) -> np.ndarray:
    """σ̂² строки row. ctx — mix.context (rv, h, d, td, st_g4, st_rv); M — {'X','L','Xr','Xh'}: σ̂² на сетке. extra_closed=1 — мутация MD1."""
    members, rule = SPEC[row]
    S, rv = stack(members, ctx, M), ctx["rv"]
    if rule == "mean_lvl":
        return S.mean(axis=1)
    if rule == "mean_log":
        return np.exp(np.log(S).mean(axis=1))
    m = closed(ctx, extra_closed)
    if rule in ("roll", "exp"):
        w = w_window(rv, S, m, win if rule == "roll" else 0, step)
    elif rule in ("g4", "rv"):
        w = w_regime(rv, S, m, ctx["st_" + rule], step)
    else:
        w = w_inverse(rv, S, m, win)
    return np.einsum("nk,nk->n", w, S)
