"""FS: σ-признаки надстроек — предрасчёт для всех окон и кэш по содержимому ряда.

Ранг — как `fastbt.s9_stats.past_rank`: доля значений s[t−W..t−1] (без NaN) строго меньше s_t,
текущий бар в окно НЕ входит; NaN при < minp значений окна или NaN в s_t. Ядро то же
(целые счётчики, одно деление), поэтому результат побитно равен past_rank.

`precompute(eng, labels)` за один проход numba (prange по тикерам) считает ранги σ для окон
{126, 252, 504}, ранги EMA(3/5/10) σ для окон {126, 252}, медианы прошлого окна {22, 126, 252}
и ранг отношения σ к прошлой медиане за 22 дня (реакция на всплеск, а не на уровень) для
окон {126, 252, 504}. Всё кладётся в память (`_MEMO`) под ключом (sha1 ряда, окно, minp):
вызовы past_rank / past_median / ema из `fastbt.s9` во время быстрого прогона
(`fastbt.fast_grid.patched`) берут готовое, промах считается старой функцией и тоже кэшируется.
"""
from __future__ import annotations

import hashlib

import numpy as np
from numba import njit, prange

RANK_W = (126, 252, 504)
EMA_SPANS = (3, 5, 10)
EMA_RANK_W = (126, 252)
MED_W = (22, 126, 252)
REL_MED_W = 22

_MEMO: dict = {}
STATS = {"hit": 0, "miss": 0}


@njit(cache=True, parallel=True)
def past_rank_rows(S, n_t, W, minp, out):
    """out[i, t] = past_rank(S[i, :n_t[i]], W, minp)[t]; хвост за n_t — NaN."""
    T = S.shape[0]
    for i in prange(T):
        n = n_t[i]
        for t in range(S.shape[1]):
            out[i, t] = np.nan
        for t in range(n):
            v = S[i, t]
            if v != v:
                continue
            lo = t - W if t - W > 0 else 0
            c = 0
            k = 0
            for j in range(lo, t):
                x = S[i, j]
                if x == x:
                    c += 1
                    if x < v:
                        k += 1
            if c >= minp:
                out[i, t] = k / c


def _dig(a) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(a, np.float64).tobytes()).digest()


def _orig():
    from fastbt import s9_stats
    return s9_stats.__dict__.get("_fs_orig", None) or dict(
        past_rank=s9_stats.past_rank, past_median=s9_stats.past_median, ema=s9_stats.ema)


def past_rank(s, W, minp):
    k = ("rank", _dig(s), int(W), int(minp))
    r = _MEMO.get(k)
    if r is None:
        STATS["miss"] += 1
        r = _orig()["past_rank"](np.ascontiguousarray(s, np.float64).copy(), W, minp)
        _MEMO[k] = r
    else:
        STATS["hit"] += 1
    return r.copy()


def past_median(s, W):
    k = ("med", _dig(s), int(W))
    r = _MEMO.get(k)
    if r is None:
        STATS["miss"] += 1
        r = _orig()["past_median"](s, W)
        _MEMO[k] = r
    else:
        STATS["hit"] += 1
    return r.copy()


def ema(s, span):
    k = ("ema", _dig(s), int(span))
    r = _MEMO.get(k)
    if r is None:
        STATS["miss"] += 1
        r = _orig()["ema"](s, span)
        _MEMO[k] = r
    else:
        STATS["hit"] += 1
    return r.copy()


def minp(W: int) -> int:
    from fastbt.s9_stats import minp as m
    return m(W)


def _register_ranks(S, n_t, W):
    R = np.empty_like(S)
    past_rank_rows(np.ascontiguousarray(S, np.float64), n_t.astype(np.int64), W, minp(W), R)
    for i in range(S.shape[0]):
        n = n_t[i]
        _MEMO[("rank", _dig(S[i, :n]), int(W), int(minp(W)))] = R[i, :n].copy()
    return R


def precompute(eng, labels) -> dict:
    """Ранги/медианы/EMA для σ-источников labels ([тикер, бар] из eng.sigma). Возвращает
    {label: {имя признака: матрица}} — в том числе ранг отношения к медиане 22 дней."""
    o = _orig()
    feats = {}
    n_t = np.asarray(eng.n_t, np.int64)
    for lab in labels:
        S = eng.sigma(lab)
        T, N = S.shape
        f = {}
        for W in RANK_W:
            f[f"rank{W}"] = _register_ranks(S, n_t, W)
        for span in EMA_SPANS:
            Se = np.full((T, N), np.nan)
            for i in range(T):
                n = n_t[i]
                e = o["ema"](S[i, :n], span)
                _MEMO[("ema", _dig(S[i, :n]), int(span))] = e
                Se[i, :n] = e
            for W in EMA_RANK_W:
                f[f"ema{span}_rank{W}"] = _register_ranks(Se, n_t, W)
        for W in MED_W:
            M = np.full((T, N), np.nan)
            for i in range(T):
                n = n_t[i]
                m = o["past_median"](S[i, :n], W)
                _MEMO[("med", _dig(S[i, :n]), int(W))] = m
                M[i, :n] = m
            f[f"med{W}"] = M
        rel = S / f[f"med{REL_MED_W}"]
        f["rel_med22"] = rel
        for W in RANK_W:
            f[f"rel_med22_rank{W}"] = _register_ranks(rel, n_t, W)
        feats[lab] = f
    return feats


def clear() -> None:
    _MEMO.clear()
    STATS.update(hit=0, miss=0)
