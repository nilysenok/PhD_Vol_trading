"""FS: быстрый бутстрап для сетки надстроек (тот же результат, что старый путь).

- IID (`fastbt.report.boot_delta`): индексы повторов зависят только от (n, reps, seed) — они
  строятся один раз тем же генератором (RandomState(seed).randint чанками по 500) и кэшируются;
  Шарпы повторов ряда считаются numba-ядром, которое повторяет попарное суммирование numpy
  (`pairwise_sum`, блок 128, 8 аккумуляторов), поэтому совпадают с `_sh(a[idx])` побитно
  (проверено при n ≤ 8192; длиннее — запасной путь numpy). Шарпы повторов кэшируются по
  содержимому ряда: эталон A в сравнении «против A» считается один раз на всю сетку.
- Ледуа–Вольф (`fastbt.s9_stats.lw_sharpe_diff`): старты блоков — та же последовательность
  RandomState(seed).randint(0, T, l) по повторам (кэш); цикл повторов — numba. Внутри повтора
  ψ* = zᵀz/Tb считается последовательной суммой, а не BLAS, поэтому статистика повтора
  совпадает до ~1e-15 (не побитно); p совпадает, пока ни один повтор не лежит на пороге в
  пределах 1e-12 (сверка — `scripts/analysis/fastbt_equiv_fast.py`).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numba import njit, prange

PW_MAX = 8192            # до этой длины попарная сумма numpy повторена побитно


@njit(cache=True)
def _leaf(a, s, n):
    """Лист pairwise_sum: n < 8 — подряд; n ≤ 128 — 8 аккумуляторов."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[s + i]
        return res
    if n <= 128:
        r0 = a[s]
        r1 = a[s + 1]
        r2 = a[s + 2]
        r3 = a[s + 3]
        r4 = a[s + 4]
        r5 = a[s + 5]
        r6 = a[s + 6]
        r7 = a[s + 7]
        i = 8
        m = n - (n % 8)
        while i < m:
            r0 += a[s + i]
            r1 += a[s + i + 1]
            r2 += a[s + i + 2]
            r3 += a[s + i + 3]
            r4 += a[s + i + 4]
            r5 += a[s + i + 5]
            r6 += a[s + i + 6]
            r7 += a[s + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += a[s + i]
            i += 1
        return res
    return 0.0


@njit(cache=True)
def _pw(a, s, n):
    """numpy pairwise_sum (DOUBLE): n ≤ 128 — лист; иначе пополам (n2 = n/2 − (n/2 mod 8)),
    результат = сумма(левая) + сумма(правая). Рекурсия развёрнута в стек: рекурсивные
    функции numba с cache=True падают при загрузке из кэша."""
    if n <= 128:
        return _leaf(a, s, n)
    fs = np.empty(64, np.int64)
    fn = np.empty(64, np.int64)
    fp = np.empty(64, np.int64)
    vals = np.empty(64)
    top = 0
    nv = 0
    fs[0] = s
    fn[0] = n
    fp[0] = 0
    top = 1
    while top > 0:
        top -= 1
        cs = fs[top]
        cn = fn[top]
        ph = fp[top]
        if cn <= 128:
            vals[nv] = _leaf(a, cs, cn)
            nv += 1
        elif ph == 0:
            n2 = cn // 2
            n2 -= n2 % 8
            fs[top] = cs
            fn[top] = cn
            fp[top] = 1
            top += 1
            fs[top] = cs + n2           # правая — после левой
            fn[top] = cn - n2
            fp[top] = 0
            top += 1
            fs[top] = cs
            fn[top] = n2
            fp[top] = 0
            top += 1
        else:
            r = vals[nv - 1]
            l = vals[nv - 2]
            nv -= 2
            vals[nv] = l + r
            nv += 1
    return vals[0]


@njit(cache=True)
def pw_sum(a):
    return 0.0 + _pw(a, 0, len(a))


@njit(cache=True, parallel=True)
def boot_sharpe_iid(x, idx, out):
    """out[r] = _sh(x[idx[r]]) (fastbt.report._sh, ddof=1, порог 1e-12) — побитно как numpy."""
    R, n = idx.shape
    k252 = math.sqrt(252.0)
    for r in prange(R):
        buf = np.empty(n)
        for j in range(n):
            buf[j] = x[idx[r, j]]
        mu = (0.0 + _pw(buf, 0, n)) / n
        for j in range(n):
            d = buf[j] - mu
            buf[j] = d * d
        sd = math.sqrt((0.0 + _pw(buf, 0, n)) / (n - 1))
        out[r] = mu * k252 / (sd if sd > 0 else 1.0) if sd > 1e-12 else 0.0


_IDX: dict = {}
_SH: dict = {}


def iid_index(n: int, reps: int, seed: int = 42) -> np.ndarray:
    """Индексы повторов ровно как в report.boot_delta / final_test.ce_boot (block=1)."""
    k = (n, reps, seed)
    if k not in _IDX:
        rng = np.random.RandomState(seed)
        parts = []
        for i0 in range(0, reps, 500):
            parts.append(rng.randint(0, n, (min(500, reps - i0), n)))
        _IDX[k] = np.ascontiguousarray(np.concatenate(parts))
    return _IDX[k]


def _digest(x: np.ndarray) -> bytes:
    return hashlib.sha1(np.ascontiguousarray(x, np.float64).tobytes()).digest()


def boot_sharpes(x: np.ndarray, reps: int, seed: int = 42) -> np.ndarray:
    """Шарпы повторов IID-бутстрапа ряда x (кэш по содержимому)."""
    x = np.ascontiguousarray(x, np.float64)
    n = len(x)
    k = (_digest(x), n, reps, seed)
    if k not in _SH:
        idx = iid_index(n, reps, seed)
        if n <= PW_MAX:
            out = np.empty(reps)
            boot_sharpe_iid(x, idx, out)
        else:                                       # запасной путь — как в report._sh
            from fastbt.report import _sh
            out = np.concatenate([_sh(x[idx[i:i + 500]]) for i in range(0, reps, 500)])
        _SH[k] = out
    return _SH[k]


def boot_delta(a, b, reps, seed=42):
    """= fastbt.report.boot_delta(a, b, reps, seed) (побитно)."""
    from fastbt.report import _sh
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = boot_sharpes(a, reps, seed) - boot_sharpes(b, reps, seed)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(_sh(a) - _sh(b)), float(np.mean(d <= 0)), float(lo), float(hi)


def clear() -> None:
    _SH.clear()


# ---------------- Ледуа–Вольф ----------------

_LW: dict = {}


def lw_starts(T: int, reps: int, block: int, seed: int) -> np.ndarray:
    """Старты блоков по повторам — та же последовательность, что в lw_sharpe_diff."""
    k = (T, reps, block, seed)
    if k not in _LW:
        rng = np.random.RandomState(seed)
        l = T // block
        _LW[k] = np.ascontiguousarray(np.stack([rng.randint(0, T, l) for _ in range(reps)]))
    return _LW[k]


@njit(cache=True, parallel=True)
def _lw_loop(xx1, xx2, starts, block, d, stat, hit):
    R, l = starts.shape
    Tb = l * block
    for r in prange(R):
        b1 = np.empty(Tb)
        b2 = np.empty(Tb)
        q = np.empty(Tb)
        for i in range(l):
            s0 = starts[r, i]
            for j in range(block):
                b1[i * block + j] = xx1[s0 + j]
                b2[i * block + j] = xx2[s0 + j]
        m1 = (0.0 + _pw(b1, 0, Tb)) / Tb
        m2 = (0.0 + _pw(b2, 0, Tb)) / Tb
        for t in range(Tb):
            q[t] = b1[t] ** 2
        g1 = (0.0 + _pw(q, 0, Tb)) / Tb
        for t in range(Tb):
            q[t] = b2[t] ** 2
        g2 = (0.0 + _pw(q, 0, Tb)) / Tb
        v1 = g1 - m1 ** 2
        v2 = g2 - m2 ** 2
        ds = m1 / math.sqrt(v1) - m2 / math.sqrt(v2)
        gr0 = g1 / v1 ** 1.5
        gr1 = -g2 / v2 ** 1.5
        gr2 = -0.5 * m1 / v1 ** 1.5
        gr3 = 0.5 * m2 / v2 ** 1.5
        psi = np.zeros((4, 4))
        z = np.empty(4)
        for i in range(l):
            z[0] = 0.0
            z[1] = 0.0
            z[2] = 0.0
            z[3] = 0.0
            for j in range(block):
                t = i * block + j
                z[0] += b1[t] - m1
                z[1] += b2[t] - m2
                z[2] += b1[t] ** 2 - g1
                z[3] += b2[t] ** 2 - g2
            for a in range(4):
                for c in range(4):
                    psi[a, c] += z[a] * z[c]
        g = (gr0, gr1, gr2, gr3)
        acc = 0.0
        for a in range(4):
            row = 0.0
            for c in range(4):
                row += psi[a, c] / Tb * g[c]
            acc += g[a] * row
        ses = math.sqrt(max(acc, 1e-30) / Tb)
        hit[r] = 1 if abs(ds - d) / ses >= stat else 0


def lw_sharpe_diff(x1, x2, reps=1000, block=5, seed=42):
    """= fastbt.s9_stats.lw_sharpe_diff (Δ и с.о. — тем же кодом numpy, p — numba-циклом)."""
    from fastbt.s9_stats import _hac, _lw_parts
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    T = len(x1)
    d, grad, y = _lw_parts(x1, x2)
    se = math.sqrt(max(grad @ _hac(y) @ grad, 1e-30) / T)
    st = lw_starts(T, reps, block, seed)
    xx1 = np.concatenate([x1, x1[:block]])
    xx2 = np.concatenate([x2, x2[:block]])
    hit = np.zeros(reps, np.int64)
    _lw_loop(xx1, xx2, st, block, d, abs(d) / se, hit)
    p = (int(hit.sum()) + 1) / (reps + 1)
    return d * math.sqrt(252), se * math.sqrt(252), p
