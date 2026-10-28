"""Numba-ядра S5/S6 (O(T)): pandas-совместимые rolling sum/std, VWAP, pivot, входы/выходы.

pandas-арифметика старого кода воспроизведена побитово: Kahan-суммы rolling sum/std pandas
и FMA, как в скомпилированном pandas (arm64). Хэш этого файла входит в code_version семейства.
"""
import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic
from llvmlite import ir

from fastbt.signals.interface import WARMUP  # noqa: E402


@intrinsic
def _fma(typingctx, a, b, c):  # llvm.fma: pandas (clang/arm64) сливает a*b+c в FMA
    def codegen(context, builder, signature, args):
        f = ir.DoubleType()
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(f, [f, f, f]), "llvm.fma.f64")
        return builder.call(fn, args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def _roll_sum(x, w):
    """pandas rolling(w, min_periods=w).sum(): Kahan, раздельные add/remove-компенсации."""
    n = len(x); out = np.full(n, np.nan)
    nobs = 0; s = 0.0; ca = 0.0; cr = 0.0; same = 0; prev = np.nan
    for i in range(n):
        if i >= w and x[i - w] == x[i - w]:
            nobs -= 1; y = -x[i - w] - cr; t = s + y; cr = t - s - y; s = t
        v = x[i]
        if v == v:
            nobs += 1; y = v - ca; t = s + y; ca = t - s - y; s = t
            same = same + 1 if v == prev else 1
            prev = v
        if nobs >= w:
            out[i] = prev * nobs if same >= nobs else s
    return out


@njit(cache=True)
def _roll_std(x, w):
    """pandas rolling(w, min_periods=w).std(ddof=1): Welford+Kahan, NaN пропускаются."""
    n = len(x); out = np.full(n, np.nan)
    nobs = 0; mean = 0.0; ssq = 0.0; ca = 0.0; cr = 0.0; same = 0; prev = np.nan
    for i in range(n):
        if i >= w and x[i - w] == x[i - w]:
            v = x[i - w]; nobs -= 1
            if nobs > 0:
                pm = mean - cr; y = v - cr; t = y - mean; cr = t + mean - y
                mean = mean - t / nobs
                ssq = _fma(-(v - pm), v - mean, ssq)
            else:
                mean = 0.0; ssq = 0.0
        v = x[i]
        if v == v:
            nobs += 1
            same = same + 1 if v == prev else 1
            prev = v
            pm = mean - ca; y = v - ca; t = y - mean; ca = t + mean - y
            mean = mean + t / nobs
            ssq = _fma(v - pm, v - mean, ssq)
        if nobs >= w and nobs > 1:
            r = 0.0 if same >= nobs else ssq / (nobs - 1)
            out[i] = np.sqrt(max(r, 0.0))
    return out


@njit(cache=True)
def _vwap(close, high, low, volume, w):
    """VWAP_w и dev (S6R:201-214): dev = std(close - vwap, w), зажат снизу 1e-12."""
    n = len(close)
    tpv = np.empty(n)
    for i in range(n):
        tpv[i] = (high[i] + low[i] + close[i]) / 3.0 * volume[i]
    sv = _roll_sum(volume, w)
    vwap = _roll_sum(tpv, w)
    for i in range(n):
        vwap[i] = vwap[i] / (1e-12 if (np.isnan(sv[i]) or sv[i] < 1e-12) else sv[i])
    dev = _roll_std(close - vwap, w)
    for i in range(n):
        if np.isnan(dev[i]) or dev[i] < 1e-12:
            dev[i] = 1e-12
    return vwap, dev


@njit(cache=True)
def _pivots(gid, high, low, close, fib):
    """Pivot P,S1,S2,R1,R2 по H/L/C ПРЕДЫДУЩЕЙ группы (неделя/месяц ряда тикера), S5R:87-172."""
    n = len(close)
    out = np.full((5, n), np.nan)
    i = 0; have = False; pH = pL = pC = 0.0
    while i < n:
        j = i; H = high[i]; L = low[i]
        while j < n and gid[j] == gid[i]:
            H = max(H, high[j]); L = min(L, low[j]); j += 1
        if have:
            P = (pH + pL + pC) / 3.0; rng = pH - pL
            if fib:
                s1 = P - 0.382 * rng; r1 = P + 0.382 * rng
                s2 = P - 0.618 * rng; r2 = P + 0.618 * rng
            else:
                s1 = 2 * P - pH; r1 = 2 * P - pL; s2 = P - rng; r2 = P + rng
            for k in range(i, j):
                out[0, k] = P; out[1, k] = s1; out[2, k] = s2; out[3, k] = r1; out[4, k] = r2
        pH = H; pL = L; pC = close[j - 1]; have = True; i = j
    return out


@njit(cache=True)
def _entry(sig, long_c, short_c, ok):
    for t in range(WARMUP, len(sig)):
        if ok[t] and long_c[t]:
            sig[t] = 1
        elif ok[t] and short_c[t]:
            sig[t] = -1


@njit(cache=True)
def _s5_exit(close, P):
    n = len(close); ex = np.zeros(n, dtype=np.bool_)
    for t in range(1, n):
        if not (np.isnan(P[t]) or np.isnan(P[t - 1])):
            ex[t] = ((close[t - 1] < P[t - 1]) and close[t] >= P[t]) or \
                    ((close[t - 1] > P[t - 1]) and close[t] <= P[t])
    return ex
