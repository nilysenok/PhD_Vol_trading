"""Ускорение без изменения семантики: calc_sharpe_comm на numba (точная формула, ddof=1, комиссия COMMISSION).

patch(m) подменяет calc_sharpe_comm в модуле stage3_walkforward (m) и в strategies_walkforward.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _sharpe_comm(pos, lr, mask, ann, comm):
    n = 0
    for i in range(pos.shape[0]):
        if mask[i]:
            n += 1
    if n == 0:
        return -999.0
    net = np.empty(n)
    j = 0
    prev = 0.0
    for i in range(pos.shape[0]):
        if mask[i]:
            net[j] = pos[i] * lr[i] - abs(pos[i] - prev) * comm
            prev = pos[i]
            j += 1
    s = 0.0
    for k in range(n):
        s += net[k]
    mean = s / n
    if n < 2:
        return 0.0
    v = 0.0
    for k in range(n):
        d = net[k] - mean
        v += d * d
    sd = np.sqrt(v / (n - 1))
    if sd < 1e-12:
        return 0.0
    return mean / sd * ann


def patch(m):
    import strategies_walkforward as sw

    def calc_sharpe_comm_fast(positions, log_ret, mask, ann_factor):
        return _sharpe_comm(np.ascontiguousarray(positions, dtype=np.float64),
                            np.ascontiguousarray(log_ret, dtype=np.float64),
                            np.ascontiguousarray(mask), float(ann_factor), float(sw.COMMISSION))

    sw.calc_sharpe_comm = calc_sharpe_comm_fast
    m.calc_sharpe_comm = calc_sharpe_comm_fast
