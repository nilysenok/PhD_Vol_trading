"""К3 (Ъ владельца 27.10, блок 01:30): контроли без прогноза для точек C С9 v2 — быстрый слой.

к1 снят: Шарп не зависит от масштаба позиции (A × c даёт Шарп A) — контроль вырожден.
к2 — круговой сдвиг собственной маски точки: внутри каждой пары тикер × стратегия ряд затвора
     g = позиция / A на днях с A ≠ 0 сдвигается по кругу на случайное k; k равномерно по допустимым
     сдвигам, |k| < MIN_SHIFT дней исключены. Серии и оборот сохраняются (до шва круга). Сдвиги одни
     и те же для всех точек (индексы строятся один раз, seed фиксирован).
     p = (1 + #{Шарп сдвига ≥ Шарп точки}) / (N + 1). Пара с числом дней A ≠ 0 меньше 2·MIN_SHIFT
     допустимого сдвига не имеет — остаётся на месте (консервативно: p не занижается).
к3 — A без диапазонных стратегий S5/S6: позиции A, в которых S5 и S6 обнулены на всех днях;
     комиссия и ставка — как у A (обычный `s9.Engine.port`); здесь — только маска позиций.

Шарп сдвига — ядро `perm_sharpe`: арифметика `s9.Engine.port` (wgt=None, cap=1) →
`Engine._finish` → `msp.sharpe` (ddof=1) в том же порядке операций; на тождественном сдвиге
совпадает с Шарпом точки до ~1e-12 (тест `tests/test_s9_c8v2.py`). Старые пути не меняются.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SEED = 20261027
N_SHIFT = 1000
MIN_SHIFT = 22
RANGE_SIDS = ("S5", "S6")


def shift_index(groups: np.ndarray, n_shift: int = N_SHIFT, seed: int = SEED,
                min_shift: int = MIN_SHIFT) -> np.ndarray:
    """P[b, j] — индекс в массиве активных строк: круговой сдвиг на k_b внутри каждой группы (пары).
    Порядок внутри группы — по возрастанию индекса (время). k ∈ [min_shift, m − min_shift], m — размер
    группы (это все сдвиги по модулю m с |k| ≥ min_shift), равномерно. int32 [n_shift, n_act]."""
    groups = np.asarray(groups, np.int64)
    order0 = np.argsort(groups, kind="stable")
    gs = groups[order0]
    cuts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
    rng = np.random.Generator(np.random.PCG64(seed))
    P = np.empty((n_shift, len(groups)), np.int32)
    for s, e in zip(cuts[:-1], cuts[1:]):
        idx, m = order0[s:e], e - s
        lo, hi = min_shift, m - min_shift
        k = rng.integers(lo, hi + 1, n_shift) if hi >= lo else np.zeros(n_shift, np.int64)
        P[:, idx] = idx[(np.arange(m)[None, :] - k[:, None]) % m]
    return P


def check_share(g_act: np.ndarray, P: np.ndarray, groups: np.ndarray) -> None:
    """Сдвиг сохраняет мультимножество уровней затвора в каждой группе (паре) — иначе ошибка."""
    lv = np.round(np.asarray(g_act, float) * 2).astype(np.int64)      # 0, 1 (=0,5), 2 (=1)
    G = int(groups.max()) + 1
    want = np.zeros((G, 3), np.int64)
    np.add.at(want, (groups, np.clip(lv, 0, 2)), 1)
    for b in (0, len(P) // 2, len(P) - 1):
        got = np.zeros((G, 3), np.int64)
        np.add.at(got, (groups, np.clip(lv[P[b]], 0, 2)), 1)
        if not np.array_equal(got, want):
            raise AssertionError(f"к2: сдвиг {b} меняет долю/уровни затвора внутри пар")


def k3_positions(A: np.ndarray, pair: np.ndarray, pairs: list, sids=RANGE_SIDS) -> np.ndarray:
    """A с обнулёнными позициями стратегий S5/S6 (по номеру пары → (тикер, sid)); остальное — как есть."""
    drop = np.array([sid in sids for _, sid in pairs], bool)
    return np.where(drop[np.asarray(pair, np.int64)], 0.0, np.asarray(A, np.float64))


@njit(cache=True, parallel=True)
def perm_sharpe(A, gr, act, P, lr, first, last, date, nd, cnt, rf, kr, cal, comm, out):
    """out[b] — Шарп избыточной доходности портфеля A·g'_b (g'_b[act[j]] = gr[P[b, j]], вне act —
    позиция 0, как у A). Порядок операций — s9.Engine.port / _finish / msp.sharpe."""
    B = P.shape[0]
    n = A.shape[0]
    na = act.shape[0]
    k252 = math.sqrt(252.0)
    for b in prange(B):
        pos = np.zeros(n)
        for j in range(na):
            i = act[j]
            pos[i] = A[i] * gr[P[b, j]]
        rs = np.zeros(nd)
        es = np.zeros(nd)
        prev = 0.0
        for i in range(n):
            p = pos[i]
            dp = p if first[i] else p - prev
            sides = abs(dp) + (abs(p) if last[i] else 0.0)
            net = p * lr[i] - sides * comm
            d = date[i]
            rs[d] += net
            es[d] += abs(p)
            prev = p
        ex = np.empty(nd)
        S = 0.0
        for d in range(nd):
            r = rs[d] / cnt[d]
            used = es[d] / cnt[d]
            fdr = max(0.0, 1.0 - used) * rf[d] * cal[d]
            mg = max(0.0, used - 1.0) * (kr[d] + 0.02) / 365.0 * cal[d]
            total = r + fdr - mg
            ex[d] = total - rf[d] * cal[d]
            S += ex[d]
        m = S / nd
        Q = 0.0
        for d in range(nd):
            Q += (ex[d] - m) ** 2
        sd = math.sqrt(Q / (nd - 1))
        out[b] = m * k252 / sd if sd > 1e-12 else 0.0


class Kontrol:
    """Общие для всех точек данные к2/к3 на движке s9.Engine (строки пара-день 2018–2021)."""

    def __init__(self, eng, comm: float, n_shift: int = N_SHIFT, seed: int = SEED,
                 min_shift: int = MIN_SHIFT):
        self.e = eng
        self.comm = float(comm)
        A = np.ascontiguousarray(eng.A_row, np.float64)
        self.A = A
        self.act = np.flatnonzero(A != 0).astype(np.int64)
        self.groups = eng.r["pair"][self.act].astype(np.int64)
        self.P = shift_index(self.groups, n_shift, seed, min_shift)
        self.n_shift, self.seed = n_shift, seed
        first = np.asarray(eng.first, bool)
        self.first = first
        self.last = np.r_[first[1:], True]
        self.lr = np.ascontiguousarray(eng.lr_row, np.float64)
        self.date = np.ascontiguousarray(eng.r["date"], np.int64)
        self.nd = len(eng.dates)
        self.cnt = np.ascontiguousarray(eng.cnt, np.float64)
        self.rf = np.ascontiguousarray(eng.rf, np.float64)
        self.kr = np.ascontiguousarray(eng.kr, np.float64)
        self.cal = np.ascontiguousarray(eng.cal, np.float64)

    def gate_act(self, pos: np.ndarray) -> np.ndarray:
        """g = позиция / A на днях с A ≠ 0; вне этих дней позиция точки обязана быть 0."""
        pos = np.asarray(pos, np.float64)
        out_ = np.ones(len(pos), bool)
        out_[self.act] = False
        if np.any(pos[out_] != 0):
            raise AssertionError("к2: позиция точки ≠ 0 там, где A = 0 — затвор не выделяется")
        return pos[self.act] / self.A[self.act]

    def k3_pos(self) -> np.ndarray:
        return k3_positions(self.A, self.e.r["pair"], self.e.pairs)

    def k2_sharpes(self, pos: np.ndarray, P: np.ndarray | None = None) -> np.ndarray:
        gr = np.ascontiguousarray(self.gate_act(pos))
        P = self.P if P is None else P
        out = np.zeros(P.shape[0])
        perm_sharpe(self.A, gr, self.act, P, self.lr, self.first, self.last, self.date, self.nd,
                    self.cnt, self.rf, self.kr, self.cal, self.comm, out)
        return out

    def identity_sharpe(self, pos: np.ndarray) -> float:
        P = np.arange(len(self.act), dtype=np.int32)[None, :]
        return float(self.k2_sharpes(pos, P)[0])


def p_perm(sh_point: float, sh_perm: np.ndarray) -> float:
    """p = (1 + #{Шарп сдвига ≥ Шарп точки}) / (N + 1)."""
    sh_perm = np.asarray(sh_perm, float)
    return float((1 + np.sum(sh_perm >= sh_point)) / (len(sh_perm) + 1))
