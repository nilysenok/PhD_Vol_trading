"""К3 (Ъ владельца 27.10, блок 01:30; к2 — постановка v3 §0.4): контроли без прогноза для точек C С9 v2.

к1 снят: Шарп не зависит от масштаба позиции (A × c даёт Шарп A) — контроль вырожден.
к2 — круговой сдвиг маски экспозиции точки по каждому тикеру на случайный лаг ∈ [21, T−21],
     независимо по тикерам; маска — затвор g точки по всем торговым дням 2018–2021 тикера (T — длина
     этого ряда), все стратегии тикера сдвигаются одним лагом; позиция к2 = A × сдвинутая маска.
     Оборот, длины серий и доля дней сохраняются (на круге). 1 000 повторов, seed 20261027,
     одни и те же лаги для всех точек. T тикера — наименьшая длина ряда среди его пар (обычно равны);
     T < 42 — допустимого лага нет, тикер остаётся на месте (консервативно: p не занижается).
     p = (1 + #{Шарп сдвига ≥ Шарп точки}) / (N + 1).
к3 — A без диапазонных стратегий S5/S6: позиции A, в которых S5 и S6 обнулены на всех днях;
     комиссия и ставка — как у A (обычный `s9.Engine.port`); здесь — только маска позиций.

Шарп сдвига — ядро `shift_sharpe`: арифметика `s9.Engine.port` (wgt=None, cap=1) →
`Engine._finish` → `msp.sharpe` (ddof=1) в том же порядке операций; при лаге 0 совпадает с Шарпом
точки до ~1e-12 (тест `tests/test_s9_c8v2.py`). Старые пути не меняются.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from numba import njit, prange

SEED = 20261027
N_SHIFT = 1000
MIN_LAG = 21
K3_YAML = Path(__file__).resolve().parent / "designs" / "s9" / "k3.yaml"


def k3_sids(path=K3_YAML) -> tuple:
    """Семьи, исключаемые из A в к3, — из конфига `designs/s9/k3.yaml` (не из кода)."""
    return tuple(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["isklyuchit_semyi"])


RANGE_SIDS = k3_sids()


def draw_lags(t_tick: np.ndarray, n_shift: int = N_SHIFT, seed: int = SEED,
              min_lag: int = MIN_LAG) -> np.ndarray:
    """LAG[b, тикер] — равномерно по целым в [min_lag, T−min_lag], независимо по тикерам.
    t_tick — длина ряда тикера; T < 2·min_lag — лаг 0. int64 [n_shift, n_тикеров]."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.zeros((n_shift, len(t_tick)), np.int64)
    for ti, T in enumerate(np.asarray(t_tick, np.int64)):
        if T >= 2 * min_lag:
            out[:, ti] = rng.integers(min_lag, T - min_lag + 1, n_shift)
    return out


def src_rows(lag: np.ndarray, tick, pstart, plen, loc) -> np.ndarray:
    """Номер строки-источника для сдвинутой маски: g'[j] = g[(j − lag) mod T] внутри своей пары;
    lag[тикер] — один на все пары тикера."""
    return pstart + (loc - lag[tick]) % plen


def check_share(g: np.ndarray, src: np.ndarray, pair: np.ndarray) -> None:
    """Сдвиг сохраняет в каждой паре мультимножество уровней затвора (доля дней) и число
    переключений на круге (оборот и длины серий) — иначе ошибка."""
    g, s = np.asarray(g, float), np.asarray(g, float)[src]
    pair = np.asarray(pair, np.int64)
    for name, x in (("доля/уровни", np.sort), ("переключения", None)):
        for p in np.unique(pair):
            m = pair == p
            a, b = g[m], s[m]
            if x is not None:
                bad = not np.array_equal(x(a), x(b))
            else:
                bad = np.sum(a != np.roll(a, 1)) != np.sum(b != np.roll(b, 1))
            if bad:
                raise AssertionError(f"к2: сдвиг меняет {name} в паре {p}")


def k3_positions(A: np.ndarray, pair: np.ndarray, pairs: list, sids=RANGE_SIDS) -> np.ndarray:
    """A с обнулёнными позициями стратегий S5/S6 (по номеру пары → (тикер, sid)); остальное — как есть."""
    drop = np.array([sid in sids for _, sid in pairs], bool)
    return np.where(drop[np.asarray(pair, np.int64)], 0.0, np.asarray(A, np.float64))


@njit(cache=True, parallel=True)
def shift_sharpe(A, gr, tick, pstart, plen, loc, LAG, first, last, date, nd, lr, cnt, rf, kr, cal,
                 comm, out):
    """out[b] — Шарп избыточной доходности портфеля A·g'_b, g'_b[i] = gr[pstart[i] + (loc[i] −
    LAG[b, tick[i]]) mod plen[i]]. Порядок операций — s9.Engine.port / _finish / msp.sharpe."""
    B = LAG.shape[0]
    n = A.shape[0]
    k252 = math.sqrt(252.0)
    for b in prange(B):
        rs = np.zeros(nd)
        es = np.zeros(nd)
        prev = 0.0
        for i in range(n):
            T = plen[i]
            j = (loc[i] - LAG[b, tick[i]]) % T
            if j < 0:
                j += T
            p = A[i] * gr[pstart[i] + j]
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
                 min_lag: int = MIN_LAG):
        self.e = eng
        self.comm = float(comm)
        self.A = np.ascontiguousarray(eng.A_row, np.float64)
        self.pair = np.asarray(eng.r["pair"], np.int64)
        self.tick = np.asarray(eng.r["tick"], np.int64)
        first = np.asarray(eng.first, bool)
        self.first = first
        self.last = np.r_[first[1:], True]
        starts = np.flatnonzero(first)
        lens = np.diff(np.r_[starts, len(first)])
        self.pstart = np.repeat(starts, lens).astype(np.int64)
        self.plen = np.repeat(lens, lens).astype(np.int64)
        self.loc = (np.arange(len(first)) - self.pstart).astype(np.int64)
        nt = int(self.tick.max()) + 1
        t_tick = np.full(nt, np.iinfo(np.int64).max)
        np.minimum.at(t_tick, self.tick[starts], lens)
        t_tick[t_tick == np.iinfo(np.int64).max] = 0
        self.t_tick = t_tick
        self.LAG = draw_lags(t_tick, n_shift, seed, min_lag)
        self.n_shift, self.seed = n_shift, seed
        self.lr = np.ascontiguousarray(eng.lr_row, np.float64)
        self.date = np.ascontiguousarray(eng.r["date"], np.int64)
        self.nd = len(eng.dates)
        self.cnt = np.ascontiguousarray(eng.cnt, np.float64)
        self.rf = np.ascontiguousarray(eng.rf, np.float64)
        self.kr = np.ascontiguousarray(eng.kr, np.float64)
        self.cal = np.ascontiguousarray(eng.cal, np.float64)

    def src(self, b: int) -> np.ndarray:
        return src_rows(self.LAG[b], self.tick, self.pstart, self.plen, self.loc)

    def check(self, g: np.ndarray) -> None:
        for b in (0, self.n_shift // 2, self.n_shift - 1):
            check_share(g, self.src(b), self.pair)

    def k3_pos(self) -> np.ndarray:
        return k3_positions(self.A, self.pair, self.e.pairs)

    def k2_sharpes(self, g: np.ndarray, LAG: np.ndarray | None = None) -> np.ndarray:
        """Шарпы позиций A × сдвинутая маска g (g — затвор точки по строкам пара-день, все дни)."""
        gr = np.ascontiguousarray(g, np.float64)
        LAG = self.LAG if LAG is None else LAG
        out = np.zeros(LAG.shape[0])
        shift_sharpe(self.A, gr, self.tick, self.pstart, self.plen, self.loc, LAG, self.first,
                     self.last, self.date, self.nd, self.lr, self.cnt, self.rf, self.kr, self.cal,
                     self.comm, out)
        return out

    def identity_sharpe(self, g: np.ndarray) -> float:
        return float(self.k2_sharpes(g, np.zeros((1, self.LAG.shape[1]), np.int64))[0])


def p_perm(sh_point: float, sh_perm: np.ndarray) -> float:
    """p = (1 + #{Шарп сдвига ≥ Шарп точки}) / (N + 1)."""
    sh_perm = np.asarray(sh_perm, float)
    return float((1 + np.sum(sh_perm >= sh_point)) / (len(sh_perm) + 1))
