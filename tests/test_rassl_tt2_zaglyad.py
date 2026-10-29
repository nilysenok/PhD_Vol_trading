"""Расследование 29.10 (находка П3, PR #589): стоп бара t в trend_run строится по high/low/close бара t.

Воспроизведение: (1) tt=2 — стоп подтянут по high бара t и сработал по low того же бара; (2) безубыток по close
бара t сработал по low того же бара; (3) tt=3 — то же через максимум с входа. В правиле «стоп бара t по состоянию
на t−1» (эталон _stop_t_minus_1 ниже) эти бары не закрываются — порядок обновления меняет исход.
Исполнение — по close бара t (спецификация п. 5, tests/audit_fastbt/test_15_gap.py), поэтому решение бара t
использует только данные ≤ close[t]: инвариант причинности (данные после бара k не меняют сделок, закрытых
до k, и не закрывают сделку, открытую на k) проверяется на всех tt; подложенное заглядывание (бар t+1
в стопе бара t) инвариант обязан ловить.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "audit_fastbt"))
from _eng import mutant, pnl, run_trend  # noqa: E402
from fastbt import engine, engine_trend  # noqa: E402

N = 8


def _flat():
    c = np.full(N, 100.0)
    return c, c + 0.5, c - 0.5


def _stop_t_minus_1(c, h, lo, e, isl, ps, pm, bth, ea=1.0):
    """Эталон-альтернатива (лонг, tt=2): стоп бара u — по ext/step/be на конец u−1; выход по close бара u."""
    ep, sl, ext, step, be = c[e], c[e] - isl * ea, h[e], ps, False
    for u in range(e + 1, len(c)):
        tsl = sl + step * (ext - sl)
        if be:
            tsl = max(tsl, ep)
        sl = max(sl, tsl)
        if lo[u] <= sl:
            return u
        if h[u] > ext:
            ext, step = h[u], min(step + ps, pm)
        if not be and bth < 1e10 and c[u] - ep > bth * ea:
            be = True
    return None


def test_tt2_stop_po_high_t_srabotal_po_low_t():
    c, h, lo = _flat()
    c[2], h[2], lo[2] = 105.0, 110.0, 99.0
    # вход 1 по 100, sl0 = 98, ext = 100,5; бар 2: ext → 110, step 0,2 → 0,4, tsl = 98 + 0,4·12 = 102,8 ≥ low 99
    tr = run_trend(c, h, lo, longs=[1], isl=2.0, tt=2, ps=0.2, pm=0.5)
    assert tr == [(1, 2, 1)]
    assert pnl(tr, c) == [pytest.approx(np.log(105.0 / 100.0))]      # исполнение по close бара 2, не по уровню 102,8
    assert _stop_t_minus_1(c, h, lo, 1, 2.0, 0.2, 0.5, 1e12) == 3     # по t−1: стоп 98,5 < low 99 — бар 2 не закрыт


def test_bezubytok_po_close_t_srabotal_po_low_t():
    c, h, lo = _flat()
    c[2], h[2], lo[2] = 102.0, 102.5, 99.8
    # bth = 1: close 102 − 100 > 1·ATR → be в баре 2, tsl = max(tsl, 100) ≥ low 99,8 → выход на баре 2
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=99.0, bth=1.0, tt=0) == [(1, 2, 1)]
    assert _stop_t_minus_1(c, h, lo, 1, 2.0, 0.0, 0.0, 1.0) == 3


def test_tt3_maksimum_s_tekushchim_barom():
    c, h, lo = _flat()
    c[2], h[2], lo[2] = 104.0, 110.0, 99.0
    # ext = 110 (бар 2), tsl = 110 − 2·1 = 108 ≥ low 99 → выход на баре 2; по t−1 стоп 100,5 − 2 = 98,5 < 99
    assert run_trend(c, h, lo, longs=[1], isl=2.0, tmult=2.0, tt=3) == [(1, 2, 1)]


# --- инвариант причинности ---------------------------------------------------------------------------------

def _data(rng, n):
    c = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.012, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.012, n)))
    atr = c * (0.01 + 0.01 * rng.random(n))
    return c, h, lo, atr


def _run(fn, sig, exl, exs, c, h, lo, atr, tt, bth, ps=0.02, pm=0.2, isl=2.0, tmult=2.5):
    n = len(c)
    rh = pd.Series(h).rolling(11, min_periods=1).max().values
    rl = pd.Series(lo).rolling(11, min_periods=1).min().values
    te, tx, td = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int8)
    nt = fn(sig, exl, exs, engine.next_signal(sig), c, h, lo, atr, atr, np.full(n, 0.01), rh, rl,
            0, isl, tmult, bth, tt, ps, pm, 0, 30.0, -1.0, 0.0, 0, n, te, tx, td)
    return [(int(te[i]), int(tx[i]), int(td[i])) for i in range(nt)]


def _narusheniya(fn, tt, bth, seeds=range(3), n=160):
    """Сколько раз данные после бара k (каждый k) изменили сделки, решённые к закрытию бара k."""
    bad = 0
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        c, h, lo, atr = _data(rng, n)
        sig = rng.choice(np.array([-1, 0, 1], np.int8), n, p=[0.1, 0.8, 0.1])
        exl, exs = rng.random(n) < 0.03, rng.random(n) < 0.03
        base = _run(fn, sig, exl, exs, c, h, lo, atr, tt, bth)
        for k in range(15, n - 1):
            c2, h2, lo2, atr2 = _data(np.random.default_rng(7 + s + k), n)
            sc = c[k] / c2[k]
            c2, h2, lo2, atr2 = [np.concatenate([a[:k + 1], b[k + 1:] * sc])
                                 for a, b in ((c, c2), (h, h2), (lo, lo2), (atr, atr2))]
            alt = _run(fn, sig, exl, exs, c2, h2, lo2, atr2, tt, bth)
            done = [t for t in base if t[1] <= k]
            bad += done != [t for t in alt if t[1] <= k]
            opn = [t[0] for t in base if t[0] <= k < t[1]]
            bad += any(t[0] in opn and t[1] <= k for t in alt)
    return bad


@pytest.mark.parametrize("tt", [0, 1, 2, 3])
@pytest.mark.parametrize("bth", [1e12, 1.0])
def test_prichinnost_trend_run(tt, bth):
    assert _narusheniya(engine_trend.trend_run, tt, bth) == 0


NXT = "min(u + 1, end - 1)"
MUT = {
    0: [("tsl = rh[u] - tmult * ca", f"tsl = rh[{NXT}] - tmult * ca", 1)],
    2: [("if high[u] > ext:", f"if high[{NXT}] > ext:", 1), ("ext = high[u]\n", f"ext = high[{NXT}]\n", 1)],
    3: [("ext = max(ext, high[u])", f"ext = max(ext, high[{NXT}])", 1)],
}


@pytest.mark.parametrize("tt", [0, 2, 3])
def test_mutaciya_bar_t_plus_1_v_stope_lovitsya(tt):
    assert _narusheniya(mutant(engine_trend.trend_run, MUT[tt]), tt, 1e12) > 0


def test_mutaciya_bezubytok_po_close_t_plus_1_lovitsya():
    m = mutant(engine_trend.trend_run, [("(close[u] - ep) > bth * ea", f"(close[{NXT}] - ep) > bth * ea", 1)])
    assert _narusheniya(m, 0, 0.5) > 0
