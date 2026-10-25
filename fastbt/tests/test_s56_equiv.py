"""S5/S6: побитовая эквивалентность сигналов старому коду c22a4b3 (V41: S5R/S6R).

Для каждого из 5 тикеров: (а) вся сетка (12 для S5, 128 для S6) и (б) 50 случайных наборов
параметров (подсетка из всех комбинаций S5 ∪ S6, seed=0). Сигналы/выходы — побитово, сила — 1e-12.
Запуск: python3 -m pytest fastbt/tests/test_s56_equiv.py  (или python3 fastbt/tests/test_s56_equiv.py)
"""
import random
import time

import numpy as np
import pytest

from fastbt.data import load_ticker
from fastbt.indicators import compute_base
from fastbt.signals.interface import check_family, get_family

TICKERS = ["SBER", "IRAO", "MOEX", "MTLR", "LSRG"]


def _ind(td):
    return compute_base(td.close, td.high, td.low, td.open, td.volume)


@pytest.mark.parametrize("ticker", TICKERS)
@pytest.mark.parametrize("sid", ["S5", "S6"])
def test_full_grid(sid, ticker):
    td = load_ticker(ticker)
    r = check_family(get_family(sid), td, _ind(td))
    assert r["bad"] == [] and r["n"] == (12 if sid == "S5" else 128)


@pytest.mark.parametrize("ticker", TICKERS)
def test_random_50(ticker):
    td = load_ticker(ticker)
    ind = _ind(td)
    pool = [("S5", p) for p in get_family("S5").params()] + [("S6", p) for p in get_family("S6").params()]
    pick = random.Random(0).sample(pool, 50)
    for sid in ("S5", "S6"):
        sub = [p for s, p in pick if s == sid]
        r = check_family(get_family(sid), td, ind, grid=sub)
        assert r["bad"] == [] and r["n"] == len(sub)


def test_shapes_and_warmup():
    td = load_ticker("SBER")
    for sid in ("S5", "S6"):
        f = get_family(sid)
        ss = f.generate(td, _ind(td), f.params())
        assert ss.sig.dtype == np.int8 and ss.sig.shape == (len(f.params()), td.n)
        assert not ss.sig[:, :200].any() and not ss.split


def grid_time(ticker="SBER"):
    td = load_ticker(ticker)
    ind = _ind(td)
    out = {}
    for sid in ("S5", "S6"):
        f = get_family(sid)
        f.generate(td, ind, f.params()[:2])           # прогрев numba/кэша
        t = time.perf_counter()
        f.generate(td, ind, f.params())
        out[sid] = time.perf_counter() - t
    return out


if __name__ == "__main__":
    print("время генерации сетки, с:", grid_time())
    pytest.main([__file__, "-q"])
