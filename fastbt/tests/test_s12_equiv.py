"""S1/S2: сигналы fastbt == старая функция c22a4b3 побитово (сигналы, выходы), сила до 1e-12.

5 тикеров; полная сетка + 50 случайных наборов сигнальных параметров на семейство
(окна и пороги вне сетки тоже), включая прогрев WARMUP и NaN-края.
Запуск: python3 -m pytest fastbt/tests/test_s12_equiv.py -q
"""
import numpy as np
import pytest

from fastbt.data import load_ticker
from fastbt.indicators import compute_base
from fastbt.signals.interface import WARMUP, check_family, get_family

TICKERS = ["SBER", "IRAO", "MOEX", "MTLR", "LSRG"]


def _random_grid(sid, rng, k=50):
    if sid == "S1":
        return [dict(ma_window=int(rng.integers(5, 61)), z_entry=float(rng.uniform(0.5, 3.5)),
                     z_exit=float(rng.uniform(0.1, 1.5))) for _ in range(k)]
    return [dict(bb_window=int(rng.integers(5, 61)), bb_std=float(rng.uniform(0.5, 3.5)))
            for _ in range(k)]


@pytest.mark.parametrize("sid", ["S1", "S2"])
@pytest.mark.parametrize("ticker", TICKERS)
def test_equiv(sid, ticker):
    fam = get_family(sid)
    td = load_ticker(ticker)
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    rng = np.random.default_rng(sum(map(ord, sid + ticker)))
    grid = fam.params() + _random_grid(sid, rng)
    r = check_family(fam, td, ind, grid)
    assert r["bad"] == [], r
    ss = fam.generate(td, ind, grid)
    assert ss.sig.dtype == np.int8 and not ss.sig[:, :WARMUP].any()
    assert ss.sig.shape == (len(grid), td.n)
    assert (ss.sig != 0).any()
