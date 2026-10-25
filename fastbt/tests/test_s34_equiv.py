"""Эквивалентность S3/S4 старому коду c22a4b3: сигналы, выходы (побитово), сила (1e-12).

На каждом из 5 тикеров: вся сигнальная сетка S3 и S4 + 50 случайных наборов параметров
(в том числе вне сетки: окна/периоды/множитель), проверка через check_family.
Дополнительно: ADX IRAO побитово равен старому (искажение до 29.07.2015 сохраняется как есть).
"""
import numpy as np
import pytest

from fastbt import data
from fastbt.indicators import compute_base
from fastbt.legacy import load_legacy
from fastbt.signals.interface import check_family, get_family

TICKERS = ["SBER", "IRAO", "MOEX", "MTLR", "LSRG"]


def _random_grid(sid, seed, k=50):
    rng = np.random.default_rng(seed)
    if sid == "S3":
        return [{"dc_window": int(w)} for w in rng.integers(3, 61, k)]
    return [{"atr_period": int(p), "multiplier": float(round(m, 3))}
            for p, m in zip(rng.integers(3, 41, k), rng.uniform(1.0, 5.0, k))]


@pytest.fixture(scope="module")
def ctx():
    cache = {}

    def get(tk):
        if tk not in cache:
            td = data.load_ticker(tk)
            cache[tk] = (td, compute_base(td.close, td.high, td.low, td.open, td.volume))
        return cache[tk]
    return get


@pytest.mark.parametrize("sid", ["S3", "S4"])
@pytest.mark.parametrize("tk", TICKERS)
def test_grid_and_random(sid, tk, ctx):
    td, ind = ctx(tk)
    fam = get_family(sid)
    assert check_family(fam, td, ind)["bad"] == []
    grid = _random_grid(sid, seed=sum(map(ord, sid + tk)))
    r = check_family(fam, td, ind, grid)
    assert r["n"] == 50 and r["bad"] == []


def test_grid_shape_and_warmup(ctx):
    td, ind = ctx("SBER")
    for sid in ("S3", "S4"):
        fam = get_family(sid)
        ss = fam.generate(td, ind, fam.params())
        assert ss.sig.dtype == np.int8 and ss.sig.shape == (len(fam.params()), td.n)
        assert not ss.sig[:, :200].any() and ss.split
        assert len(fam.params()) == {"S3": 4, "S4": 12}[sid]


def test_irao_adx_as_is(ctx):
    """ADX IRAO после скачка 20.01.2015 не чинится: ind['adx14'] == SW.calc_adx."""
    td, ind = ctx("IRAO")
    L = load_legacy()
    old = L.sw.calc_adx(td.high, td.low, td.close, 14)
    assert np.array_equal(ind["adx14"], old, equal_nan=True)
