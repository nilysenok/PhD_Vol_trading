"""compute_base (fastbt/indicators.py) на ряду с ведущим NaN-префиксом (прогрев rel): индикаторы считаются на конечной части
и дополняются NaN назад; без префикса — побайтно прежнее. Баг: adx14 = 0.0 везде, ADX>20 отсекал все сигналы S4 на sektor/rynok."""
import os

import numpy as np
import pytest

from fastbt import indicators as I
from fastbt.signals import s34

K = 54


def ohlcv(n=500, seed=5):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    o = c * (1 + rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    return c, h, l, o, rng.uniform(1e3, 5e3, n)


def s_nan():
    c, h, l, o, v = ohlcv()
    for x in (c, h, l, o, v):
        x[:K] = np.nan
    return c, h, l, o, v


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_prefiks_ravno_ryadu_bez_prefiksa_sdvinutomu():
    c, h, l, o, v = s_nan()
    ind = I.compute_base(c, h, l, o, v)
    ref = I._compute_base0(c[K:], h[K:], l[K:], o[K:], v[K:])
    assert set(ind) == set(ref)
    for k in ref:
        if k == "_volume":
            continue
        assert len(ind[k]) == len(c), k
        assert same(ind[k][K:], ref[k]), k
        if ref[k].dtype != bool:
            assert np.isnan(ind[k][:K]).all(), k
        else:
            assert ind[k][:K].all(), k


def test_adx_konechen_posle_progreva_i_ne_nol():
    c, h, l, o, v = s_nan()
    ind = I.compute_base(c, h, l, o, v)
    adx = ind["adx14"]
    assert np.isfinite(adx[K + 40:]).all() and (adx[K + 40:] > 0).all()
    assert (ind["adx14"][K + 40:] > 20).any()
    for k in ("atr14", "bb_width", "sma20_slope10", "vol_sma20"):
        assert np.isfinite(ind[k][K + 60:]).all(), k


def test_supertrend_na_prefikse_soglasovan():
    c, h, l, o, v = s_nan()
    st = s34.supertrend(h, l, c, 14, 3.0)
    assert np.isfinite(st[K + 14:]).all()


def test_bez_prefiksa_pobaytno_prezhnee():
    c, h, l, o, v = ohlcv()
    a, b = I.compute_base(c, h, l, o, v), I._compute_base0(c, h, l, o, v)
    assert set(a) == set(b) and all(same(a[k], b[k]) for k in a)


def test_mutaciya_srez_bez_vozvrata_nan_na_mesto_lovitsya():
    """Мутация: срез без дополнения NaN впереди (длина короче, даты сдвинуты) — тест на длину и позицию падает."""
    c, h, l, o, v = s_nan()
    bad = I._compute_base0(c[K:], h[K:], l[K:], o[K:], v[K:])
    good = I.compute_base(c, h, l, o, v)
    assert len(bad["adx14"]) != len(good["adx14"])
    shifted = np.concatenate([bad["adx14"], np.full(K, np.nan)])      # NaN в хвосте вместо головы
    assert not same(shifted, good["adx14"])


def test_vse_nan_i_pustoj():
    x = np.full(30, np.nan)
    ind = I.compute_base(x, x, x, x, x)
    assert len(ind["adx14"]) == 30
    e = np.array([])
    assert len(I.compute_base(e, e, e, e, e)["atr14"]) == 0


def test_real_syroy_pobaytno_prezhnee():
    if not os.environ.get("FASTBT_MAIN"):
        pytest.skip("нет FASTBT_MAIN")
    from fastbt import daily_1850, etap1
    if not daily_1850.out_path().exists():
        pytest.skip("нет дневного ряда 18:50")
    inp = etap1.build_inputs("S4", period="train4")
    for t in list(inp.sk.tickers)[:5]:
        td, ind = inp.sk._td[t]
        ref = I._compute_base0(td.close, td.high, td.low, td.open, td.volume)
        assert set(ind) == set(ref) and all(same(ind[k], ref[k]) for k in ref), t
