"""EMA на ряду с NaN прогрева (rel_td_ind, progrev=True): старт с первого конечного значения, без подстановок (Д4).

До правки out[0] = x[0] = NaN заражал рекурсию до конца ряда: S1/S2 с ma_tip=ema/dema/tema на rel-ряду
не давали ни одного сигнала. При конечном x[0] результат побайтно прежний (v5 не меняется).
"""
import numpy as np
import pytest

from fastbt import ma_tipy as M
from fastbt.filtry_v5 import calc_ema
from fastbt.indicators import compute_base
from fastbt.signals import v5_s12 as v5
from fastbt.signals.interface import TickerData

T, K = 700, 54
RNG = np.random.default_rng(5)
X = 100 * np.exp(np.cumsum(RNG.normal(0, 0.02, T)))
XN = np.r_[np.full(K, np.nan), X[K:]]


def src_ema(arr, w):   # эталон: формула источника, старт x[0]
    a = 2.0 / (w + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


@pytest.mark.parametrize("numba", [False, True])
def test_ema_nan_proglev_start_s_pervogo_konechnogo(numba):
    out = M.ma(XN, "ema", 20, numba=numba)
    assert np.isnan(out[:K]).all()
    assert np.isfinite(out[K:]).all()
    assert out[K:].tobytes() == src_ema(X[K:], 20).tobytes()   # известный ответ: ряд, начатый в K


@pytest.mark.parametrize("tip", ["ema", "dema", "tema"])
@pytest.mark.parametrize("numba", [False, True])
def test_nan_ne_zarazhaet_ryad(tip, numba):
    out = M.ma(XN, tip, 20, numba=numba)
    assert np.isnan(out[:K]).all() and np.isfinite(out[K:]).all()


def test_filtry_v5_calc_ema_to_zhe():
    assert np.isnan(calc_ema(XN, 20)[:K]).all() and np.isfinite(calc_ema(XN, 20)[K:]).all()


@pytest.mark.parametrize("tip", ["ema", "dema", "tema"])
@pytest.mark.parametrize("w", [5, 20, 60])
def test_konechnyj_start_pobajtno_prezhnij(tip, w):
    e1 = src_ema(X, w)
    e2 = src_ema(e1, w)
    ref = {"ema": e1, "dema": 2 * e1 - e2, "tema": 3 * e1 - 3 * e2 + src_ema(e2, w)}[tip]
    assert M.ma(X, tip, w).tobytes() == ref.tobytes()
    assert np.allclose(M.ma(X, tip, w, numba=True), ref, rtol=0, atol=1e-9)


def test_vse_nan_ostayutsya_nan():
    for numba in (False, True):
        assert np.isnan(M.ma(np.full(50, np.nan), "ema", 10, numba=numba)).all()


def test_prichinnost_ema_ne_vidit_budushchego():
    a = M.ema(XN, 20)
    b = XN.copy()
    b[400:] *= 3.0
    assert M.ema(b, 20)[:400].tobytes() == a[:400].tobytes()
    # мутация: засев средним по всему ряду (заглядывание вперёд) — тест обязан её ловить
    seeded = XN.copy()
    seeded[:K] = np.nanmean(XN)
    assert not np.array_equal(M.ema(seeded, 20)[K:], a[K:])
    seeded2 = XN.copy()
    seeded2[:K] = np.nanmean(b)
    assert not np.array_equal(M.ema(seeded2, 20)[:400], a[:400])


def _td(c):
    n = len(c)
    h, lo = c * 1.005, c * 0.995
    o, v = np.r_[c[0], c[:-1]], np.full(n, 3e3)
    lr = np.zeros(n)
    td = TickerData("X", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), o, h, lo, c, v, lr)
    return td, compute_base(c, h, lo, o, v)


@pytest.mark.parametrize("gen,key,grid", [
    (v5.generate_s1, "ma_window", dict(dev_tip="std", z_entry=1.0, z_exit=0.5)),
    (v5.generate_s2, "bb_window", dict(dev_tip="std", bb_std=1.0, podtv="none")),
])
def test_signaly_ema_na_rel_ryade_ravny_ryadu_bez_progreva(gen, key, grid):
    w = 20
    p = [dict(ma_tip="ema", **{key: w}, **grid)]
    full = gen(*_td(XN), p).sig[0]
    cut = gen(*_td(X[K:]), p).sig[0]
    assert (full[:K] == 0).all()
    assert int(np.abs(cut).sum()) > 0                       # сигналы на ряду без прогрева есть
    assert int(np.abs(full).sum()) > 0                      # до правки было 0
    assert np.array_equal(full[K + 252:], cut[252:])        # после прогрева std/MA — те же дни и знаки


def test_sma_i_ema_oba_zhivye_na_rel_ryade():
    """Перекос sma 690 / ema 0 (причина дефекта) невозможен: оба типа дают сигналы одного порядка."""
    n = {}
    for tip in ("sma", "ema"):
        p = [dict(ma_tip=tip, ma_window=20, dev_tip="std", z_entry=1.0, z_exit=0.5)]
        n[tip] = int(np.abs(v5.generate_s1(*_td(XN), p).sig[0]).sum())
    assert n["ema"] > 0 and n["sma"] > 0
    assert 0.3 < n["ema"] / n["sma"] < 3.0
