"""V6-5 (К4, ось norm ∈ {okno, s252}): S1, S2, S4, S6 в fastbt/signals/v5_s12.py, v5_s34.py, v5_s56.py.

Синтетика, без сети и данных. «okno» (и отсутствие ключа) = v5 побайтно; «s252» = масштаб за 252 дня,
эталон считается независимо через pandas. Мутации: масштаб окна вместо 252; масштаб с баром t+1 (причинность).
"""
import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import v5_s12 as S12
from fastbt.signals import v5_s34 as S34
from fastbt.signals import v5_s56 as S56
from fastbt.signals.interface import WARMUP, TickerData
from fastbt.signals.s34 import supertrend

N = 900


def mk(n=N, seed=5):
    """Спокойные первые 450 дней, бурные — следующие (режим меняет масштаб окна и 252 по-разному)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, n) * np.where(np.arange(n) < 450, 0.008, 0.03)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    v = rng.uniform(1e5, 5e5, n)
    d = pd.bdate_range("2018-01-01", periods=n).values.astype("datetime64[ns]")
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    td = TickerData("SYN", d, o, h, lo, c, v, lr)
    return td, compute_base(c, h, lo, o, v)


def _same(a, b):
    assert np.array_equal(a.sig, b.sig) and np.array_equal(a.exit_l, b.exit_l)
    assert np.array_equal(a.exit_s, b.exit_s) and a.strength.tobytes() == b.strength.tobytes()


G1 = [dict(ma_tip="sma", ma_window=w, dev_tip="std", z_entry=z, z_exit=0.5) for w in (20, 50) for z in (1.0, 2.0)]
G2 = [dict(ma_tip="sma", bb_window=w, bb_std=k, podtv="none") for w in (20, 50) for k in (1.5, 2.0)]
G4 = [dict(period=p, mnozh=m) for p in (10, 20) for m in (1.5, 3.0)]
G6 = [dict(vwap_window=w, dev_mult=m, dir_bias="OFF", filter_set="net") for w in (10, 30) for m in (1.0, 2.0)]


@pytest.mark.parametrize("gen,grid", [(S12.generate_s1, G1), (S12.generate_s2, G2),
                                      (S34.generate_s4_v5, G4), (S56.generate6, G6)])
def test_okno_ravno_v5_pobaytno_i_s252_menyaet(gen, grid):
    td, ind = mk()
    base = gen(td, ind, grid)
    assert (base.sig != 0).any()
    _same(gen(td, ind, [dict(p, norm="okno") for p in grid]), base)
    s252 = gen(td, ind, [dict(p, norm="s252") for p in grid])
    assert not np.array_equal(s252.sig, base.sig)       # ось действует


def test_s1_s252_ravno_nezavisimomu_etalonu():
    td, ind = mk()
    c = pd.Series(td.close)
    for p in G1:
        ma = c.rolling(p["ma_window"]).mean().values
        z = (td.close - ma) / c.rolling(252).std(ddof=1).values
        ref = np.zeros(N, np.int8)
        ref[(z < -p["z_entry"])] = 1
        ref[(z > p["z_entry"])] = -1
        ref[:WARMUP] = 0
        got = S12.generate_s1(td, ind, [dict(p, norm="s252")]).sig[0]
        assert np.array_equal(got, ref) and (got != 0).any()


def test_s1_s252_atr_i_s2_s252_etalon():
    td, ind = mk()
    c = pd.Series(td.close)
    tr = np.maximum(td.high - td.low, np.maximum(abs(td.high - np.r_[td.close[0], td.close[:-1]]),
                                                 abs(td.low - np.r_[td.close[0], td.close[:-1]])))
    atr = pd.Series(tr).rolling(252).mean().values
    ma = c.rolling(20).mean().values
    ref = np.zeros(N, np.int8)
    z = (td.close - ma) / atr
    ref[z < -1.0] = 1
    ref[z > 1.0] = -1
    ref[:WARMUP] = 0
    p = dict(ma_tip="sma", ma_window=20, dev_tip="atr", z_entry=1.0, z_exit="net", norm="s252")
    got = S12.generate_s1(td, ind, [p]).sig[0]
    assert np.array_equal(got, ref) and (got != 0).any()
    sd = c.rolling(252).std(ddof=1).values
    lo, up = ma - 2.0 * sd, ma + 2.0 * sd
    ref2 = np.where(td.close <= lo, 1, np.where(td.close >= up, -1, 0)).astype(np.int8)
    ref2[:WARMUP] = 0
    got2 = S12.generate_s2(td, ind, [dict(G2[1], norm="s252")]).sig[0]
    assert np.array_equal(got2, ref2) and (got2 != 0).any()


def test_s4_s252_eto_supertrend_252_period_neupodvizhen():
    td, ind = mk()
    a = S34.generate_s4_v5(td, ind, [dict(period=10, mnozh=2.0, norm="s252")])
    b = S34.generate_s4_v5(td, ind, [dict(period=40, mnozh=2.0, norm="s252")])
    _same(a, b)
    st = supertrend(td.high, td.low, td.close, 252, 2.0)
    assert np.isnan(st[:252]).all() and not np.isnan(st[252:]).any()
    assert (a.sig[0, :252] == 0).all() and (a.sig[0] != 0).any()


def test_s6_s252_etalon_pandas():
    td, ind = mk()
    w, m = 10, 1.0
    tpv = (td.high + td.low + td.close) / 3.0 * td.volume
    vwap = pd.Series(tpv).rolling(w).sum().values / pd.Series(td.volume).rolling(w).sum().values
    dev = pd.Series(td.close - vwap).rolling(252).std(ddof=1).values
    dev = np.where(dev < 1e-12, 1e-12, dev)
    with np.errstate(invalid="ignore"):
        ref = np.where(td.close < vwap - m * dev, 1, np.where(td.close > vwap + m * dev, -1, 0)).astype(np.int8)
    ref[:WARMUP] = 0
    got = S56.generate6(td, ind, [dict(G6[0], norm="s252")]).sig[0]
    assert np.array_equal(got, ref) and (got != 0).any()
    assert (got[:252] == 0).all()    # 252 дней истории нет — входов нет (а не dev = 1e-12 и вход на каждом баре)


def test_mutaciya_masshtab_okna_vmesto_252_lovitsya():
    """Если бы s252 считался по окну w, сигналы совпали бы с «okno»; тест это запрещает."""
    td, ind = mk()
    p = G1[0]
    okno = S12.generate_s1(td, ind, [p]).sig[0]
    s252 = S12.generate_s1(td, ind, [dict(p, norm="s252")]).sig[0]
    assert (okno != s252).sum() > 10


@pytest.mark.parametrize("gen,grid", [(S12.generate_s1, G1), (S12.generate_s2, G2),
                                      (S34.generate_s4_v5, G4), (S56.generate6, G6)])
def test_prichinnost_s252_prefiks(gen, grid):
    """Мутация «масштаб с баром t+1»: сигнал на t не меняется при отрезании будущего."""
    td, ind = mk()
    pts = [dict(p, norm="s252") for p in grid]
    full = gen(td, ind, pts).sig
    cut = 700
    td2, ind2 = mk(n=N)
    td2 = TickerData(td.ticker, td.dates[:cut], td.open[:cut], td.high[:cut], td.low[:cut], td.close[:cut],
                     td.volume[:cut], td.log_ret[:cut] if hasattr(td, "log_ret") else td.lr[:cut])
    ind2 = compute_base(td2.close, td2.high, td2.low, td2.open, td2.volume)
    assert np.array_equal(gen(td2, ind2, pts).sig, full[:, :cut])


def test_neizvestnaya_norm_otkaz():
    td, ind = mk()
    for gen, g in ((S12.generate_s1, G1), (S12.generate_s2, G2), (S34.generate_s4_v5, G4), (S56.generate6, G6)):
        with pytest.raises(ValueError):
            gen(td, ind, [dict(g[0], norm="s126")])
