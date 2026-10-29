"""Л3 сетки v5: S3@v5 (okno × podtv), S4@v5 (period × mnozh). Только синтетика, без сети."""
import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import compute_base
from fastbt.signals.interface import WARMUP, TickerData
from fastbt.signals.s34 import generate_s3, generate_s4
from fastbt.signals import v5_s34 as V

N = 700


def _td(seed=1, n=N):
    r = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(r.normal(0, 0.02, n)))
    high = close * (1 + r.uniform(0, 0.02, n))
    low = close * (1 - r.uniform(0, 0.02, n))
    opn = close * (1 + r.normal(0, 0.005, n))
    vol = r.uniform(500, 1500, n)
    td = TickerData("SYN", np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"),
                    opn, high, low, close, vol, np.zeros(n))
    return td, compute_base(close, high, low, opn, vol)


def _ref_gen_sig_s3(close, high, low, w):
    """Дословно gen_sig_s3 @6a7a4636 (strategies_walkforward.py), с pandas-каналом."""
    hc = pd.Series(high).rolling(w).max().values
    lc = pd.Series(low).rolling(w).min().values
    ph = np.empty_like(hc); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty_like(lc); pl[0] = np.nan; pl[1:] = lc[:-1]
    valid = ~np.isnan(ph)
    sig = np.zeros(len(close), dtype=np.int8)
    sig[valid & (close > ph)] = 1; sig[valid & (close < pl)] = -1
    sig[:WARMUP] = 0
    return sig


def _ref_extended(td, ind, w, conf):
    """gen_s3_extended @6a7a4636 (grid_scan_phase3.py:62-117), переписано для проверки."""
    close, high, low, volume = td.close, td.high, td.low, td.volume
    adx, vs20 = ind["adx14"], ind["vol_sma20"]
    hc = pd.Series(high).rolling(w).max().values
    lc = pd.Series(low).rolling(w).min().values
    n = len(close)
    ph = np.empty(n); ph[0] = np.nan; ph[1:] = hc[:-1]
    pl = np.empty(n); pl[0] = np.nan; pl[1:] = lc[:-1]
    v = ~np.isnan(ph)
    sig = np.zeros(n, dtype=np.int8)
    ok = np.ones(n, bool); up_x = np.ones(n, bool); dn_x = np.ones(n, bool)
    if conf == "two_close":
        pa = np.zeros(n, bool); pb = np.zeros(n, bool)
        for i in range(1, n):
            if not np.isnan(ph[i]):
                pa[i] = close[i - 1] > ph[i - 1] if not np.isnan(ph[i - 1]) else False
                pb[i] = close[i - 1] < pl[i - 1] if not np.isnan(pl[i - 1]) else False
        up_x, dn_x = pa, pb
    elif conf == "volume":
        vs = np.where(np.isnan(vs20) | (vs20 < 1), 1.0, vs20)
        ok = volume / vs > 1.0
    elif conf == "adx_above_20":
        ok = (~np.isnan(adx)) & (adx > 20)
    elif conf == "momentum":
        mom = np.zeros(n); mom[5:] = close[5:] - close[:-5]
        up_x, dn_x = mom > 0, mom < 0
    sig[v & (close > ph) & ok & up_x] = 1
    sig[v & (close < pl) & ok & dn_x] = -1
    sig[:WARMUP] = 0
    return sig


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_s3_none_ravno_gen_sig_s3(seed):
    td, ind = _td(seed)
    grid = [{"okno": w, "podtv": p} for w in V.OKNO for p in ("none", "close_confirm")]
    ss = V.generate_s3_v5(td, ind, grid)
    assert (ss.sig != 0).sum() > 0
    for k, p in enumerate(grid):
        assert np.array_equal(ss.sig[k], _ref_gen_sig_s3(td.close, td.high, td.low, p["okno"]))


@pytest.mark.parametrize("podtv", V.PODTV)
def test_s3_vse_podtv_ravny_extended(podtv):
    td, ind = _td(5)
    grid = [{"okno": w, "podtv": podtv} for w in (5, 20, 60)]
    ss = V.generate_s3_v5(td, ind, grid)
    for k, p in enumerate(grid):
        assert np.array_equal(ss.sig[k], _ref_extended(td, ind, p["okno"], podtv))


def test_podtv_realno_menyayut_signal():
    td, ind = _td(5)
    base = V.generate_s3_v5(td, ind, [{"okno": 20, "podtv": "none"}]).sig[0]
    reduced = 0
    for p in ("two_close", "volume", "adx_above_20", "momentum"):
        s = V.generate_s3_v5(td, ind, [{"okno": 20, "podtv": p}]).sig[0]
        assert (s != 0).sum() <= (base != 0).sum() and np.all((s == 0) | (s == base))
        reduced += (s != 0).sum() < (base != 0).sum()
    assert reduced >= 3


def test_none_snimaet_zashitye_filtry_v4():
    """v4 держит ADX>20/объём/vf_pass: v5 none даёт строго больше входов и содержит все v4."""
    td, ind = _td(7)
    v4 = generate_s3(td, ind, [{"dc_window": 20}]).sig[0]
    v5 = V.generate_s3_v5(td, ind, [{"okno": 20, "podtv": "none"}]).sig[0]
    assert (v5 != 0).sum() > (v4 != 0).sum()
    assert np.all((v4 == 0) | (v4 == v5))


def test_s3_vyhody_i_sila_kak_v4():
    td, ind = _td(8)
    a = generate_s3(td, ind, [{"dc_window": 25}])
    b = V.generate_s3_v5(td, ind, [{"okno": 25, "podtv": "none"}])
    assert np.array_equal(a.exit_l, b.exit_l) and np.array_equal(a.exit_s, b.exit_s)
    assert np.array_equal(a.strength, b.strength)


def test_s4_na_setke_v4_pobaytno():
    td, ind = _td(9)
    g4 = [{"atr_period": a, "multiplier": m} for a in (10, 14, 20) for m in (2.0, 2.5, 3.0, 3.5)]
    g5 = [{"period": p["atr_period"], "mnozh": p["multiplier"]} for p in g4]
    a, b = generate_s4(td, ind, g4), V.generate_s4_v5(td, ind, g5)
    assert (a.sig != 0).any()
    for f in ("sig", "exit_l", "exit_s", "strength"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
    assert b.params == g5


def test_s4_novye_tochki_setki_schitayutsya():
    td, ind = _td(9)
    ss = V.generate_s4_v5(td, ind, [{"period": 3, "mnozh": 0.5}, {"period": 80, "mnozh": 6.0}])
    assert ss.sig.shape == (2, N) and (ss.sig[0] != 0).sum() > (ss.sig[1] != 0).sum()


def test_chislo_urovney_osey():
    assert (len(V.OKNO), len(V.PODTV), len(V.PERIOD), len(V.MNOZH)) == (17, 6, 15, 15)


def test_signal_ne_zavisit_ot_bara_t_v_kanale():
    """Огромный high[t] не меняет sig[t] (канал по барам до t-1), но гасит пробой на t+1."""
    td, ind = _td(11)
    g = [{"okno": 20, "podtv": "none"}]
    t = 400
    prob = td.high[t - 20:t].max() * 1.01
    td.close[t + 1] = prob
    td.high[t + 1] = prob * 1.001
    s0 = V.generate_s3_v5(td, ind, g).sig[0]
    td.high[t] = prob * 5
    s1 = V.generate_s3_v5(td, ind, g).sig[0]
    assert s0[t + 1] == 1 and s1[t + 1] == 0
    assert np.array_equal(s1[:t + 1], s0[:t + 1])


def test_mutatsiya_kanal_s_barom_t_lovitsya():
    """Мутант: канал включает бар t (без сдвига) — close > max(high[t-w+1..t]) невозможен;
    эталонный тест обязан отличать его от настоящего сигнала."""
    td, ind = _td(1)
    w = 20
    hc = pd.Series(td.high).rolling(w).max().values
    lc = pd.Series(td.low).rolling(w).min().values
    mut = np.zeros(N, np.int8)
    mut[td.close > hc] = 1; mut[td.close < lc] = -1
    mut[:WARMUP] = 0
    real = V.generate_s3_v5(td, ind, [{"okno": w, "podtv": "none"}]).sig[0]
    assert (real != 0).sum() > 0
    assert not np.array_equal(real, mut)
    assert np.array_equal(real, _ref_gen_sig_s3(td.close, td.high, td.low, w))
