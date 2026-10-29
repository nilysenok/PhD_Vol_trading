"""V6-4: оси hurst_th, slope_max/slope_min, vol_regime_max, tihiy_othod в fastbt/signals/v5_filtr.py.
Эталон — формулы источника (daily_full_opt_v5.py@6a7a4636, s12.py) циклами; «выкл.» = побайтно v5; префикс (причинность)."""
import numpy as np
import pytest

from fastbt.indicators import calc_atr, calc_sma
from fastbt.signals import v5_filtr as V
from test_v5_filtr import GRID, sig_for, synth


def ref_hurst(close, w=100):
    """calc_hurst источника, цикл."""
    n = len(close); h = np.full(n, np.nan)
    lr = np.diff(np.log(np.maximum(close, 1e-12)))
    for i in range(w, n):
        r = lr[i - w:i]; y = np.cumsum(r - np.mean(r))
        R = np.max(y) - np.min(y); S = np.std(r, ddof=1)
        if S > 1e-12 and R > 1e-12:
            h[i] = np.log(R / S) / np.log(w)
    return h


def ref_slope(ma, n=10):
    """calc_slope источника, цикл."""
    sl = np.full_like(ma, np.nan)
    for i in range(n, len(ma)):
        if not np.isnan(ma[i]) and not np.isnan(ma[i - n]) and abs(ma[i - n]) > 1e-12:
            sl[i] = (ma[i] - ma[i - n]) / ma[i - n] / n * 100
    return sl


def test_hurst_sovpadaet_s_istochnikom():
    c = synth(500)["close"].to_numpy()
    a, b = V.hurst_rs(c), ref_hurst(c)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], atol=1e-12)
    assert np.isnan(a[:100]).all() and not np.isnan(a[100:]).any()


def test_hurst_izvestnyi_otvet_konstanta_i_koroten():
    assert np.isnan(V.hurst_rs(np.full(300, 50.0))).all()        # нулевой разброс — NaN, не 0
    assert np.isnan(V.hurst_rs(np.arange(1, 90.0))).all()        # короче окна


def test_slope_sovpadaet_i_linejnyj_ryad():
    c = synth(300)["close"].to_numpy()
    assert np.allclose(V.naklon_pct(c), ref_slope(calc_sma(c, 50)), equal_nan=True, atol=1e-12)
    lin = 100 + np.arange(200.0)                                   # SMA50 линейной: за 10 баров +10 при MA[d-10]
    s = V.naklon_pct(lin)
    ma = calc_sma(lin, 50)
    assert s[100] == pytest.approx((ma[100] - ma[90]) / ma[90] / 10 * 100) and s[100] > 0


def mask_ref(df, tip, p):
    c, h, l, v = (df[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    m = np.ones(len(c), bool)
    if p.get("hurst_th", 0) > 0 and tip == "kontr":
        hh = ref_hurst(c); m &= ~np.isnan(hh) & (hh < p["hurst_th"])
    if p.get("vol_regime_max", 0) > 0:
        a = calc_atr(h, l, c, 14); s50 = calc_sma(a, 50)
        vr = a / np.where(np.isnan(s50) | (s50 < 1e-12), 1e-12, s50)
        m &= ~np.isnan(vr) & (vr < p["vol_regime_max"])
    if p.get("tihiy_othod", 0):
        v5 = calc_sma(v, 5)
        m &= ~(~np.isnan(v5) & (v5 > 0) & (v >= v5))
    if p.get("slope_max", 0) > 0 and tip == "kontr":
        sl = ref_slope(calc_sma(c, 50)); m &= ~np.isnan(sl) & (np.abs(sl) < p["slope_max"])
    return m


@pytest.mark.parametrize("tip", ["kontr", "trend"])
def test_osi_sovpadayut_s_istochnikom(tip):
    df = synth(600)
    for p in ({"hurst_th": 0.5}, {"vol_regime_max": 1.0}, {"tihiy_othod": 1}, {"slope_max": 0.1},
              {"hurst_th": 0.55, "vol_regime_max": 1.1, "tihiy_othod": 1, "slope_max": 0.2}):
        got = V.primenit(df, tip, p)
        assert np.array_equal(got, mask_ref(df, tip, p)), (tip, p)
        assert 0 < got.sum() < len(got) or tip == "trend", (tip, p)    # ось что-то режет, но не всё


def test_slope_min_trend_napravlenie():
    df = synth(600); sig = sig_for(600)
    c = df["close"].to_numpy(float); sl = ref_slope(calc_sma(c, 50))
    got = V.primenit(df, "trend", {"slope_min": 0.05}, sig)
    exp = ~(((sig == 1) & ~(sl >= 0.05)) | ((sig == -1) & ~(sl <= -0.05)))
    assert np.array_equal(got, exp) and not got.all()
    assert V.primenit(df, "kontr", {"slope_min": 0.05}).all()      # контр: slope_min не действует
    with pytest.raises(ValueError):
        V.primenit(df, "trend", {"slope_min": 0.05})


def test_vyklyucheno_pobaitno_ravno_v5():
    df = synth(); sig = sig_for(len(df))
    off = {"hurst_th": 0, "slope_max": 0, "slope_min": 0, "vol_regime_max": 0, "tihiy_othod": 0}
    for tip in ("kontr", "trend"):
        for p in ({}, {"adx_th": 20, "vol_th": 1.0, "rsi_th": 30, "mom_n": 3, "dow": 2}):
            assert V.primenit(df, tip, {**p, **off}, sig).tobytes() == V.primenit(df, tip, p, sig).tobytes()


def test_prefiks_novykh_osei():
    df = synth(); sig = sig_for(len(df))
    p = {"hurst_th": 0.55, "slope_max": 0.2, "slope_min": 0.02, "vol_regime_max": 1.2, "tihiy_othod": 1}
    for tip in ("kontr", "trend"):
        full = V.primenit(df, tip, p, sig)
        for d in (110, 160, 300, 450, 699):
            assert V.primenit(df.iloc[:d + 1], tip, p, sig[:d + 1])[d] == full[d], (tip, d)


def test_mutatsiya_hurst_so_sdvigom_vpered_lovitsya():
    """Мутация: H[d] по доходностям до d+1 (утечка). Префикс-тест обязан её видеть."""
    c = synth()["close"].to_numpy()
    ok, bad = V.hurst_rs(c), np.r_[V.hurst_rs(c)[1:], np.nan]
    d = 300
    pref = V.hurst_rs(c[:d + 1])[d]
    assert pref == ok[d] and pref != bad[d]


def test_mutatsiya_kontr_filtr_na_trende_lovitsya():
    """Мутация: hurst/slope_max применены к trend. Эталон для trend их не режет — проверяем, что режут разные маски."""
    df = synth(600)
    assert V.primenit(df, "trend", {"hurst_th": 0.5, "slope_max": 0.05}).all()
    assert not V.primenit(df, "kontr", {"hurst_th": 0.5, "slope_max": 0.05}).all()


def test_nol_obyoma_tihiy_othod_ne_rezhet_pri_nulevom_sma():
    df = synth(100); df["volume"] = 0.0
    assert V.primenit(df, "kontr", {"tihiy_othod": 1}).all()
