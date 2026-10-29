"""Независимые тесты V6-4 (hurst_th, slope_max/min, vol_regime_max, tihiy_othod). Тестировщик, не автор.
Свои синтетические ряды; эталоны — циклы по формулам, записанным здесь заново."""
import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import calc_atr, calc_sma
from fastbt.signals import v5_filtr as V

NEW = ("hurst_th", "slope_max", "slope_min", "vol_regime_max", "tihiy_othod")


def mk(n, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(r))
    hi = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    vol = rng.integers(1000, 5000, n).astype(float)
    ix = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"high": hi, "low": lo, "close": close, "volume": vol}, index=ix)


# ---------- эталоны (циклы) ----------
def h_loop(close, w=100):
    n = len(close)
    out = np.full(n, np.nan)
    lr = [np.log(max(close[i], 1e-12)) - np.log(max(close[i - 1], 1e-12)) for i in range(1, n)]
    for i in range(w, n):
        r = np.array(lr[i - w:i])
        y = np.cumsum(r - r.mean())
        R, S = y.max() - y.min(), r.std(ddof=1)
        if S > 1e-12 and R > 1e-12:
            out[i] = np.log(R / S) / np.log(w)
    return out


def slope_loop(close, n=50, k=10):
    ma = np.full(len(close), np.nan)
    for i in range(n - 1, len(close)):
        ma[i] = close[i - n + 1:i + 1].mean()
    out = np.full(len(close), np.nan)
    for i in range(n - 1 + k, len(close)):
        out[i] = (ma[i] - ma[i - k]) / ma[i - k] / k * 100.0
    return out


def vr_loop(df):
    a = calc_atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        w = a[max(0, i - 49):i + 1]
        if i >= 49 and not np.isnan(w).any():
            out[i] = a[i] / w.mean()
    return out


# ---------- hurst ----------
def test_hurst_mask_protiv_cikla_i_granica_strogaya():
    df = mk(400, 1)
    h = h_loop(df["close"].to_numpy())
    th = float(np.nanmedian(h))
    m = V.primenit(df, "kontr", {"hurst_th": th})
    exp = ~np.isnan(h) & (h < th)
    assert m.tolist() == exp.tolist()
    k = 250
    m2 = V.primenit(df, "kontr", {"hurst_th": float(h[k])})
    assert not m2[k]                                  # строго <: равенство запрещает
    assert not m[:100].any()                          # прогрев — запрет


def test_hurst_trend_ignoriruet():
    df = mk(300, 2)
    assert V.primenit(df, "trend", {"hurst_th": 0.3}).all()


def test_hurst_znak_trend_vs_mean_reversion():
    n = 300
    ix = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(3)
    trend_r = 0.01 + 0.002 * rng.standard_normal(n)
    mr_r = 0.01 * np.where(np.arange(n) % 2, 1, -1) + 0.0005 * rng.standard_normal(n)
    ht = V.hurst_rs(100 * np.exp(np.cumsum(trend_r)))
    hm = V.hurst_rs(100 * np.exp(np.cumsum(mr_r)))
    assert np.nanmean(ht[100:]) > np.nanmean(hm[100:]) + 0.2


def test_hurst_nan_i_koroten():
    assert np.isnan(V.hurst_rs(np.arange(1, 101.0))).all()       # ровно w точек: нет ни одного окна
    c = mk(200, 4)["close"].to_numpy().copy()
    c[150] = np.nan
    h = V.hurst_rs(c)
    assert np.isnan(h[150:250]).all()                              # NaN в окне — NaN, не число


# ---------- slope ----------
def test_slope_max_kontr_protiv_cikla_trend_ignoriruet():
    df = mk(300, 5)
    sl = slope_loop(df["close"].to_numpy())
    assert np.allclose(V.naklon_pct(df["close"].to_numpy()), sl, equal_nan=True, atol=1e-12)
    s = np.sort(np.abs(sl[~np.isnan(sl)]))
    th = float((s[len(s) // 2] + s[len(s) // 2 + 1]) / 2)     # между данными: без ulp-равенства
    m = V.primenit(df, "kontr", {"slope_max": th})
    assert m.tolist() == (~np.isnan(sl) & (np.abs(sl) < th)).tolist()
    assert V.primenit(df, "trend", {"slope_max": th}).all()


def test_slope_min_trend_napravlenie():
    df = mk(300, 6)
    sl = slope_loop(df["close"].to_numpy())
    th = float(np.nanmedian(np.abs(sl)))
    rng = np.random.default_rng(6)
    sig = rng.integers(-1, 2, len(df)).astype(np.int8)
    m = V.primenit(df, "trend", {"slope_min": th}, sig=sig)
    exp = np.ones(len(df), bool)
    for i in range(len(df)):
        if sig[i] == 1:
            exp[i] = (not np.isnan(sl[i])) and sl[i] >= th
        elif sig[i] == -1:
            exp[i] = (not np.isnan(sl[i])) and sl[i] <= -th
    assert m.tolist() == exp.tolist()
    assert m[sig == 0].all()                           # нулевой сигнал фильтр не трогает
    assert V.primenit(df, "kontr", {"slope_min": th}, sig=sig).all()   # контр — игнор
    with pytest.raises(ValueError):
        V.primenit(df, "trend", {"slope_min": th})


# ---------- vol_regime_max ----------
def test_vol_regime_protiv_cikla_strogo_nan_zapret():
    df = mk(300, 7)
    vr = vr_loop(df)
    th = float(np.nanmedian(vr))
    for tip in ("kontr", "trend"):
        m = V.primenit(df, tip, {"vol_regime_max": th})
        assert m.tolist() == (~np.isnan(vr) & (vr < th)).tolist()
    k = 200
    assert not V.primenit(df, "kontr", {"vol_regime_max": float(vr[k])})[k]
    assert not V.primenit(df, "kontr", {"vol_regime_max": 1e9})[:60].any()   # прогрев ATR+SMA50 — запрет


def test_vol_regime_izvestnyj_otvet_postoyannaya_volatilnost():
    n = 200
    ix = pd.bdate_range("2018-01-01", periods=n)
    c = 100 + np.zeros(n)
    df = pd.DataFrame({"high": c + 1, "low": c - 1, "close": c, "volume": np.full(n, 10.0)}, index=ix)
    # ATR постоянен -> отношение = 1: порог 1.0 запрещает (строго), 1.01 разрешает после прогрева
    assert not V.primenit(df, "kontr", {"vol_regime_max": 1.0}).any()
    m = V.primenit(df, "kontr", {"vol_regime_max": 1.01})
    assert m[100:].all() and not m[:40].any()


# ---------- tihiy_othod ----------
def test_tihiy_othod_protiv_cikla():
    df = mk(120, 8)
    v = df["volume"].to_numpy()
    exp = np.ones(len(v), bool)
    for i in range(len(v)):
        if i >= 4:
            s5 = v[i - 4:i + 1].mean()
            exp[i] = v[i] < s5
    for tip in ("kontr", "trend"):
        assert V.primenit(df, tip, {"tihiy_othod": 1}).tolist() == exp.tolist()
    assert exp[:4].all()                                  # SMA5 нет — разрешено


def test_tihiy_othod_ravenstvo_i_nulevoj_obyom():
    n = 30
    ix = pd.bdate_range("2018-01-01", periods=n)
    base = dict(high=np.full(n, 11.0), low=np.full(n, 9.0), close=np.full(n, 10.0))
    const = pd.DataFrame({**base, "volume": np.full(n, 7.0)}, index=ix)
    m = V.primenit(const, "kontr", {"tihiy_othod": 1})
    assert m[:4].all() and not m[4:].any()                # vol == SMA5 -> запрет (строго <)
    zero = pd.DataFrame({**base, "volume": np.zeros(n)}, index=ix)
    assert V.primenit(zero, "kontr", {"tihiy_othod": 1}).all()   # SMA5 <= 0 -> разрешено


# ---------- «выкл.» = v5 ----------
V5 = {"adx_th": 20, "atr_lo": 30, "vol_th": 0.8, "dow": 3, "rsi_th": 30, "mom_n": 5}


@pytest.mark.parametrize("tip", ["kontr", "trend"])
def test_vykl_pobaytno_v5(tip):
    df = mk(400, 9)
    sig = np.random.default_rng(9).integers(-1, 2, len(df)).astype(np.int8)
    base = V.primenit(df, tip, dict(V5), sig=sig)
    zeros = V.primenit(df, tip, {**V5, **{k: 0 for k in NEW}}, sig=sig)
    assert base.tobytes() == zeros.tobytes()
    assert V.primenit(df, tip, {}).all() and V.primenit(df, tip, {k: 0 for k in NEW}).all()


# ---------- причинность ----------
@pytest.mark.parametrize("tip,par", [
    ("kontr", {"hurst_th": 0.5}), ("kontr", {"slope_max": 0.3}), ("trend", {"slope_min": 0.05}),
    ("kontr", {"vol_regime_max": 1.2}), ("trend", {"vol_regime_max": 1.0}), ("kontr", {"tihiy_othod": 1}),
    ("kontr", {"hurst_th": 0.55, "slope_max": 0.4, "vol_regime_max": 1.5, "tihiy_othod": 1}),
])
def test_prichinnost_isporchennoe_budushchee(tip, par):
    K = 260
    df = mk(400, 10)
    sig = np.random.default_rng(10).integers(-1, 2, len(df)).astype(np.int8)
    a = V.primenit(df, tip, par, sig=sig)
    bad = df.copy()
    rng = np.random.default_rng(11)
    for c in ("high", "low", "close"):
        bad.iloc[K + 1:, bad.columns.get_loc(c)] = rng.uniform(1, 1e4, len(df) - K - 1)
    bad.iloc[K + 1:, bad.columns.get_loc("volume")] = rng.uniform(0, 1e6, len(df) - K - 1)
    b = V.primenit(bad, tip, par, sig=sig)
    assert a[:K + 1].tolist() == b[:K + 1].tolist()


# ---------- границы ----------
@pytest.mark.parametrize("tip", ["kontr", "trend"])
def test_pustaya_i_korotkaya_lenta(tip):
    sig_par = {"hurst_th": 0.5, "slope_max": 0.3, "vol_regime_max": 1.2, "tihiy_othod": 1}
    for n in (0, 1, 5, 60):
        df = mk(n, 12) if n else mk(1, 12).iloc[:0]
        sig = np.zeros(len(df), np.int8)
        out = V.primenit(df, tip, {**sig_par, "slope_min": 0.05}, sig=sig)
        assert len(out) == n
        if tip == "kontr" and n:
            assert not out.any() or n > 100          # нет истории для H/наклона -> запрет


def test_nan_v_close_ne_dayot_razresheniya():
    df = mk(300, 13)
    df.iloc[200, df.columns.get_loc("close")] = np.nan
    m = V.primenit(df, "kontr", {"hurst_th": 0.9})
    assert not m[200:300].any()                      # NaN в окне Хёрста: H = NaN -> запрет
