"""Л1 сетки v5: фильтры Р по типу (fastbt/signals/v5_filtr.py) и реестр семей с setka."""
import itertools

import numpy as np
import pandas as pd
import pytest

from fastbt.filtry_v5 import rsi_wilder
from fastbt.indicators import calc_adx, calc_atr, calc_sma
from fastbt.signals import interface as I
from fastbt.signals import v5_filtr as V


def ref_generic(sig, close, volume, ind, rsi, atr_arr, dow, params, is_trend):
    """apply_filters_generic@6a7a4636 (daily_upper_bound.py:302-342), формулы без изменений (кроме atr_lo)."""
    n = len(sig); mask = np.ones(n, dtype=bool)
    adx = ind["adx14"]
    at = params.get("adx_th", 0)
    if at > 0:
        if is_trend: mask &= (~np.isnan(adx)) & (adx > at)
        else: mask &= (~np.isnan(adx)) & (adx < at)
    al = params.get("atr_lo", 0)
    if al > 0:
        va = atr_arr[~np.isnan(atr_arr)]
        if len(va) > 50: mask &= (~np.isnan(atr_arr)) & (atr_arr > np.percentile(va, al))
    vt = params.get("vol_th", 0)
    if vt > 0:
        vs = ind["vol_sma20"]; vs_s = np.where(np.isnan(vs) | (vs < 1), 1.0, vs)
        mask &= volume / vs_s > vt
    rt = params.get("rsi_th", 0)
    if rt > 0:
        if not is_trend:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi < rt))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi > (100 - rt)))] = 0
        else:
            sig[(sig == 1) & ~((~np.isnan(rsi)) & (rsi > 50))] = 0
            sig[(sig == -1) & ~((~np.isnan(rsi)) & (rsi < 50))] = 0
    mn = params.get("mom_n", 0)
    if mn > 0:
        mom = np.zeros(n); mom[mn:] = close[mn:] - close[:-mn]
        if is_trend: sig[(sig == 1) & (mom <= 0)] = 0; sig[(sig == -1) & (mom >= 0)] = 0
        else: sig[(sig == 1) & (mom >= 0)] = 0; sig[(sig == -1) & (mom <= 0)] = 0
    dw = params.get("dow", 0)
    if dw > 0 and dow is not None:
        if dw == 1: mask &= dow != 0
        elif dw == 2: mask &= dow != 4
        elif dw == 3: mask &= (dow != 0) & (dow != 4)
    sig[~mask] = 0
    return sig


def synth(n=700, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    vol = rng.uniform(5e3, 5e4, n)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": vol}, index=idx)


def sig_for(n, seed=5):
    return np.random.default_rng(seed).choice([-1, 0, 1], n, p=[.2, .6, .2]).astype(np.int8)


def ref_sig(df, tip, params, sig):
    c, h, l, v = (df[k].to_numpy(float) for k in ("close", "high", "low", "volume"))
    ind = {"adx14": calc_adx(h, l, c, 14), "vol_sma20": calc_sma(v, 20)}
    return ref_generic(sig.copy(), c, v, ind, rsi_wilder(c, 14), calc_atr(h, l, c, 14),
                       np.asarray(df.index.dayofweek), params, tip == "trend")


GRID = {"adx_th": [0, 20, 30], "vol_th": [0, 1.0, 1.5], "rsi_th": [0, 30], "mom_n": [0, 3], "dow": [0, 1, 2, 3]}


@pytest.mark.parametrize("tip", ["kontr", "trend"])
def test_sovpadaet_s_apply_filters_generic(tip):
    df = synth(); sig = sig_for(len(df))
    keys = list(GRID)
    n_neprazdn = 0
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(keys, vals))
        got = np.where(V.primenit(df, tip, p, sig), sig, 0)
        want = ref_sig(df, tip, p, sig)
        assert np.array_equal(got, want), p
        n_neprazdn += int((want != sig).any())
    assert n_neprazdn > 20   # фильтры реально что-то отсекают


def test_mutatsiya_ref_otlichaetsya_ot_znaka_adx():
    df = synth(); sig = sig_for(len(df))
    assert not np.array_equal(ref_sig(df, "trend", {"adx_th": 25}, sig), ref_sig(df, "kontr", {"adx_th": 25}, sig))
    assert not np.array_equal(V.primenit(df, "trend", {"adx_th": 25}), V.primenit(df, "kontr", {"adx_th": 25}))


def test_rsi_trend_ne_zavisit_ot_poroga():
    df = synth(); sig = sig_for(len(df))
    a = V.primenit(df, "trend", {"rsi_th": 20}, sig)
    assert np.array_equal(a, V.primenit(df, "trend", {"rsi_th": 40}, sig))
    assert not a.all()


def test_vyklyucheno_vse_razresheno():
    df = synth()
    assert V.primenit(df, "kontr", {}).all()
    assert V.primenit(df, "trend", {"adx_th": 0, "atr_lo": 0, "vol_th": 0, "dow": 0, "squeeze_pct": 0}).all()


def test_napravlenie_bez_sig_oshibka():
    with pytest.raises(ValueError):
        V.primenit(synth(), "kontr", {"mom_n": 3})


def ref_atr_expand(atr, pct):
    out = np.zeros(len(atr), bool)
    for d in range(len(atr)):
        va = atr[:d][~np.isnan(atr[:d])]
        if len(va) > 50 and not np.isnan(atr[d]):
            out[d] = atr[d] > np.percentile(va, pct)
    return out


@pytest.mark.parametrize("pct", [5, 10, 25])
def test_atr_protsentil_ekvivalent_rasshiryayushchemu_oknu(pct):
    df = synth()
    atr = calc_atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    assert np.array_equal(V.maska_atr_expand(atr, pct), ref_atr_expand(atr, pct))
    assert np.array_equal(V.primenit(df, "kontr", {"atr_lo": pct}), ref_atr_expand(atr, pct))


def test_prefiks_vsekh_filtrov():
    df = synth(); sig = sig_for(len(df))
    p = {"adx_th": 25, "atr_lo": 10, "vol_th": 0.7, "rsi_th": 30, "mom_n": 3, "dow": 3, "squeeze_pct": 40}
    for tip in ("kontr", "trend"):
        full = V.primenit(df, tip, p, sig)
        for d in (60, 120, 260, 300, 450, 699):
            assert V.primenit(df.iloc[:d + 1], tip, p, sig[:d + 1])[d] == full[d], (tip, d)


def test_mutatsiya_protsentil_po_vsemu_ryadu_lovitsya():
    df = synth()
    atr = calc_atr(df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy(), 14)
    va = atr[~np.isnan(atr)]
    bad = ~np.isnan(atr) & (atr > np.percentile(va, 20))      # как в источнике: по всему ряду
    assert not np.array_equal(bad, V.maska_atr_expand(atr, 20))
    # и тест префикса его бы поймал: маска на префиксе != маске полного ряда
    d = 300
    bad_pref = ~np.isnan(atr[:d + 1]) & (atr[:d + 1] > np.percentile(atr[:d + 1][~np.isnan(atr[:d + 1])], 20))
    assert bad_pref[d] != bad[d] or (bad_pref != bad[:d + 1]).any()


def test_szhatie_prefiks_i_rabotaet():
    df = synth(900)
    m = V.primenit(df, "kontr", {"squeeze_pct": 40})
    assert m.any() and not m.all() and not m[:50].any()


def test_den_nedeli_iz_kolonki_date():
    df = synth().reset_index().rename(columns={"index": "date"})
    p = {"dow": 3}
    assert np.array_equal(V.primenit(df, "kontr", p), V.primenit(synth(), "kontr", p))


def test_reestr_setka():
    f = lambda: None
    I.register_family("S_test_l1", f, setka="v5")
    assert I.get_family("S_test_l1", setka="v5") is f
    with pytest.raises(KeyError):
        I.get_family("S_test_l1_net", setka="v5")
    assert "S_test_l1" not in I.families()          # v4 не затронут
    with pytest.raises(KeyError):
        I.register_family("S1", f, setka="v9")
    assert I.get_family("S1").sid == "S1"             # вызов без setka — как раньше
