"""Параболический трейлинг v4 (fastbt/vyhod_parabolic.py): известный ответ, эталон v4, мутации, numba = эталон."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fastbt.vyhod_parabolic import NIVELI_V4, parabolic_exit, parabolic_exit_ref

INF = 1e10


def _call(fn, d, e, ep, ea, isl, ps, pm, bth, o, h, lo, c, sig):
    return fn(d, e, len(c), ep, ea, isl, ps, pm, bth, o, h, lo, c, sig)


def v4(d, e, ep, ea, isl, ps, pm, bth, o, h, lo, c, sig,
       same_bar=False, signal_first=False, gap_at_level=False):
    """Формула v4 (bt_trend_v3, tt=2), развёрнутая по направлениям; флаги — подложенные ошибки.

    same_bar=True — буквальный v4: экстремум и close бара t входят в стоп бара t.
    """
    long = d == 1
    sl = ep - isl * ea if long else ep + isl * ea
    ext = h[e] if long else lo[e]
    step, be = ps, False
    for t in range(e + 1, len(c)):
        def upd():
            nonlocal ext, step, be
            if long and h[t] > ext:
                ext, step = h[t], min(step + ps, pm)
            if not long and lo[t] < ext:
                ext, step = lo[t], min(step + ps, pm)
            if not be and bth < INF and d * (c[t] - ep) > bth * ea:
                be = True
        if same_bar:
            upd()
        tsl = sl + step * (ext - sl) if long else sl - step * (sl - ext)
        if be:
            tsl = max(tsl, ep) if long else min(tsl, ep)
        sl = max(sl, tsl) if long else min(sl, tsl)
        hit = lo[t] <= sl if long else h[t] >= sl
        if signal_first and sig[t]:
            return t, c[t], False
        if hit:
            if gap_at_level:
                return t, sl, True
            return t, (min(sl, o[t]) if long else max(sl, o[t])), True
        if sig[t]:
            return t, c[t], False
        if not same_bar:
            upd()
    return -1, math.nan, False


def _eq(a, b):
    return a[0] == b[0] and a[2] == b[2] and (a[1] == pytest.approx(b[1], abs=1e-12, nan_ok=True))


def _walk(rng, n=60):
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.01, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.01, n))
    sig = rng.random(n) < 0.04
    return o, h, lo, c, sig


# --- известный ответ (счёт руками) ------------------------------------------------------------

def _hand():
    o = np.array([100, 100, 102, 100.0])
    h = np.array([101, 103, 103, 102.0])
    lo = np.array([100, 101, 101, 98.8])
    c = np.array([100, 102, 102, 99.0])
    return o, h, lo, c, np.zeros(4, bool)


@pytest.mark.parametrize("fn", [parabolic_exit, parabolic_exit_ref])
def test_izvestnyj_otvet_long(fn):
    # sl0=96; б1: 96+0,1*(101-96)=96,5, новый максимум 103 -> шаг 0,2; б2: 96,5+0,2*6,5=97,8;
    # б3: 97,8+0,2*5,2=98,84 >= low 98,8 -> стоп по уровню 98,84
    o, h, lo, c, s = _hand()
    u, fill, stop = _call(fn, 1, 0, 100.0, 2.0, 2.0, 0.1, 0.3, INF, o, h, lo, c, s)
    assert (u, stop) == (3, True) and fill == pytest.approx(98.84, abs=1e-12)


@pytest.mark.parametrize("fn", [parabolic_exit, parabolic_exit_ref])
def test_izvestnyj_otvet_short_zerkalo(fn):
    o, h, lo, c, s = _hand()
    u, fill, stop = _call(fn, -1, 0, 100.0, 2.0, 2.0, 0.1, 0.3, INF, 200 - o, 200 - lo, 200 - h, 200 - c, s)
    assert (u, stop) == (3, True) and fill == pytest.approx(200 - 98.84, abs=1e-12)


def test_shag_ne_vyshe_maksimuma():
    # pm = ps: шаг не растёт; б1: 96,5; б2: 96,5+0,1*6,5=97,15; б3: 97,15+0,1*5,85=97,735
    o, h, lo, c, s = _hand()
    assert _call(parabolic_exit, 1, 0, 100.0, 2.0, 2.0, 0.1, 0.1, INF, o, h, lo, c, s)[0] == -1
    lo = lo.copy()
    lo[3] = 97.7
    u, fill, stop = _call(parabolic_exit, 1, 0, 100.0, 2.0, 2.0, 0.1, 0.1, INF, o, h, lo, c, s)
    assert (u, stop) == (3, True) and fill == pytest.approx(97.735, abs=1e-12)


def test_ne_zakryta_i_pustaya_lenta():
    o, h, lo, c, s = _hand()
    assert parabolic_exit(1, 0, 1, 100.0, 2.0, 2.0, 0.1, 0.3, INF, o, h, lo, c, s)[0] == -1
    z = np.zeros(0)
    assert parabolic_exit(1, 0, 0, 100.0, 2.0, 2.0, 0.1, 0.3, INF, z, z, z, z, np.zeros(0, bool))[0] == -1


# --- эталон v4 и numba = эталон ---------------------------------------------------------------

def test_raven_formule_v4_i_numba_raven_etalonu():
    rng = np.random.default_rng(7)
    n_stop = 0
    for k in range(400):
        o, h, lo, c, sig = _walk(rng)
        ps, pm = NIVELI_V4[k % 4]
        d = 1 if k % 2 == 0 else -1
        bth = INF if k % 3 else 1.0
        ep, ea, isl = c[0], 0.02 * c[0], 2.0
        a = parabolic_exit(d, 0, len(c), ep, ea, isl, ps, pm, bth, o, h, lo, c, sig)
        r = parabolic_exit_ref(d, 0, len(c), ep, ea, isl, ps, pm, bth, o, h, lo, c, sig)
        f = v4(d, 0, ep, ea, isl, ps, pm, bth, o, h, lo, c, sig)
        assert _eq(a, r) and _eq(a, f), (k, a, r, f)
        n_stop += a[2]
    assert n_stop > 100  # перебор не вырожден: стопы срабатывают


# --- мутации ----------------------------------------------------------------------------------

def _count_diff(**flags):
    rng = np.random.default_rng(11)
    diff = 0
    for k in range(400):
        o, h, lo, c, sig = _walk(rng)
        ps, pm = NIVELI_V4[k % 4]
        d = 1 if k % 2 == 0 else -1
        ep, ea = c[0], 0.02 * c[0]
        a = parabolic_exit(d, 0, len(c), ep, ea, 2.0, ps, pm, 1.0, o, h, lo, c, sig)
        m = v4(d, 0, ep, ea, 2.0, ps, pm, 1.0, o, h, lo, c, sig, **flags)
        diff += not _eq(a, m)
    return diff


def test_mutaciya_sar_dnya_t_po_high_low_dnya_t_lovitsya():
    assert _count_diff(same_bar=True) > 0


def test_mutaciya_signal_ranshe_stopa_lovitsya():
    assert _count_diff(signal_first=True) > 0


def test_mutaciya_gep_po_urovnyu_lovitsya():
    assert _count_diff(gap_at_level=True) > 0


def test_stop_i_signal_v_odnom_bare_stop_pobezhdaet():
    o, h, lo, c, s = _hand()
    s = s.copy()
    s[3] = True
    u, fill, stop = _call(parabolic_exit, 1, 0, 100.0, 2.0, 2.0, 0.1, 0.3, INF, o, h, lo, c, s)
    assert stop and fill == pytest.approx(98.84, abs=1e-12)


def test_gep_cherez_stop_ispolnyaetsya_po_otkrytiyu():
    o, h, lo, c, s = _hand()
    o = o.copy(); o[3] = 98.0; lo = lo.copy(); lo[3] = 97.5
    u, fill, stop = _call(parabolic_exit, 1, 0, 100.0, 2.0, 2.0, 0.1, 0.3, INF, o, h, lo, c, s)
    assert (u, stop, fill) == (3, True, 98.0)
