"""Л4 (Ж-127 п. 2): S5@v5 / S6@v5 — fastbt/signals/v5_s56.py. Синтетика, без сети и данных.

Эталоны: S5 v4 (`fastbt/signals/s56.py`) побайтно; формулы Р @6a7a4636 — в test_v5_s56_formuly.py.
Мутация «пивот по текущему периоду» (mutations.md № 1, сдвиг входа) — в конце файла.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest

from fastbt.indicators import compute_base
from fastbt.signals import s56 as v4
from fastbt.signals import v5_s56 as v5
from fastbt.signals.interface import TickerData, expand_grid

TIPY = ["classic", "fibonacci", "woodie", "camarilla", "demark"]


def make_td(n=700, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    o = c * np.exp(rng.normal(0, 0.006, n))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.012, n))
    l = np.minimum(o, c) * (1 - rng.uniform(0, 0.012, n))
    v = rng.uniform(1e5, 5e5, n)
    d = pd.bdate_range("2018-01-01", periods=n).values.astype("datetime64[ns]")
    lr = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    return TickerData("SYN", d, o, h, l, c, v, lr)


def ind_of(td):
    return compute_base(td.close, td.high, td.low, td.open, td.volume)


@pytest.fixture(scope="module")
def td():
    return make_td()


@pytest.fixture(scope="module")
def ind(td):
    return ind_of(td)


def eq(a, b):
    return np.array_equal(a, b, equal_nan=True)


# ───────────── S5: v4 побайтно ─────────────
def test_s5_classic_fib_nedelya_mesyac_fs234_ravny_v4(td, ind):
    g5 = expand_grid(dict(pivot_type=["classic", "fibonacci"], pivot_period=["nedelya", "mesyac"],
                          uroven=[1], atr_filtr=[0], filter_set=[2, 3, 4]))
    g4 = [dict(pivot_type=p["pivot_type"], filter_set=p["filter_set"],
               pivot_period={"nedelya": "weekly", "mesyac": "monthly"}[p["pivot_period"]]) for p in g5]
    a, b = v5.generate5(td, ind, g5), v4.generate5(td, ind, g4)
    assert (a.sig != 0).sum() > 20
    assert eq(a.sig, b.sig) and eq(a.exit_l, b.exit_l) and eq(a.exit_s, b.exit_s)
    assert eq(a.strength, b.strength)
    for k in ("P", "S1", "S2", "R1", "R2"):
        assert eq(a.extra[k], b.extra[k]), k


def test_atr_filtr_ne_rasshiryaet_signaly(td, ind):
    g = expand_grid(dict(pivot_type=["woodie"], pivot_period=["den"], uroven=[1], atr_filtr=[0, 1.0, 2.0],
                         filter_set=["net"]))
    s = v5.generate5(td, ind, g).sig
    assert ((s[1] != 0) <= (s[0] != 0)).all() and ((s[2] != 0) <= (s[1] != 0)).all()
    assert (s[2] != 0).sum() < (s[0] != 0).sum()


def test_fs_net_nadmnozhestvo_fs2(td, ind):
    g = expand_grid(dict(pivot_type=["classic"], pivot_period=["den"], uroven=[1], atr_filtr=[0],
                         filter_set=["net", 2]))
    s = v5.generate5(td, ind, g).sig
    assert ((s[1] != 0) <= (s[0] != 0)).all() and (s[0] != 0).sum() > (s[1] != 0).sum()


def test_ob_em_setki():
    assert len(v5.params5()) == 5 * 3 * 3 * 5 * 4 and len(v5.params6()) == 15 * 12 * 2 * 5


def test_nedostatochno_dannyh_ne_padaet():
    t = make_td(n=12)
    ss = v5.generate5(t, ind_of(t), v5.params5()[:3])
    assert not ss.sig.any() and ss.sig.shape == (3, 12)


# ───────────── S6 ─────────────
def test_s6_fs1_4_dir_bias_ravny_v4(td, ind):
    g = expand_grid(dict(vwap_window=[5, 20, 60], dev_mult=[0.5, 1.5], dir_bias=["OFF", "ON"],
                         filter_set=[1, 2, 3, 4]))
    a, b = v5.generate6(td, ind, g), v4.generate6(td, ind, g)
    assert (a.sig != 0).sum() > 20
    for k in ("sig", "exit_l", "exit_s", "strength"):
        assert eq(getattr(a, k), getattr(b, k)), k
    assert eq(a.extra["vwap"], b.extra["vwap"]) and eq(a.extra["dev"], b.extra["dev"])


def test_s6_net_bez_nabora_fs(td, ind):
    g = expand_grid(dict(vwap_window=[3, 100], dev_mult=[0.25, 1.0], dir_bias=["OFF"], filter_set=["net", 1]))
    ss = v5.generate6(td, ind, g)
    for i, p in enumerate(g):
        vw, dv = ss.extra["vwap"][i], ss.extra["dev"][i]
        lc = ~np.isnan(vw) & (td.close < vw - p["dev_mult"] * dv)
        sc = ~np.isnan(vw) & (td.close > vw + p["dev_mult"] * dv)
        if p["filter_set"] == "net":
            want = np.where(lc, 1, np.where(sc, -1, 0)).astype(np.int8); want[:200] = 0
            assert eq(ss.sig[i], want) and (want != 0).sum() > 50
        else:
            j = g.index(dict(p, filter_set="net"))      # порядок expand_grid: фильтр не соседний
            net = ss.sig[j]
            assert ((ss.sig[i] != 0) <= (net != 0)).all() and (net != 0).sum() > (ss.sig[i] != 0).sum()


def test_s6_dir_bias_on_rezhet_napravlenie(td, ind):
    g = expand_grid(dict(vwap_window=[10], dev_mult=[0.5], dir_bias=["OFF", "ON"], filter_set=["net"]))
    s = v5.generate6(td, ind, g).sig
    assert ((s[1] != 0) <= (s[0] != 0)).all() and (s[1] != 0).sum() < (s[0] != 0).sum()


# ───────────── общие фильтры Р через primenit ─────────────
@pytest.fixture
def stub_filtr(monkeypatch):
    calls = []
    m = types.ModuleType("fastbt.signals.v5_filtr")
    m.primenit = lambda df, tip, params, sig=None: (calls.append((len(df), tip, dict(params), sig is not None))
                                                    or np.zeros(len(df), bool))
    monkeypatch.setitem(sys.modules, "fastbt.signals.v5_filtr", m)
    return calls


def test_filtr_cherez_primenit(td, ind, stub_filtr):
    base = dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net")
    ss = v5.generate5(td, ind, [dict(base), dict(base, filtr={"adx": 25})])
    assert (ss.sig[0] != 0).any() and not ss.sig[1].any()
    assert stub_filtr == [(td.n, "kontr", {"adx": 25}, True)]
    s6 = v5.generate6(td, ind, [dict(vwap_window=5, dev_mult=0.5, dir_bias="OFF", filter_set="net",
                                     filtr={"mom": 1})])
    assert not s6.sig.any()


def test_nastoyashchiy_primenit_rsi_mom_i_registraciya(td, ind):
    from fastbt.signals.interface import get_family
    g = [dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
         dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net",
              filtr={"adx_th": 25, "rsi_th": 30, "mom_n": 3})]
    s = v5.generate5(td, ind, g).sig
    assert ((s[1] != 0) <= (s[0] != 0)).all() and (s[1] != 0).sum() < (s[0] != 0).sum()
    assert get_family("S5", setka="v5") is v5.generate5 and get_family("S6", setka="v5") is v5.generate6


# ───────────── мутации ─────────────
def pivots_tekushchiy(gid, o, h, l, c, tip):
    """МУТАНТ: уровни из H/L/C ТЕКУЩЕГО периода (без сдвига на прошлый)."""
    starts = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]); ends = np.r_[starts[1:], len(gid)]
    G = np.maximum.reduceat(h, starts), np.minimum.reduceat(l, starts), c[ends - 1], o[starts]
    return [np.repeat(a, ends - starts) for a in v5._levels(tip, *G)]


def zavisimost_ot_nastoyashchego(fn, td, tip, period, t):
    """Меняем все данные с начала периода бара t и позже; уровни бара t обязаны не измениться."""
    gid = v5._group_ids(td.dates, period)
    start = np.flatnonzero(gid == gid[t])[0]
    base = fn(gid, td.open, td.high, td.low, td.close, tip)
    arrs = [a.copy() for a in (td.open, td.high, td.low, td.close)]
    for a in arrs:
        a[start:] *= 1.37
    alt = fn(gid, *arrs, tip)
    return [b[t] == a[t] or (np.isnan(b[t]) and np.isnan(a[t])) for b, a in zip(base, alt)]


@pytest.mark.parametrize("period", ["den", "nedelya", "mesyac"])
@pytest.mark.parametrize("tip", TIPY)
def test_pivot_ne_vidit_tekushchiy_period(td, tip, period):
    for t in (150, 333, 512, 650):
        assert all(zavisimost_ot_nastoyashchego(v5.pivots, td, tip, period, t)), (tip, period, t)


@pytest.mark.parametrize("period", ["den", "nedelya", "mesyac"])
def test_mutaciya_pivot_po_tekushchemu_periodu_lovitsya(td, period):
    ok = [all(zavisimost_ot_nastoyashchego(pivots_tekushchiy, td, "classic", period, t))
          for t in (150, 333, 512, 650)]
    assert not any(ok)


def test_mutaciya_pivot_po_tekushchemu_periodu_menyaet_signaly(td, ind, monkeypatch):
    g = [dict(pivot_type="classic", pivot_period="nedelya", uroven=1, atr_filtr=0, filter_set="net")]
    good = v5.generate5(td, ind, g).sig.copy()
    monkeypatch.setattr(v5, "pivots", pivots_tekushchiy)
    bad = v5.generate5(td, ind, g).sig
    assert not eq(good, bad)
