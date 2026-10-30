"""Правило 9 canon_v6_nesovm (bb_predel): S2 при SMA и norm = okno требует |z| ≥ bb_std, а |z| ≤ (w−1)/√w (σ выборочная, окно с текущим баром).
Сигнал берётся из боевого кода: fastbt.signals.v5_s12.generate_s2 (ядро s12._s2, std = indicators.calc_std ddof=1). Оракул в обе стороны:
не выброшенная пара → сигнал на синтетике есть; выброшенная → сигналов 0 на любом ряде. Данные и сеть не нужны."""
import functools
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy.stats import qmc

from fastbt import canon_v6 as c6, canon_v6_nesovm as nsv
from fastbt.signals.v5_s12 import generate_s2

T = 1500
KUB = c6._kub("S2")
OSI = {n: g for n, g, _ in c6.osi(KUB, None)}
PARY = [(w, k) for w in OSI["bb_window"] for k in OSI["bb_std"]]


def _td(close):
    n = len(close)
    return SimpleNamespace(close=close, open=close, high=close, low=close, volume=np.full(n, 1e6), n=n, dates=np.arange(n))


def signaly(close, w, k, tip="sma"):
    """Число сигнал-дней (вход ±1) S2 с podtv = none, фильтры выкл."""
    td = _td(close)
    ind = {"vol_sma20": np.full(td.n, 1e6), "atr14": np.ones(td.n)}
    ss = generate_s2(td, ind, [dict(ma_tip=tip, bb_window=w, bb_std=k, podtv="none")])
    return int((ss.sig[0] != 0).sum())


def ryad_gauss():
    return 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, T))


def ryad_tolstye_hvosty():
    """Блуждание: 60 % нулевых приращений (плоские участки, как на тиковой сетке), остальное — t(2) с редкими выбросами ×20."""
    rng = np.random.default_rng(2)
    r = rng.standard_t(2, T) * (rng.random(T) > 0.6)
    r[rng.random(T) < 0.01] *= 20
    return 1000 + np.cumsum(r)


def ryad_vybros():
    """Плоский ряд с одиночными выбросами вверх и вниз: по w−1 равных цен и скачок — вплотную к пределу."""
    c = np.full(T, 100.0)
    for i in range(210, T, 37):
        c[i] += (-1) ** (i // 37) * 50.0
    return c + np.random.default_rng(3).normal(0, 1e-3, T)


RYADY = (ryad_gauss, ryad_tolstye_hvosty, ryad_vybros)


@functools.cache
def est_signal(w, k):
    return any(signaly(r(), w, k) > 0 for r in RYADY)


def vybroshena(w, k):
    return nsv.bb_predel("S2", dict(ma_tip="sma", bb_window=w, bb_std=k))


@pytest.mark.parametrize("w,k", PARY)
def test_para_seti_ne_vybroshena_znachit_signal_est(w, k):
    if not vybroshena(w, k):
        assert est_signal(w, k), (w, k)


@pytest.mark.parametrize("w,k", PARY)
def test_vybroshennaya_para_ne_dayot_signalov_na_lyubom_ryade(w, k):
    if vybroshena(w, k):
        for r in RYADY:
            assert signaly(r(), w, k) == 0, (w, k, r.__name__)


def test_tochnyj_predel_dostizhim_i_ne_prevyshen():
    """Предел (w−1)/√w достигается (w−1 равных цен и скачок) и сверху недостижим: k чуть ниже предела даёт сигнал, k на пределе (в запасе 1e-6) — нет."""
    for w in (3, 7, 10, 15, 20):
        c = np.full(T, 100.0)
        c[300::50] += 40.0
        assert signaly(c, w, nsv.predel_z(w) * (1 - 1e-6)) > 0, w
        assert signaly(c, w, nsv.predel_z(w) * (1 + 1e-6)) == 0, w


def test_pary_pod_pravilom_vklyuchayut_7_3_i_7_4():
    assert vybroshena(7, 3.0) and vybroshena(7, 4.0)
    assert not vybroshena(7, 2.25) and not vybroshena(18, 4.0)       # 2,25 < 2,268; 4,0 < 4,12
    assert est_signal(7, 2.25) and est_signal(18, 4.0)


def test_pravilo_ne_trogaet_drugie_ma_norm_i_kuby():
    q = dict(bb_window=7, bb_std=4.0)
    assert not nsv.bb_predel("S2", {**q, "ma_tip": "ema"})            # предел для не-SMA не доказан
    assert not nsv.bb_predel("S2", {**q, "ma_tip": "sma", "norm": "s252"})
    assert not nsv.bb_predel("S1", {**q, "ma_tip": "sma"})
    assert nsv.bb_predel("S2", {**q, "ma_tip": "sma", "norm": "okno"})


def test_canon_v6_vozvrashchaet_none():
    from test_canon_nesovm import baza
    assert c6.canon_v6("S2", baza("S2", ma_tip="sma", bb_window=7, bb_std=4.0), KUB) is None
    assert c6.canon_v6("S2", baza("S2", ma_tip="sma", bb_window=7, bb_std=2.0), KUB) is not None


@functools.cache
def tochki_sobol(n=3000):
    ox, s = c6.osi(KUB), KUB["sobol"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = qmc.Sobol(len(ox), scramble=s["scramble"], seed=s["seed"]).random(4096)[:n]
    out = []
    for row in u:
        pt = {a: g[c6._indeks(x, len(g), sp.get("p_vykl"))] for (a, g, sp), x in zip(ox, row)}
        with mock.patch.object(nsv, "PRAVILA", tuple(p for p in nsv.PRAVILA if p is not nsv.bb_predel)):
            c0 = c6.canon_v6("S2", pt, KUB)
        out.append((c6.canon_v6("S2", pt, KUB), c0))
    return out


def test_orakul_v_obe_storony_na_tochkah_sobolya():
    """Не-None точка S2 с SMA/okno → сигнал возможен; точка, выброшенная только правилом 9 (c0 есть, итога нет), — без правила сигналов 0."""
    chislo = dict(est=0, net=0)
    for real, c0 in tochki_sobol():
        if c0 is None or c0.get("ma_tip", "sma") != "sma" or c0.get("norm", "okno") != "okno":
            continue
        w, k = c0["bb_window"], c0["bb_std"]
        if real is not None:
            assert est_signal(w, k), (w, k)
            chislo["est"] += 1
        else:
            assert all(signaly(r(), w, k) == 0 for r in RYADY), (w, k)
            chislo["net"] += 1
    assert chislo["est"] and chislo["net"], chislo


MUTANTY = {"ddof0": lambda w: (w - 1) ** 0.5, "bez_minus_1": lambda w: w ** 0.5, "slishkom_nizkiy": lambda w: (w - 2) / w ** 0.5}


def proshel_li_orakul():
    """True, если оракул пар не нашёл расхождений правила и сигнала на всей сетке."""
    return all(est_signal(w, k) for w, k in PARY if not vybroshena(w, k)) and \
        all(signaly(r(), w, k) == 0 for w, k in PARY if vybroshena(w, k) for r in RYADY)


def test_orakul_prohodit_na_pravile():
    assert proshel_li_orakul()


@pytest.mark.parametrize("imya", MUTANTY)
def test_mutaciya_predel_ne_toy_storony_lovitsya(imya):
    with mock.patch.object(nsv, "predel_z", MUTANTY[imya]):
        assert not proshel_li_orakul()


def test_mutaciya_pravilo_snyato_lovitsya():
    from test_canon_nesovm import baza
    with mock.patch.object(nsv, "PRAVILA", tuple(p for p in nsv.PRAVILA if p is not nsv.bb_predel)):
        assert c6.canon_v6("S2", baza("S2", ma_tip="sma", bb_window=7, bb_std=4.0), KUB) is not None      # без правила точка проходит
    with mock.patch.object(nsv, "bb_predel", lambda sid, q: False):
        assert not proshel_li_orakul()                                                                     # и оракул это видит
