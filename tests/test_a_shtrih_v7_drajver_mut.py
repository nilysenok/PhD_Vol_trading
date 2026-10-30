"""Тестировщик PR #876 (деньги, Ж-149 п. 1): известные ответы руками и мутации комиссии книги A′ (код не меняется).
Формула автора: тикер платит max(0, ставка·|w_i − w_{i−1}| − ставка·|ΔP_i|·F_i·m_i·f_i); хедж — ставка·|Δh|; w = P·F·m·f."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import a_shtrih_v7_kniga as K
from tests.test_a_shtrih_v7_drajver import D, _vhod
from tests.test_a_shtrih_v7_kniga import _run

KOM = 0.0004


def _d(poz, sektora, T, seed=2):
    rng = np.random.default_rng(seed)
    poz = np.asarray(poz, float)
    return dict(X=rng.normal(0, 0.01, poz.shape), poz=poz, sektora=np.asarray(sektora),
                imoex=rng.normal(0, 0.005, T), mix=rng.normal(0, 0.005, T), ks=np.zeros(T))


def _signaly(znak):
    """6 тикеров в разных секторах, множитель константен (m=f=1, сектор не режет: |P| = 1/6); позиции входят/выходят/идут вспять."""
    T = 12
    poz = np.zeros((1, 6, T))
    poz[0, :, 3:8] = znak * 1.0
    poz[0, :, 8:] = -znak * 1.0
    return _d(poz, np.arange(6), T)


@pytest.mark.parametrize("znak", [1.0, -1.0])
def test_postoyannyj_mnozhitel_vhod_centra_ne_dvojnoj_uchet(znak):
    r = _run(_signaly(znak), sektor_netto=True)
    assert (r.m == 1).all() and (r.f == 1).all() and (np.abs(r.poz_itog).sum(0) > 0).any()
    assert not r.kom_tik.any()                                       # вход, выход и разворот оплачены центрами: книга 0


def test_mutatsiya_dvojnoe_oblozhenie_vhoda_centra_lovitsya(monkeypatch):
    orig = K.kom_tikerov
    monkeypatch.setattr(K, "kom_tikerov", lambda W, dP, Fc, st=KOM: orig(W, np.zeros_like(dP), Fc, st))
    assert _run(_signaly(1.0), sektor_netto=True).kom_tik.sum() > 0   # мутант берёт комиссию второй раз -> тест выше упал бы


def test_sektor_urezal_poziciyu_komissiya_za_urezanie():
    """Два тикера в одном секторе. P = poz/2. Дни 0–4: poz=[1,1] -> P=[.5,.5], нетто 1 -> F=.2, w=[.1,.1].
    Дни 5+: poz=[1,0] -> P=[.5,0], F=[.4,1], w=[.2,0]. Центры: ΔP=[0,−.5], Fc=[.4,1]: оплачено [0, .5]·ставка.
    Книга: тикер 0 |Δw|=.1 -> ставка·.1; тикер 1 |Δw|=.1 < оплачено .5 -> 0. Итого в день 5: ставка·.1; остальные дни 0."""
    T = 10
    poz = np.ones((1, 2, T))
    poz[0, 1, 5:] = 0.0
    r = _run(_d(poz, [0, 0], T), sektor_netto=True)
    ozid = np.zeros(T)
    ozid[5] = KOM * 0.1
    assert r.kom_tik == pytest.approx(ozid, abs=1e-15)


def test_den_rebalansirovki_bez_signalov_komissiya_bolshe_nulya():
    T = 140
    rng = np.random.default_rng(4)
    sd = np.where(np.arange(T) < 70, 0.004, 0.02)
    d = dict(X=rng.normal(0, 1, (1, 1, T)) * sd, poz=np.ones((1, 1, T)), sektora=np.array([0]),
             imoex=rng.normal(0, 0.005, T), mix=np.zeros(T), ks=np.zeros(T))
    r = _run(d, tsel_vol=True, vykl=True)
    assert r.kom_tik[1:][np.abs(np.diff(r.m * r.f)) > 1e-9].min() > 0


def test_hedzh_vkl_vykl_komissiya_ot_dh_a_bez_hedzha_nol():
    T = 200
    rng = np.random.default_rng(7)
    im = rng.normal(0, 0.005, T)
    d = dict(X=(1.5 * im + rng.normal(0, 0.001, T))[None, None, :], poz=np.ones((1, 1, T)), sektora=np.array([0]),
             imoex=im, mix=rng.normal(0, 0.005, T), ks=np.zeros(T))
    d["mix"][120:] = np.nan                                          # хедж гаснет на 120-й день
    r = _run(d, hedzh=True)
    h_prev = np.concatenate([[0.0], r.h[:-1]])
    assert np.array_equal(r.kom_hedzh, KOM * np.abs(r.h - h_prev))
    assert r.kom_hedzh[120] == pytest.approx(KOM * abs(r.h[119])) and r.kom_hedzh[120] > 0   # выключение платит
    off = _run(d, hedzh=False)
    assert not off.h.any() and not off.kom_hedzh.any()


# ---- мутации формулы
def test_mutatsiya_modul_w_vmesto_modulya_izmeneniya_lovitsya(monkeypatch):
    T = 200
    rng = np.random.default_rng(5)
    im = rng.normal(0, 0.001, T)
    d = dict(X=(2.0 * im)[None, None, :], poz=np.ones((1, 1, T)), sektora=np.array([0]), imoex=im,
             mix=rng.normal(0, 0.01, T), ks=np.zeros(T))
    ok = _run(d, hedzh=True)
    assert ok.kom_hedzh[100] < 1e-9 and abs(ok.h[100]) > 0.5          # β стоит: ответ руками — 0
    monkeypatch.setattr(K, "komissiya_oborota", lambda w, wp, st=KOM: float(st * np.abs(np.asarray(w, float)).sum()))
    assert _run(d, hedzh=True).kom_hedzh[100] > 1e-4                   # мутант |w| берёт ~ставка·|h| каждый день


def test_mutatsiya_znak_u_v_komissii_lovitsya(monkeypatch):
    T = 140
    rng = np.random.default_rng(4)
    sd = np.where(np.arange(T) < 70, 0.004, 0.02)
    d = dict(X=rng.normal(0, 1, (1, 1, T)) * sd, poz=np.ones((1, 1, T)), sektora=np.array([0]),
             imoex=rng.normal(0, 0.005, T), mix=np.zeros(T), ks=np.zeros(T))
    ok = _run(d, tsel_vol=True, vykl=True)
    assert ok.kom_tik[1:] == pytest.approx(KOM * np.abs(np.diff(ok.m * ok.f)), abs=1e-15)
    monkeypatch.setattr(K, "komissiya_oborota",
                        lambda w, wp, st=KOM: float(st * np.abs(np.asarray(w, float) + np.asarray(wp, float)).sum()))
    r = _run(d, tsel_vol=True, vykl=True)
    assert not np.allclose(r.kom_tik[1:], ok.kom_tik[1:], atol=1e-12)   # мутант (w + w_prev) расходится с ответом руками


# ---- драйвер
def test_drajver_bez_hedzha_hedzh_nol_i_komissiya_nol():
    d = _vhod(T=300, seed=5)
    rez = D.schitat(d)
    bez = rez["kontrol_bez_hedzha"]
    assert not bez.h.any() and not bez.kom_hedzh.any()
    tab = D.tablica(d, rez)
    assert tab.loc["kontrol_bez_hedzha", "kom_hedzh_god"] == 0 and tab.loc["osnovnaya", "kom_hedzh_god"] > 0


def test_ablaciya_kazhdaya_stupen_odin_sloj():
    r = D.schitat(_vhod(T=300, seed=6))
    a1, a2, a3, a4 = (r[f"ablaciya_{i}"] for i in (1, 2, 3, 4))
    assert not a1.h.any() and not a1.kom_tik.any() and not a1.marzha.any()
    assert not a2.h.any() and not a2.kom_tik.any() and not a2.marzha.any()       # 2: только веса (poz_itog зависит от весов)
    assert a3.h.any() and (a3.m == 1).all() and (a3.f == 1).all()                  # 3: хедж/сектор, таргета нет
    assert np.array_equal(a3.a, a2.a) and np.array_equal(a4.a, a2.a)
    assert np.array_equal(a4.poz_itog, a3.poz_itog)                                 # 4 тот же сектор, что 3
    assert not (a4.m == 1).all() and not (a4.f == 1).all()                         # 4: таргет и выключатель
    assert np.array_equal(r["osnovnaya"].ryad, a4.ryad, equal_nan=True)
