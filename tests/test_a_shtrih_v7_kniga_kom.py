"""Л3.3 (Ж-149 п. 1): комиссия 0,04 % на каждое изменение позиции при масштабировании книги A′-расширенной.
Формула: по тикеру доплата = max(0, ставка·|w_i − w_{i−1}| − ставка·|ΔP_i|·F_i·m_i·f_i); сделки центров уже оплачены
в X (на |ΔP| с тем же масштабом), доплачивается только оборот от смены множителей. Хедж — тем же `komissiya_oborota`.
Тест написан ДО кода: на базе падает (нет `kom_tik`, `kom_tikerov`, `KOM_HEDZH` ещё есть)."""
from __future__ import annotations

import numpy as np
import pytest

from fastbt import a_shtrih_v7_kniga as K
from tests.test_a_shtrih_v7_kniga import _mk, _run

KOM = 0.0004   # литерал: независимая сверка


def _bez_signalov(znak=1.0, T=140, seed=4):
    """Позиции константны (сигналов нет), вола X растёт с дня 70: множитель таргета m меняется."""
    rng = np.random.default_rng(seed)
    sd = np.where(np.arange(T) < 70, 0.004, 0.02)
    X = rng.normal(0, 1, (1, 2, T)) * sd
    poz = np.zeros((1, 2, T))
    poz[0, 0], poz[0, 1] = znak, -znak
    return dict(X=X, poz=poz, sektora=np.array([0, 1]), imoex=rng.normal(0, 0.005, T),
                mix=rng.normal(0, 0.005, T), ks=np.zeros(T))


def test_den_rebalansirovki_bez_signalov_spisyvaet_komissiyu():
    d = _bez_signalov()
    r = _run(d, tsel_vol=True, vykl=True)
    c = r.m * r.f
    dc = np.abs(np.diff(c))
    assert (dc > 1e-9).sum() > 20                                        # ребалансировок много, сигналов нет
    ozid = np.concatenate([[0.0], KOM * 1.0 * dc])                        # P = poz/N = ±0,5 -> Σ|P| = 1, знак не важен
    assert (r.kom_tik[1:][dc > 1e-9] > 0).all()
    assert r.kom_tik == pytest.approx(ozid, abs=1e-15)
    ex_port = _run(d).ryad
    assert r.ryad == pytest.approx(ex_port * c - r.marzha - r.kom_tik, abs=1e-15, nan_ok=True)


def test_znak_pozicii_ne_vliyaet_u_shorta_net_vozvrata():
    a = _run(_bez_signalov(1.0), tsel_vol=True, vykl=True).kom_tik
    b = _run(_bez_signalov(-1.0), tsel_vol=True, vykl=True).kom_tik
    assert np.array_equal(a, b) and (a >= 0).all() and a.sum() > 0


def test_kom_tikerov_nizhnyaya_granitsa_nol_po_kazhdomu_tikeru():
    """Тикер A: центр оплатил больше нужного (платить нечего, возврата нет); B: недоплата. Сумма без границы дала бы < 0."""
    W = np.array([[0.5, 0.4], [0.2, 0.5]])
    dP = np.array([[0.5, 0.5], [0.2, 0.0]])
    Fc = np.ones((2, 2))
    kom = K.kom_tikerov(W, dP, Fc)
    assert kom[0] == 0.0                                                  # день 0: вход оплачен центром (w_prev = 0)
    assert kom[1] == pytest.approx(KOM * 0.3, abs=1e-15)                  # только B; без границы было бы KOM·(0,1+0,3−0,5)<0


def test_vse_vykl_net_komissii_tikerov():
    d = _mk(S=3, N=4, T=120, seed=8, nan=False)
    r = _run(d)
    assert not r.kom_tik.any() and r.kom_tik.shape == (120,)


def test_bez_pozicij_net_komissii_tikerov_i_hedzh_rabotaet():
    d = _mk(T=130, seed=9)
    d["poz"] = None
    r = _run(d, tsel_vol=True, hedzh=True)
    assert not r.kom_tik.any() and np.isfinite(r.ryad).any()


def test_hedzh_cherez_komissiya_oborota_i_net_otdelnoj_konstanty(monkeypatch):
    assert not hasattr(K, "KOM_HEDZH")
    d = _bez_signalov()
    base = _run(d, hedzh=True).kom_hedzh
    assert base.sum() > 0 and np.array_equal(base, KOM * np.abs(np.diff(_run(d, hedzh=True).h, prepend=0.0)))
    monkeypatch.setattr(K, "komissiya_oborota", lambda w, wp, stavka=KOM: 2 * KOM * float(np.abs(np.subtract(w, wp)).sum()))
    assert _run(d, hedzh=True).kom_hedzh == pytest.approx(2 * base, abs=1e-15)      # мутация: ставка вытащена из общего определения


def test_tiker_komissiya_idyot_cherez_komissiya_oborota(monkeypatch):
    d = _bez_signalov()
    base = _run(d, tsel_vol=True).kom_tik
    monkeypatch.setattr(K, "komissiya_oborota", lambda w, wp, stavka=KOM: 0.0)
    assert not _run(d, tsel_vol=True).kom_tik.any() and base.sum() > 0


def test_md1_komissiya_tikerov_ne_vidit_budushchego():
    d = _mk(T=150, seed=2)
    j = 100
    r0 = _run(d, tsel_vol=True, vykl=True, sektor_netto=True)
    d2 = {**d, "X": d["X"].copy(), "poz": d["poz"].copy()}
    rng = np.random.default_rng(3)
    d2["X"][:, :, j + 1:] = rng.normal(0, 0.05, d2["X"][:, :, j + 1:].shape)
    d2["poz"][:, :, j + 1:] = rng.choice([-1.0, 1.0], size=d2["poz"][:, :, j + 1:].shape)
    r1 = _run(d2, tsel_vol=True, vykl=True, sektor_netto=True)
    assert np.array_equal(r0.kom_tik[:j + 1], r1.kom_tik[:j + 1]) and not np.array_equal(r0.kom_tik, r1.kom_tik)


def test_sektor_rezhet_poziciyu_bez_signala_platit_komissiyu():
    """Известный ответ. Один сектор, N = 2 (P = poz/2). A держит poz 0,4 (P = 0,2); с дня 10 B входит с poz 0,4:
    нетто 0,4 > 0,2, обе режутся до 0,1 (F = 0,5). A: 0,2 -> 0,1 без сигнала, центр A не торговал -> KOM·0,1.
    B: 0 -> 0,1, центр оплатил KOM·ΔP·F = KOM·0,2·0,5 = KOM·0,1 -> доплаты нет."""
    T = 30
    poz = np.zeros((1, 2, T))
    poz[0, 0] = 0.4
    poz[0, 1, 10:] = 0.4
    d = dict(X=np.full((1, 2, T), 0.001), poz=poz, sektora=np.array([0, 0]), imoex=np.zeros(T), mix=np.zeros(T),
             ks=np.zeros(T))
    r = _run(d, sektor_netto=True)
    ozid = np.zeros(T)
    ozid[10] = KOM * 0.1
    assert r.poz_itog[:, 10] == pytest.approx([0.1, 0.1], abs=1e-15)
    assert r.kom_tik == pytest.approx(ozid, abs=1e-15)
