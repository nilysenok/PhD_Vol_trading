"""Тестировщик PR #857: мутации и границы окон для `fastbt/a_shtrih_v7_kniga.py` (код не меняется)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7_kniga as K
from fastbt import progon_urovni as pu
from fastbt.kniga_msp import mnozhitel_vola
from tests.test_a_shtrih_v7_kniga import _kniga_2beta, _mk, _run

KOM = 0.0004


def _beta_menyaetsya():
    """Книга, у которой β меняется со временем: комиссия обязана идти от |Δh|, не от h."""
    rng = np.random.default_rng(5)
    T = 200
    im = rng.normal(0, 0.001, T)
    k = np.where(np.arange(T) < 120, 2.0, 3.0)
    return dict(X=(k * im)[None, None, :], poz=np.ones((1, 1, T)), sektora=np.array([0]), imoex=im,
                mix=rng.normal(0, 0.01, T), ks=np.zeros(T))


def test_g1_tri_sinteticheskih_vykl_ravno_a_shtrih_pobajtno():
    konf = next(c for c in pu.konfiguracii() if c["rol"] == "ravnyj_bez_targeta")
    for S, N, T, seed, nan in ((2, 3, 90, 11, True), (5, 6, 200, 12, True), (3, 2, 60, 13, False)):
        d = _mk(S=S, N=N, T=T, seed=seed, nan=nan)
        daty = pd.bdate_range("2016-01-04", periods=T)
        ref = pu.a_shtrih({f"S{k}": d["X"][k] for k in range(S)}, konf, daty)
        assert ref.tobytes() == _run(d).ryad.tobytes(), (S, N, T)


def test_mutatsiya_komissiya_hedzha_ot_beta_vmesto_modulya_izmeneniya_lovitsya():
    d = _beta_menyaetsya()
    r = _run(d, hedzh=True)
    assert np.array_equal(r.kom_hedzh, KOM * np.abs(np.diff(r.h, prepend=0.0)))
    assert r.kom_hedzh[100] < 1e-9                                  # β стоит -> комиссии нет
    assert r.kom_hedzh[100] != pytest.approx(KOM * abs(r.h[100]))   # мутация (от |h|) дала бы 0,0008


def test_mutatsiya_tsel_bez_okna_s_dnyom_d_lovitsya(monkeypatch):
    d = _mk(T=150, seed=3)
    base = _run(d, tsel_vol=True).m
    orig = K.mnozhitel_vola
    monkeypatch.setattr(K, "mnozhitel_vola", lambda ex, i, g, tsel=0.10, plecho=2.0: orig(ex, i + 1, g, tsel, plecho))
    assert not np.array_equal(base, _run(d, tsel_vol=True).m)


def test_mutatsiya_plecho_tri_vmesto_dvuh_lovitsya():
    rng = np.random.default_rng(4)
    T = 80
    d = dict(X=rng.normal(0, 1e-5, (1, 1, T)), poz=np.full((1, 1, T), 1.0), sektora=np.array([0]),
             imoex=rng.normal(0, 0.01, T), mix=np.zeros(T), ks=np.zeros(T))
    assert (np.abs(_run(d, tsel_vol=True).m[30:]) <= 2.0 + 1e-12).all()
    assert (np.abs(_run(d, tsel_vol=True, plecho=3.0).m[30:]) > 2.0 + 1e-9).all()    # мутация видна


def test_okno_tseli_rovno_21_strogo_menshe_d():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.01, 100)
    i = 60
    y = x.copy(); y[i - 22] += 0.5
    assert mnozhitel_vola(x, i, 1.0) == mnozhitel_vola(y, i, 1.0)
    y = x.copy(); y[i - 21] += 0.5
    assert mnozhitel_vola(x, i, 1.0) != mnozhitel_vola(y, i, 1.0)
    y = x.copy(); y[i] += 0.5
    assert mnozhitel_vola(x, i, 1.0) == mnozhitel_vola(y, i, 1.0)


def test_beta_okno_63_strogo_menshe_d_cherez_knigu():
    d = _kniga_2beta(T=200)
    i = 150
    h0 = _run(d, hedzh=True).h[i]
    for j, menyaet in ((i - 64, False), (i - 63, True), (i, False)):
        d2 = {**d, "imoex": d["imoex"].copy(), "X": d["X"].copy()}
        d2["imoex"][j] += 0.05
        d2["X"][0, 0, j] += 0.03
        assert (_run(d2, hedzh=True).h[i] != h0) == menyaet, j


def test_granitsa_mix_nan_vkljuchenie_i_otklyuchenie_platyat_komissiyu():
    d = _kniga_2beta(T=200)
    d["mix"][:100] = np.nan
    d["mix"][150:] = np.nan
    r = _run(d, hedzh=True)
    assert r.kom_hedzh[100] > 0 and r.kom_hedzh[150] == pytest.approx(KOM * abs(r.h[149]))
    assert r.h[150:].tolist() == [0.0] * 50


def test_imoex_nan_tozhe_gasit_hedzh_i_ryad_konechen():
    d = _kniga_2beta(T=200)
    d["imoex"][120] = np.nan
    r = _run(d, hedzh=True)
    assert r.mix_nan[120] and r.h[120] == 0 and np.isfinite(r.ryad[120])


def test_kommissiya_pri_smene_mnozhitelya_beretsya_zh149():
    """Известный ответ руками: один тикер, позиция P = 1 константна (сигналов нет, ΔP = 0 со 2-го дня), центры не платят.
    Книга обязана взять ставка·|Δ(m·f)|·|P| на каждой смене множителя; день 0 — вход оплачен центром (0)."""
    rng = np.random.default_rng(21)
    T = 200
    sd = np.where(np.arange(T) < 100, 0.004, 0.02)
    d = dict(X=(rng.normal(0, 1, T) * sd)[None, None, :], poz=np.ones((1, 1, T)), sektora=np.array([0]),
             imoex=rng.normal(0, 0.005, T), mix=np.zeros(T), ks=np.zeros(T))
    r = _run(d, tsel_vol=True, vykl=True, sektor_netto=False)
    dmf = np.abs(np.diff(r.m * r.f))
    assert (dmf > 1e-9).sum() > 20
    assert r.kom_tik[0] == 0.0
    assert r.kom_tik[1:] == pytest.approx(KOM * dmf, abs=1e-15)
    ex_port = _run(d).ryad
    assert r.ryad == pytest.approx(ex_port * r.m * r.f - r.marzha - r.kom_tik, abs=1e-15, nan_ok=True)


# ---- Ж-149: нетто сектора от капитала, а не от валовой сектора (усиление мутации автора, она связана через `or`)
def _poz_itog(poz_tikerov, sektora):
    from tests.test_a_shtrih_v7_kniga import _odin_den
    N = len(poz_tikerov)                         # _odin_den делит позицию на N (P = poz/N)
    return _odin_den(np.asarray(poz_tikerov) * N, sektora).poz_itog[:, 0]


def _proverka_pravila(f):
    """Известный ответ: одиночный 0,1 цел; long 0,3 + short 0,05 -> нетто ровно 0,2 (long режется до 0,25)."""
    a = f([0.1, 0.0], [0, 1])
    b = f([0.3, -0.05], [0, 0])
    return np.allclose(a, [0.1, 0.0]) and np.allclose(b, [0.25, -0.05]) and abs(b.sum() - 0.20) < 1e-12


def test_g3_pravilo_ot_kapitala_na_pravilnom_kode():
    assert _proverka_pravila(_poz_itog)


def test_g3_mutatsiya_netto_ot_valovoj_sektora_lovitsya_zhestko(monkeypatch):
    from fastbt.kniga_msp import netto

    def staroe(u, sektor, maks=0.2):
        out = np.asarray(u, float).copy()
        for s in np.unique(sektor):
            idx = np.flatnonzero(sektor == s)
            out[idx] = netto(out[idx], maks)
        return out

    monkeypatch.setattr(K, "sektor_netto", staroe)
    assert not _proverka_pravila(_poz_itog)     # на мутанте одиночный 0,1 обнуляется -> проверка проваливается
