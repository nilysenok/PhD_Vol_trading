"""Л3.2 (Ж-146 п. 3): сборка дня A′-расширенной. Тест написан ДО кода (G0: ImportError на базе).

Вход: X[S,N,T] excess пар (стратегия × тикер), poz[S,N,T] знаковые позиции, сектора[N], imoex[T], mix[T].
Инварианты: G1 выкл. = progon_urovni.a_shtrih(ravnyj_bez_targeta) побайтно; G2 M-d1; G3 нетто сектора;
G4 комиссия хеджа 0,0004·|Δ|; G5 MIX = NaN -> хедж 0 с флагом. Мутации подкладываются подменой функций модуля.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7_kniga as K
from fastbt import progon_urovni as pu
from fastbt.kniga_msp import marzha_v_den

KOM = 0.0004   # литерал в тесте: независимая сверка с константой модуля


def _mk(S=3, N=4, T=150, seed=0, nan=True):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (S, N, T))
    if nan:
        X[rng.random(X.shape) < 0.05] = np.nan
    poz = rng.choice([-1.0, 0.0, 1.0], size=(S, N, T))
    return dict(X=X, poz=poz, sektora=np.arange(N) // 2, imoex=rng.normal(0, 0.01, T),
                mix=rng.normal(0, 0.01, T), ks=np.full(T, 0.08))


def _run(d, **kw):
    a = dict(vesa="vykl", hedzh=False, sektor_netto=False, tsel_vol=False, vykl=False)
    a.update(kw)
    return K.kniga_ryada(d["X"], d["poz"], d["sektora"], d["imoex"], d["mix"], d["ks"], **a)


def _sek_ok(res, sektora, maks=0.20):
    for s in np.unique(sektora):
        P = res.poz_itog[sektora == s]
        if not (np.abs(P.sum(0)) <= maks + 1e-12).all():        # от капитала (NAV), Ж-093 б, не от валовой сектора
            return False
    return True


# ---- G1: выкл. = текущее поведение побайтно
def test_g1_vykl_ravno_a_shtrih_pobajtno():
    d = _mk(S=4, N=5, T=140, seed=1)
    daty = pd.bdate_range("2016-01-04", periods=140)
    konf = next(c for c in pu.konfiguracii() if c["rol"] == "ravnyj_bez_targeta")
    ref = pu.a_shtrih({f"S{k}": d["X"][k] for k in range(4)}, konf, daty)
    got = _run(d).ryad
    assert ref.dtype == got.dtype and ref.tobytes() == got.tobytes()


def test_g1_mutatsiya_lyubaya_vkljuchennaya_opciya_menyaet_ryad():
    d = _mk(S=4, N=5, T=140, seed=1)
    base = _run(d).ryad
    for kw in ({"hedzh": True}, {"sektor_netto": True}, {"tsel_vol": True}, {"vesa": "ravnyj_risk"}):
        assert not np.array_equal(base, _run(d, **kw).ryad, equal_nan=True), kw


# ---- G2: M-d1
def _no_lookahead(d, j=100, **kw):
    r0 = _run(d, **kw)
    d2 = {**d, "X": d["X"].copy(), "imoex": d["imoex"].copy()}
    rng = np.random.default_rng(9)
    d2["X"][:, :, j:] = rng.normal(0, 0.05, d2["X"][:, :, j:].shape)
    d2["imoex"][j:] = rng.normal(0, 0.05, len(d2["imoex"][j:]))
    r1 = _run(d2, **kw)
    return all(np.array_equal(getattr(r0, n)[..., :j + 1], getattr(r1, n)[..., :j + 1], equal_nan=True)
               for n in ("a", "m", "f", "h"))


_ALL = dict(vesa="ravnyj_risk", hedzh=True, sektor_netto=True, tsel_vol=True, vykl=True)
_MUT = {**_ALL, "sektor_netto": False}     # сектор может обнулить день и скрыть мутацию; он проверяется в G3


def test_g2_net_utechki_v_vesa_m_f_h():
    assert _no_lookahead(_mk(T=150, seed=2), **_ALL)


def test_g2_mutatsiya_beta_s_dnyom_d_lovitsya(monkeypatch):
    orig = K.beta
    monkeypatch.setattr(K, "beta", lambda ex, im, i, okno=63: orig(ex, im, i + 1, okno))
    assert not _no_lookahead(_mk(T=150, seed=2), **_MUT)


def test_g2_mutatsiya_sigma_tselevaya_s_dnyom_d_lovitsya(monkeypatch):
    orig = K.mnozhitel_vola
    monkeypatch.setattr(K, "mnozhitel_vola", lambda ex, i, g, tsel=0.10, plecho=2.0: orig(ex, i + 1, g, tsel, plecho))
    assert not _no_lookahead(_mk(T=150, seed=2), **_MUT)


def test_g2_mutatsiya_sigma_vesov_s_dnyom_d_lovitsya(monkeypatch):
    orig = K.vesa_dnya
    monkeypatch.setattr(K, "vesa_dnya", lambda ex, i, okno=252, rezhim="paritet": orig(ex, i + 1, okno, rezhim))
    assert not _no_lookahead(_mk(T=150, seed=2), **_MUT)


def test_g2_mutatsiya_vyklyuchatel_s_dnyom_d_lovitsya(monkeypatch):
    orig = K.vyklyuchatel
    monkeypatch.setattr(K, "vyklyuchatel", lambda ex, i, tsel=0.10: orig(ex, i + 1, tsel))
    d = _mk(T=150, seed=2)
    assert not _no_lookahead(d, vesa="vykl", vykl=True)


# ---- G3: нетто сектора
def _d9(seed=3, N=18, T=120):
    rng = np.random.default_rng(seed)       # позиции ×15: P тикера ~ ±0,1 NAV, сектор из двух легко > 20 %
    return dict(X=rng.normal(0, 0.01, (3, N, T)), poz=15 * rng.choice([-1.0, 1.0], size=(3, N, T)) * rng.random((3, N, T)),
                sektora=np.arange(N) // 2, imoex=rng.normal(0, 0.01, T), mix=rng.normal(0, 0.01, T), ks=np.zeros(T))


def test_g3_netto_sektora_na_kazhdyj_den_9_sektorov():
    d = _d9()
    assert not _sek_ok(_run(d), d["sektora"])                       # без правила нарушения есть
    assert _sek_ok(_run(d, sektor_netto=True), d["sektora"])


def test_g3_mutatsiya_porog_60_vmesto_20_lovitsya():
    d = _d9()
    assert not _sek_ok(_run(d, sektor_netto=True, netto_sek=0.60), d["sektora"])


def test_g3_mutatsiya_ot_valovoj_sektora_lovitsya(monkeypatch):
    from fastbt.kniga_msp import netto
    monkeypatch.setattr(K, "sektor_netto", lambda u, sektor, maks=0.2: netto(u, maks))   # старый дефект
    d = _d9()
    assert not _sek_ok(_run(d, sektor_netto=True), d["sektora"]) or \
        not np.allclose(_odin_den([0.1, 0.0], [0, 1]).poz_itog[0, 0], 0.05)


def _odin_den(poz_tikerov, sektora):
    """K=1 стратегия, N тикеров, одна позиция на тикер, T=3: P тикера = poz/N (все пары присутствуют)."""
    N = len(poz_tikerov)
    p = np.tile(np.asarray(poz_tikerov, float)[None, :, None], (1, 1, 3))
    d = dict(X=np.full((1, N, 3), 0.001), poz=p, sektora=np.asarray(sektora), imoex=np.zeros(3),
             mix=np.zeros(3), ks=np.zeros(3))
    return _run(d, sektor_netto=True)


def test_g3_sektor_iz_odnogo_tikera_do_20_procentov_ne_trogaetsya():
    P = _odin_den([0.1, 0.0], [0, 1]).poz_itog[:, 0]            # P = poz/2 = 0,05 NAV
    assert P[0] == pytest.approx(0.05, abs=1e-15)


def test_g3_odin_tiker_030_rezhetsya_do_020_obe_storony():
    assert _odin_den([0.6, 0.0], [0, 1]).poz_itog[0, 0] == pytest.approx(0.2, rel=1e-12)
    assert _odin_den([-0.6, 0.0], [0, 1]).poz_itog[0, 0] == pytest.approx(-0.2, rel=1e-12)


@pytest.mark.parametrize("tk", ["LSRG", "AFLT"])
def test_g3_realnye_sektora_odin_tiker_005_prohodit_030_rezhetsya_zh149(tk):
    """Ж-149: сектор из одного тикера — обычный лимит. Сектора — PIT-файл `2026-10-27_sektory.csv` (17 тикеров)."""
    from pathlib import Path
    tab = pd.read_csv(Path(__file__).resolve().parent.parent / "docs/issledovaniya/2026-10-27_sektory.csv")
    sek = tab["indeks"].to_numpy()
    j = list(tab["ticker"]).index(tk)
    assert (sek == sek[j]).sum() == 1                                  # сектор действительно из одного тикера
    N = len(tab)
    for P, ozid in ((0.05, 0.05), (0.3, 0.2), (-0.3, -0.2)):
        poz = np.zeros(N)
        poz[j] = P * N                                                 # K=1, все пары есть: P тикера = poz/N
        r = _odin_den(list(poz), list(sek))
        assert r.poz_itog[j, 0] == pytest.approx(ozid, rel=1e-12)


def test_g3_long_030_short_005_daet_netto_020():
    P = _odin_den([1.2, -0.2, 0.0, 0.0], [0, 0, 1, 1]).poz_itog[:, 0]   # P = [.30, -.05]
    assert P[1] == pytest.approx(-0.05, abs=1e-15) and P[0] == pytest.approx(0.25, rel=1e-12)


def test_g3_vse_tikery_odnoj_storony_srezayutsya_proporcionalno():
    r = _odin_den([1.2, 1.2, 0.0, 0.0], [0, 0, 1, 1])           # P = [.3, .3] -> по 0,1
    assert r.poz_itog[:2, 0] == pytest.approx([0.1, 0.1], rel=1e-12) and np.isfinite(r.ryad).all()


def test_g3_perevernutyj_znak_dayot_tot_zhe_ryad():
    d = _mk(S=3, N=6, T=80, seed=5, nan=False)
    d["sektora"] = np.array([0, 0, 0, 1, 1, 1])
    a = _run(d, sektor_netto=True)
    d["poz"] = -d["poz"]
    b = _run(d, sektor_netto=True)
    assert np.array_equal(a.ryad, b.ryad, equal_nan=True) and np.array_equal(a.poz_itog, -b.poz_itog)


def test_g3_sektor_vkl_bez_pozicij_otkaz():
    d = _mk(T=30)
    d["poz"] = None
    with pytest.raises(ValueError):
        _run(d, sektor_netto=True)


# ---- G4: комиссия хеджа
def _kniga_2beta(T=130, seed=7):
    rng = np.random.default_rng(seed)
    im = rng.normal(0, 0.001, T)       # тихая книга: выключатель молчит, пока тест не вызовет скачок
    return dict(X=(2.0 * im)[None, None, :], poz=np.ones((1, 1, T)), sektora=np.array([0]), imoex=im,
                mix=rng.normal(0, 0.01, T), ks=np.zeros(T))


def test_g4_komissiya_hedzha_rovno_0004_na_modul_izmeneniya():
    d = _kniga_2beta()
    r = _run(d, hedzh=True)
    assert K.KOM_HEDZH == KOM
    assert (r.h[:63] == 0).all()
    assert r.h[63] == pytest.approx(-2.0, abs=1e-9)                # β=2 -> продажа 2·NAV
    assert r.kom_hedzh[63] == pytest.approx(KOM * 2.0, abs=1e-11)
    assert np.abs(r.kom_hedzh[64:]).max() < 1e-10                  # β не менялась
    dh = np.abs(np.diff(r.h, prepend=0.0))
    assert np.array_equal(r.kom_hedzh, KOM * dh)
    book = _run(d).ryad
    assert np.allclose(r.ryad, book + r.h * d["mix"] - r.kom_hedzh, atol=1e-15)


def test_g4_znak_hedzha_protivopolozhen_bete():
    d = _kniga_2beta()
    assert _run(d, hedzh=True).h[100] < 0
    d["X"] = -d["X"]
    assert _run(d, hedzh=True).h[100] > 0


def test_g4_mutatsiya_bez_komissii_lovitsya(monkeypatch):
    d = _kniga_2beta()
    monkeypatch.setattr(K, "KOM_HEDZH", 0.0)
    assert not np.isclose(_run(d, hedzh=True).kom_hedzh[63], KOM * 2.0)


# ---- G5: MIX = NaN
def test_g5_mix_nan_hedzh_nol_s_flagom():
    d = _kniga_2beta(T=160)
    d["mix"][:70] = np.nan
    d["mix"][140:] = np.nan
    r = _run(d, hedzh=True)
    assert (r.h[:70] == 0).all() and r.mix_nan[:70].all() and r.mix_nan[140:].all() and not r.mix_nan[70:140].any()
    assert (r.h[140:] == 0).all()
    assert r.kom_hedzh[70] == pytest.approx(KOM * 2.0, abs=1e-10)       # открытие
    assert r.kom_hedzh[140] == pytest.approx(KOM * 2.0, abs=1e-10)      # закрытие при NaN
    assert np.isfinite(r.ryad).all()


# ---- плечо, таргет, маржа
def test_plecho_dva_i_target_odnovremenno():
    rng = np.random.default_rng(4)
    T = 100
    d = dict(X=rng.normal(0, 1e-5, (1, 1, T)), poz=np.full((1, 1, T), 1.5), sektora=np.array([0]),
             imoex=rng.normal(0, 0.01, T), mix=np.zeros(T), ks=np.full(T, 0.08))
    r = _run(d, tsel_vol=True)
    assert (r.m[:21] == 1.0).all()
    assert np.allclose(r.m[22:], 2.0 / 1.5, rtol=1e-12)                 # tsel/σ огромно, режет плечо 2
    assert (np.abs(r.poz_itog[0]) * r.m <= 2.0 + 1e-12).all()
    mz = marzha_v_den(2.0, 0.08)
    assert mz == pytest.approx(r.marzha[50], rel=1e-12) and mz > 0
    assert r.ryad[50] == pytest.approx(d["X"][0, 0, 50] * r.m[50] - mz, rel=1e-12)


def test_target_10_procentov_pri_obychnoj_vole():
    rng = np.random.default_rng(6)
    T = 200
    x = rng.normal(0, 0.02, T)
    d = dict(X=x[None, None, :], poz=np.full((1, 1, T), 0.5), sektora=np.array([0]), imoex=np.zeros(T),
             mix=np.zeros(T), ks=np.zeros(T))
    r = _run(d, tsel_vol=True)
    i = 150
    assert r.m[i] == pytest.approx(0.10 / (x[i - 21:i].std(ddof=1) * np.sqrt(252)), rel=1e-12)


# ---- выключатель и порядок: хедж ×f
def test_vyklyuchatel_poltora_i_hedzh_umnozhen_na_f():
    d = _kniga_2beta(T=140)
    d["X"][0, 0, 120:130] *= 40.0                                       # скачок воли в окне 10 дней
    r = _run(d, hedzh=True, vykl=True)
    assert r.f[131] == 0.5 and r.f[100] == 1.0
    b = r.h[131] / r.f[131]
    off = _run(d, hedzh=True, vykl=False)
    assert off.f[131] == 1.0 and (r.h[131] / 0.5) == pytest.approx(b)
    assert off.h[131] == pytest.approx(2 * r.h[131])


# ---- веса Sk
def test_vesa_ravnyj_risk_obratno_sigma_i_vykl_porovnu():
    d = _mk(S=3, N=3, T=120, seed=8, nan=False)
    d["X"][1] *= 3.0
    r = _run(d, vesa="ravnyj_risk")
    ex = np.nanmean(d["X"], axis=1).T
    w = 1.0 / ex[:100].std(axis=0)
    assert np.allclose(r.a[:, 100], w / w.sum(), rtol=1e-12) and r.vesa_flagi[100] == "korotkoe_okno"
    assert np.allclose(_run(d, vesa="vykl").a, 1 / 3)


def test_vesa_paritet_okno_252_flag_ok_i_summa_edinica():
    d = _mk(S=3, N=3, T=270, seed=10, nan=False)
    r = _run(d, vesa="paritet")
    assert r.vesa_flagi[262] == "ok" and r.vesa_flagi[100] == "korotkoe_okno"
    assert np.isfinite(r.a).all() and np.allclose(r.a.sum(0), 1.0)


def test_vesa_stolbec_nan_ne_daet_nan_i_net_validnyh_ravno():
    d = _mk(S=3, N=3, T=120, seed=11, nan=False)
    d["X"][2] = np.nan
    r = _run(d, vesa="ravnyj_risk")
    assert np.isfinite(r.a).all() and r.a[2, 100] == 0.0 and r.a[:2, 100].sum() == pytest.approx(1.0)
    d["X"][:] = 0.01                                                    # sigma = 0 у всех
    r = _run(d, vesa="ravnyj_risk")
    assert np.allclose(r.a[:, 100], 1 / 3) and r.vesa_flagi[100] == "net_validnyh"


# ---- попробуй сломать
def test_pustaya_lenta_otkaz():
    d = _mk(T=5)
    for k in ("X", "poz"):
        d[k] = d[k][..., :0]
    d["imoex"], d["mix"], d["ks"] = d["imoex"][:0], d["mix"][:0], d["ks"][:0]
    with pytest.raises(ValueError):
        _run(d)


def test_nulevoj_obem_nulevoj_ryad_konechnyj_vse_vkl():
    d = _mk(T=150, seed=12, nan=False)
    d["X"][:] = 0.0
    r = _run(d, **_ALL)
    assert np.isfinite(r.ryad).all() and (r.m[22:] == 1.0).all()


def test_dyra_v_dannyh_ryad_nan_a_ne_nol():
    d = _mk(T=60, seed=13, nan=False)
    d["X"][:, :, 30] = np.nan
    r = _run(d)
    assert np.isnan(r.ryad[30]) and np.isfinite(r.ryad[29]) and np.isfinite(r.ryad[31])


def test_nepravilnye_flagi_i_formy():
    d = _mk(T=30)
    with pytest.raises(ValueError):
        _run(d, vesa="foo")
    d["imoex"] = d["imoex"][:-1]
    with pytest.raises(ValueError):
        _run(d)
