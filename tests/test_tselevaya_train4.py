# -*- coding: utf-8 -*-
"""Ж-090 п. 1: целевая этапа 1 — mSR за весь train4 + условие «mSR каждого года > 0». Синтетика с известным ответом.

Мутации (каждая роняет проверку, `lovitsya`): min по годам вместо mSR за период; год с mSR ≤ 0 проходит; плата не вычтена;
выбор региона по прежней целевой (медиана кварталов)."""
import itertools

import numpy as np
import pandas as pd
import pytest

from fastbt import progon_region as R, progon_wf as W, tselevaya as T
from fastbt.etap1_poisk_vorota import tselevaya_train4 as tsel4
from fastbt.metriki_e import msr

DATY = pd.bdate_range("2016-01-01", "2019-12-31").values.astype("datetime64[D]")
GOD = DATY.astype("datetime64[Y]").astype(int) + 1970


def ryad(mu_po_godam, seed=0, sd=0.005):
    mu = np.array([mu_po_godam[g] for g in GOD])
    return mu + np.random.default_rng(seed).normal(0, sd, len(DATY))


def lovitsya(proverka, mutatsiya, monkeypatch, *args):
    proverka(*args)                                             # на правке проходит
    monkeypatch.setattr(*mutatsiya)
    with pytest.raises(AssertionError):                         # на мутации падает
        proverka(*args)


# ---- tselevaya_train4 ----
def proverka_vsego(fn):
    r = ryad({2016: 0.004, 2017: 0.001, 2018: 0.004, 2019: 0.001})
    with pytest.warns(UserWarning):
        m, py, ust = fn(r, DATY)
    assert m == pytest.approx(msr(r))                           # одно определение mSR, весь период
    assert ust and set(py) == {2016, 2017, 2018, 2019}
    assert m != pytest.approx(min(py.values()), abs=1e-6)       # не min по годам


def test_msr_vsego_a_ne_min_po_godam():
    proverka_vsego(tsel4)


def test_mutatsiya_min_po_godam_lovitsya(monkeypatch):
    def mut(r, daty, zaem_baza=None, stavka=0.05):
        m, py, ust = tsel4(r, daty, zaem_baza, stavka)
        return min(py.values()), py, ust
    proverka_vsego(tsel4)
    with pytest.raises(AssertionError):
        proverka_vsego(mut)


def proverka_ustojchivost(fn):
    r = ryad({2016: 0.004, 2017: -0.004, 2018: 0.004, 2019: 0.004})
    with pytest.warns(UserWarning):
        m, py, ust = fn(r, DATY)
    assert py[2017] < 0 < m and not ust                         # итог положителен, но 2017 ≤ 0: не устойчива
    r2 = ryad({2016: 0.004, 2017: 0.004, 2018: 0.004, 2019: 0.004})
    with pytest.warns(UserWarning):
        assert fn(r2, DATY)[2]


def test_god_s_msr_nizhe_nulya_ne_prohodit():
    proverka_ustojchivost(tsel4)


def test_mutatsiya_god_nizhe_nulya_prohodit_lovitsya(monkeypatch):
    real = T.msr_i_gody
    monkeypatch.setattr(T, "msr_i_gody", lambda *a, **k: (lambda m, py, ust, ys: (m, py, np.ones_like(ust), ys))(*real(*a, **k)))
    with pytest.raises(AssertionError):
        proverka_ustojchivost(T.tselevaya_train4)


def test_nol_ili_pustoj_god_ne_ustojchiv():
    r = ryad({y: 0.004 for y in (2016, 2017, 2018, 2019)})
    with pytest.warns(UserWarning):
        assert not tsel4(r[GOD != 2017], DATY[GOD != 2017])[2]  # нет 2017 -> nan -> не устойчива


def test_plata_za_short_vychitaetsya():
    r = ryad({y: 0.002 for y in (2016, 2017, 2018, 2019)})
    z = np.full(len(r), 0.03)                                   # 0,05 · 0,03 = 0,0015 за день
    m, py, ust = tsel4(r, DATY, z, 0.05)
    assert m == pytest.approx(msr(r - 0.0015)) and m < msr(r)
    assert all(abs(py[y] - msr((r - 0.0015)[GOD == y])) < 1e-9 for y in py)
    assert tsel4(r, DATY, z, 0.0)[0] == pytest.approx(msr(r))   # ставка 0 = без платы


def test_plata_ne_vychtena_lovitsya(monkeypatch):
    r = ryad({y: 0.002 for y in (2016, 2017, 2018, 2019)})
    z = np.full(len(r), 0.03)
    real = T.tselevaya_train4
    monkeypatch.setattr(T, "tselevaya_train4", lambda e, d, zb=None, stavka=0.05: real(e, d, None, stavka))
    with pytest.warns(UserWarning), pytest.raises(AssertionError):
        m = T.tselevaya_train4(r, DATY, z, 0.05)[0]
        assert m == pytest.approx(msr(r - 0.0015))


def test_bez_zaema_yavnoe_preduprezhdenie_i_dlina():
    r = ryad({y: 0.002 for y in (2016, 2017, 2018, 2019)})
    with pytest.warns(UserWarning, match="не учтена"):
        tsel4(r, DATY)
    with pytest.raises(ValueError):
        tsel4(r, DATY, np.zeros(5))


# ---- регион и WF ----
AX = ["a", "b", "c"]
OSN = {"t_vyhod": "open2", "vhod": "0.5"}
Q = 20


def kub():
    """a ∈ {0,1}: большой mSR за период, но 2017 отрицателен (не устойчив); a ∈ {3,4}: скромный, но положителен во все годы."""
    pts = [dict(zip(AX, v)) for v in itertools.product(range(5), repeat=3)]
    rows = [{"pt": {**p, **OSN}, "sdelok_y": {"2016": 40, "2017": 40}} for p in pts]
    start = np.array(["2016-01-01"], "datetime64[M]")[0] + 3 * np.arange(16)
    rng = np.random.default_rng(5)
    daty, cols = [], []
    for k in range(16):
        d = start[k].astype("datetime64[D]") + np.arange(Q)
        god = 2016 + k // 4
        mu = np.array([(-0.006 if god == 2017 else 0.005) if p["a"] <= 1 else 0.003 if p["a"] >= 3 else 0.0 for p in pts])
        daty.append(d)
        cols.append(mu[None, :] + rng.normal(0, 0.005, (Q, len(pts))))
    return np.vstack(cols), rows, np.concatenate(daty)


def vybrannoe_a(res):
    assert res is not None
    return {dict(zip(res["osi"], res["centr"]))["a"]}


def proverka_region(vybor):
    M, rows, d = kub()
    res = vybor(M, rows, d)
    assert vybrannoe_a(res) <= {3, 4}                           # устойчивый регион, а не «звезда» с провальным годом
    ty = R._cel_stolbcov(M, d, list(range(16)), "msr")
    assert np.isnan(ty[[j for j, r in enumerate(rows) if r["pt"]["a"] <= 1]]).all()   # неустойчивые — nan
    assert np.isfinite(ty[[j for j, r in enumerate(rows) if r["pt"]["a"] >= 3]]).mean() > 0.9


def test_region_vybiraet_ustojchivyj_po_msr():
    proverka_region(R.vybor_region)


def test_mutatsiya_vybor_po_prezhnej_celi_lovitsya(monkeypatch):
    """Прежняя целевая: медиана квартальных Шарпов без условия по годам — выбирает «звезду» a ≤ 1."""
    def staraya(M, daty, kv, tsel):
        sr = R._kv_sr(M, daty, kv)
        return np.median(sr, axis=1)
    lovitsya(proverka_region, (R, "_cel_stolbcov", staraya), monkeypatch, R.vybor_region)


def test_rezhima_mediana_net():
    M, rows, d = kub()
    with pytest.raises(ValueError):
        R.vybor_region(M, rows, d, tsel="mediana")
    assert W.OSNOVNOE[2] == "msr" and {p[2] for p in W.PRAVILA} == {"msr", "dolya"}


# ---- PBO (Ж-080 + Ж-090): ранг OOS и выбор — по той же целевой ----
def dve_tochki():
    """P: высокий mSR за период, слабый последний год (min по годам низкий); Q: ровный. Обе устойчивы. Строки — точки."""
    P = ryad({2016: 0.004, 2017: 0.004, 2018: 0.004, 2019: 0.0012}, seed=1)
    Q_ = ryad({2016: 0.0025, 2017: 0.0025, 2018: 0.0025, 2019: 0.0025}, seed=2)
    return np.vstack([P, Q_])


def proverka_pbo_rang(oos_fn):
    X = dve_tochki()
    a = oos_fn(X, DATY)
    assert a[0] > a[1]                                          # по mSR за период P лучше Q (по min годов было бы наоборот)


def test_pbo_oos_rang_po_msr_vsego():
    from fastbt import pbo_pravilo as B
    proverka_pbo_rang(B.msr_oos)
    assert B.min_msr(dve_tochki(), DATY)[0] < B.min_msr(dve_tochki(), DATY)[1]     # справочная min_msr ранжирует наоборот


def test_mutatsiya_pbo_rang_po_prezhnej_celi_lovitsya():
    from fastbt import pbo_pravilo as B
    with pytest.raises(AssertionError):
        proverka_pbo_rang(B.min_msr)


def test_pbo_neustojchivaya_tochka_hudshaya_i_ne_kandidat():
    from fastbt import pbo_pravilo as B
    bad = ryad({2016: 0.004, 2017: -0.004, 2018: 0.004, 2019: 0.004}, seed=3)
    X = np.vstack([bad, dve_tochki()[1]])
    assert np.isnan(B.msr_celevaya(X, DATY)[0]) and np.isfinite(B.msr_celevaya(X, DATY)[1])
    a = B.msr_oos(X, DATY)
    assert a[0] == -np.inf and a[1] > a[0]


def test_pravilo_etapa_oos_fn_novaya():
    from fastbt import pbo_pravilo as B
    from fastbt.etap1_poisk_vorota import Ledger
    pr = B.pravilo_etapa(Ledger(lambda p, per: {}), {"k_s": [1]}, ["k_s"], {}, {})
    assert pr["oos_fn"] is B.msr_oos


def test_dolya_godov_slabee_msr():
    """Один провальный год из четырёх: «msr» отвергает, «dolya» (≥ 75 % годов) пропускает; два провальных — отвергает оба."""
    M, rows, d = kub()
    kv = list(range(16))
    zvezda = [j for j, r in enumerate(rows) if r["pt"]["a"] <= 1]
    assert np.isnan(R._cel_stolbcov(M, d, kv, "msr")[zvezda]).all()
    assert np.isfinite(R._cel_stolbcov(M, d, kv, "dolya")[zvezda]).mean() > 0.9
    M2 = M.copy()
    M2[np.isin((d.astype("datetime64[Y]").astype(int) + 1970), (2018,))] *= -1     # к 2017 добавлен провальный 2018
    assert np.isnan(R._cel_stolbcov(M2, d, kv, "dolya")[zvezda]).all()
