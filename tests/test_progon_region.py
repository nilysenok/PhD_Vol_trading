# -*- coding: utf-8 -*-
"""Л-7: регион и плато на синтетике с известным ответом (куб 4 оси × 5 + 2 шумовые оси), мутации ловятся.

Правка тестировщика (П-7, Ж-086): строка без осей t_vyhod / vhod теперь ValueError (progon_osnovnaya), поэтому в pt каждой строки
добавлены постоянные оси OSN с основными значениями. Ожидания и требования тестов не менялись."""
import itertools

import numpy as np
import pytest

from fastbt import pbo_pravilo as PR, progon_region as R

AX = ["a", "b", "c", "d"]
OSN = {"t_vyhod": "open2", "vhod": "0.5"}
CENTR = (2, 2, 2, 2)
Q = 20


def kub(spike=0.0, flip=(), n1_second_half=0.0, days_q=range(16), seed=1):
    pts = [dict(zip(AX + ["n1", "n2"], v)) for v in itertools.product(range(5), range(5), range(5), range(5), (0, 1), (0, 1))]
    rows = [{"pt": {**p, **OSN}, "sdelok_y": {"2016": 40, "2017": 40}} for p in pts]
    rng = np.random.default_rng(seed)
    start = np.array(["2016-01-01"], "datetime64[M]")[0] + 3 * np.arange(16)
    daty, cols = [], []
    for k in days_q:
        d = start[k].astype("datetime64[D]") + np.arange(Q)
        mu = np.array([0.02 * np.exp(-sum((p[a] - 2) ** 2 for a in AX) / 2) for p in pts])
        if k in flip:
            mu = -mu
        if k >= 8:
            mu = mu * (n1_second_half == 0) + n1_second_half * np.array([p["n1"] for p in pts])
        if spike:
            mu[[j for j, p in enumerate(pts) if (p["a"], p["b"], p["c"], p["d"], p["n1"], p["n2"]) == (4, 4, 4, 4, 0, 0)]] += spike
        daty.append(d)
        cols.append(mu[None, :] - 0.002 + rng.normal(0, 0.005, (Q, len(pts))))
    return np.vstack(cols), rows, np.concatenate(daty)


def centr_names(res):
    return dict(zip(res["osi"], res["centr"]))


def test_rang_osey_signal_vyshe_shuma():
    M, rows, d = kub(seed=3)
    t = R._cel_stolbcov(M, d, list(range(16)), "mediana")
    st = dict(zip(R.osi_rows(rows), R.rang_osey(rows, t, R.osi_rows(rows))))
    assert min(st[a] for a in AX) > max(st["n1"], st["n2"])


def test_region_vybiraet_plato_a_ne_pik():
    M, rows, d = kub(spike=0.05)
    res = R.vybor_region(M, rows, d)
    assert res is not None and set(res["osi"]) == set(AX)
    assert tuple(centr_names(res)[a] for a in AX) == CENTR
    assert np.allclose(res["ryad"], M[:, res["chleny"]].mean(axis=1))


def test_regiony_chleny_radius_1():
    _, rows, _ = kub()
    reg = R.regiony(rows, AX, 1)
    assert len(reg[CENTR]) == 3 ** 4 * 4 and len(reg[(0, 0, 0, 0)]) == 2 ** 4 * 4
    assert len(reg) == 5 ** 4


def test_ryad_regiona_srednee():
    M = np.array([[1.0, 2.0, 9.0], [0.0, 0.0, 3.0]])
    assert np.allclose(R.ryad_regiona(M, [0, 1, 2]), [4.0, 1.0])


def test_dolya_075_ne_dopustim():
    M, rows, d = kub(flip=(0, 1, 2, 3))
    assert R.vybor_region(M, rows, d) is None


def test_dolya_nan_kvartaly_ne_dopustim():
    M, rows, d = kub(days_q=range(12))
    assert R.vybor_region(M, rows, d) is None
    assert R.vybor_region(M, rows, d, tsel="dolya") is None
    assert R.vybor_region(M, rows, d, kvartaly_idx=range(12)) is not None


def test_pustaya_matritsa_i_nul_sigma():
    M, rows, d = kub()
    assert R.vybor_region(np.zeros_like(M), rows, d) is None
    assert np.all(R._kv_sr(np.zeros((40, 2)), d[:40], [0]) == 0)


def test_polovina_kvartalov_stavit_osi_po_polovine():
    M, rows, d = kub(n1_second_half=0.012)
    pol = R.vybor_region(M, rows, d, kvartaly_idx=range(8))
    assert pol is not None and set(pol["osi"]) == set(AX)
    osi = R.osi_rows(rows)
    st = R.rang_osey(rows, R._cel_stolbcov(M, d, list(range(16)), "mediana"), osi)
    top16 = {osi[i] for i in np.argsort(-np.asarray(st))[:R.K_OSI]}
    assert "n1" in top16


def test_select_fn_i_cscv():
    M, rows, d = kub(spike=0.05)
    fn = R.select_fn_region(rows, d)
    j = fn(M[:160].T, None, d[:160])
    assert j is not None and 0 <= j < len(rows)
    out = PR.cscv(M.T, [r["pt"] for r in rows], fn, nblk=4, dates=d)
    assert out["n"] + out["n_propusk"] == 6


# ------------------------------------------------------------------ мутации


def test_mut_argmax_tochki_menyaet_vybor():
    M, rows, d = kub(spike=0.05)
    res = R.vybor_region(M, rows, d)
    spike = int(np.argmax(M.mean(axis=0)))
    assert spike not in res["chleny"]            # argmax точки вместо региона вернул бы spike


def test_mut_osi_po_vsem_16_kvartalam(monkeypatch):
    M, rows, d = kub(n1_second_half=0.012)
    ok = R.vybor_region(M, rows, d, kvartaly_idx=range(8))
    orig = R._cel_stolbcov
    monkeypatch.setattr(R, "_cel_stolbcov", lambda M_, d_, kv, t: orig(M_, d_, list(range(16)), t))
    bad = R.vybor_region(M, rows, d, kvartaly_idx=range(8))
    assert bad is None or set(bad["osi"]) != set(ok["osi"])


def test_mut_dolya_nan_kak_polozhitelnyj(monkeypatch):
    M, rows, d = kub(days_q=range(12))
    monkeypatch.setattr(R, "_dolya_pol", lambda sr: float(1 - np.mean(np.asarray(sr, float) <= 0)))
    assert R.vybor_region(M, rows, d, tsel="dolya") is not None      # мутант допускает; без мутации — None


def test_mut_mediana_vmesto_srednego(monkeypatch):
    M = np.array([[1.0, 2.0, 9.0]])
    monkeypatch.setattr(R, "ryad_regiona", lambda M_, ch: np.median(np.asarray(M_)[:, list(ch)], axis=1))
    assert R.ryad_regiona(M, [0, 1, 2])[0] != pytest.approx(4.0)
