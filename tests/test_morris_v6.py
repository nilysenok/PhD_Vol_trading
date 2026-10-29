"""V6-14: траектории Морриса, условный μ*, U95. Синтетика с известным ответом; данных нет, в сеть не ходит.

Мутации, которые тесты обязаны поймать: Δ ≠ 2/3 (шаг на один уровень); индекс floor вместо round (двоичная ось не переключается);
в μ* входят неактивные шаги; шаг с NaN в Y считается нулём.
"""
from __future__ import annotations

import numpy as np

from fastbt import morris_v6 as M

L = [2, 2, 3, 4, 2, 5, 2, 2, 3, 2, 2, 4]  # k = 12


def _traj(r=20, seed=7, L=L):
    return M.traektorii(L, r, seed)


def test_indeks_round_i_dvoichnaya_os_pereklyuchaetsya():
    assert M.indeks([0, 1, 2, 3], 2).tolist() == [0, 0, 1, 1]
    assert M.indeks([0, 1, 2, 3], 4).tolist() == [0, 1, 2, 3]
    assert M.indeks([0, 1, 2, 3], 3).tolist() == [0, 1, 1, 2]
    assert M.indeks([0, 1, 2, 3], 5).tolist() == [0, 1, 3, 4]
    # мутация floor(j·(L−1)/3): L=2, j=2 даёт 0 — шаг 0→2/3 не переключал бы ось
    I, por = _traj()
    for m in range(len(I)):
        for t in range(1, len(L) + 1):
            a = por[m, t - 1]
            if L[a] == 2:
                assert I[m, t, a] != I[m, t - 1, a]


def test_shag_dva_urovnya_i_kazhdaya_os_ravno_raz():
    J, por = M.kandidaty(len(L), 50, np.random.default_rng(1))
    d = np.abs(J[:, 1:].astype(int) - J[:, :-1].astype(int))
    assert (d.sum(2) == 2).all() and (d.max(2) == 2).all()  # Δ = 2/3 = два уровня, одна ось за шаг
    assert all(sorted(p) == list(range(len(L))) for p in por)


def test_traektorii_forma_i_odna_os_za_shag():
    I, por = _traj()
    assert I.shape == (20, len(L) + 1, len(L)) and por.shape == (20, len(L))
    diff = I[:, 1:] != I[:, :-1]
    assert (diff.sum(2) == 1).all()  # на каждом шаге меняется ровно одна ось, и это ось порядка
    assert (np.take_along_axis(diff, por[:, :, None], axis=2)[:, :, 0]).all()
    assert (I >= 0).all() and (I < np.array(L)).all()


def test_otbor_kampalongo_raznesennee_sluchainogo_i_vosproizvodim():
    J, _ = M.kandidaty(len(L), 200, np.random.default_rng(3))
    d2 = M.razneseniye(J)
    keep = M.otbor(J, 20)
    obshchee = lambda ix: d2[np.ix_(ix, ix)].sum()
    sluchaynye = [obshchee(np.random.default_rng(s).choice(200, 20, replace=False)) for s in range(30)]
    assert len(keep) == 20 and obshchee(keep) > max(sluchaynye)
    assert np.array_equal(_traj()[0], _traj()[0])
    assert not np.array_equal(_traj()[0], _traj(seed=8)[0])


def test_razneseniye_sovpadaet_s_pryamym_schetom():
    J, _ = M.kandidaty(5, 6, np.random.default_rng(4))
    u = J / 3.0
    pryamo = np.zeros((6, 6))
    for m in range(6):
        for l in range(6):
            if m != l:
                pryamo[m, l] = sum(np.linalg.norm(a - b) for a in u[m] for b in u[l]) ** 2
    assert np.allclose(M.razneseniye(J), pryamo)


def test_masshtab_S1_k65_r75():
    I, por = M.traektorii([2] * 40 + [3] * 15 + [5] * 10, 75, M.SEED)
    assert I.shape == (75, 66, 65) and (I[:, 1:] != I[:, :-1]).sum(2).min() == 1


def _lin(I, c):
    return (I * np.asarray(c)).sum(2)


def test_izvestnyj_otvet_linejnaya_i_nulevaya_os():
    Lb = [2] * 10
    c = [1.0, -0.5, 0.25, 2.0, 0.0, 0.0, 0.1, 0.3, -0.7, 0.0]
    I, por = M.traektorii(Lb, 24, 11)
    keys = np.array([[hash(tuple(p)) for p in tr] for tr in I])
    st = M.statistika(M.effekty(_lin(I, c), keys, por), B=200)
    assert (st["n_akt"] == 24).all()
    assert np.allclose(st["mu_star"], np.abs(c))  # μ*_i = |c_i|
    assert np.allclose(st["sigma"][[4, 5, 9]], 0.0) and (st["sigma"][[0, 3]] > 0).all()  # σ по EE со знаком: ±c_i
    assert np.allclose(st["U95"], np.abs(c))  # все |EE| равны |c_i| → любая выборка даёт |c_i|
    assert st["U95"][[4, 5, 9]].max() == 0.0  # нулевая ось: μ* = 0, U95 = 0


def test_neaktivnye_shagi_ne_v_mu_star():
    Lb = [2] * 6  # ось 0 — родитель; ось 1 — дочерняя: действует только при родителе «вкл.»
    I, por = M.traektorii(Lb, 40, 5)
    kan = I.copy()
    kan[:, :, 1] = np.where(I[:, :, 0] == 0, 0, I[:, :, 1])  # canon: дочерняя при выкл. родителе схлопнута
    keys = np.array([[hash(tuple(p)) for p in tr] for tr in kan])
    st = M.statistika(M.effekty(_lin(kan, [1, 3, 0, 0, 0, 0]), keys, por), B=200)
    assert 0 < st["n_akt"][1] < 40
    assert np.isclose(st["mu_star"][1], 3.0)  # по активным шагам; мутация «все шаги» дала бы 3·n_akt/40
    assert st["n_akt"][0] == 40


def test_net_Y_shag_propuskaetsya_a_ne_nol():
    Lb = [2] * 4
    I, por = M.traektorii(Lb, 10, 2)
    Y = _lin(I, [1, 1, 1, 1]).astype(float)
    Y[0, 2] = np.nan
    keys = np.array([[hash(tuple(p)) for p in tr] for tr in I])
    E = M.effekty(Y, keys, por)
    assert np.isnan(E[0]).sum() == 2  # два шага траектории 0 затронуты NaN (в и из точки 2)
    st = M.statistika(E, B=100)
    assert np.allclose(st["mu_star"], 1.0)  # NaN-шаги не тянут μ* к нулю


def test_U95_chuvstvitelna_k_shumu_i_vyshe_mu():
    rng = np.random.default_rng(0)
    E = rng.normal(0, 1.0, size=(60, 3))
    E[:, 2] = np.nan
    st = M.statistika(E, B=500)
    assert st["U95"][0] > st["mu_star"][0] and st["U95"][0] - st["mu_star"][0] < 0.3
    assert np.isnan(st["mu_star"][2]) and np.isnan(st["U95"][2]) and st["n_akt"][2] == 0
