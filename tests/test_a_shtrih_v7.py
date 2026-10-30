"""Л3.1 a_shtrih_v7: Ledoit-Wolf (цель - диагональ), паритет риска, равный риск.

Тесты на синтетике; данные не читаются, сеть не используется
(чтение >= 2020 и цена после T: неприменимо, функции чистые).
"""

import numpy as np
import pytest

from fastbt import a_shtrih_v7 as A


def _lw_ruki(R):
    """Эталон: формула Ledoit-Wolf (2004) для цели F = diag(S), чистые циклы.

    S = (1/T) sum x x'; pi_ij = (1/T) sum_t (x_ti x_tj - s_ij)^2;
    gamma = sum_{i!=j} s_ij^2; rho = 0 вне диагонали (f_ij = 0 константа);
    delta = clip(sum_{i!=j} pi_ij / gamma / T, 0, 1).
    """
    T, n = len(R), len(R[0])
    m = [sum(R[t][i] for t in range(T)) / T for i in range(n)]
    x = [[R[t][i] - m[i] for i in range(n)] for t in range(T)]
    s = [[sum(x[t][i] * x[t][j] for t in range(T)) / T for j in range(n)] for i in range(n)]
    pi = gam = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                pi += sum((x[t][i] * x[t][j] - s[i][j]) ** 2 for t in range(T)) / T
                gam += s[i][j] ** 2
    d = min(1.0, max(0.0, pi / gam / T))
    out = [[(1 - d) * s[i][j] if i != j else s[i][i] for j in range(n)] for i in range(n)]
    return np.array(out), d


R3 = np.array(
    [[0.01, 0.02, -0.01], [-0.02, -0.01, 0.00], [0.03, 0.01, 0.02], [0.00, -0.02, -0.01],
     [0.01, 0.03, 0.01], [-0.01, 0.00, 0.02], [0.02, -0.01, -0.02], [-0.03, -0.02, 0.01]]
)


def test_lw_sovpadaet_so_spravochnoj_formuloj_3_aktiva():
    sig, d = _lw_ruki(R3.tolist())
    assert 0.0 < d < 1.0
    assert A.lw_delta(R3) == pytest.approx(d, abs=1e-12)
    assert np.allclose(A.lw(R3), sig, atol=1e-14, rtol=0)


def test_lw_sovpadaet_so_spravochnoj_formuloj_2_aktiva():
    R = R3[:, :2]
    sig, d = _lw_ruki(R.tolist())
    assert np.allclose(A.lw(R), sig, atol=1e-14, rtol=0)
    # 2 актива: delta = pi_12 / (T s_12^2) - выписано отдельно
    x = R - R.mean(0)
    s12 = (x[:, 0] * x[:, 1]).mean()
    pi12 = ((x[:, 0] * x[:, 1] - s12) ** 2).mean()
    assert A.lw_delta(R) == pytest.approx(min(1.0, pi12 / (len(R) * s12**2)), abs=1e-12)
    assert d == pytest.approx(A.lw_delta(R), abs=1e-12)


def test_lw_diagonal_sohranena_vne_diagonali_usazhena():
    S = np.cov(R3.T, bias=True)
    L = A.lw(R3)
    assert np.allclose(np.diag(L), np.diag(S), atol=1e-15)
    off = ~np.eye(3, dtype=bool)
    assert np.all(np.abs(L[off]) <= np.abs(S[off]) + 1e-18)
    assert np.allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() > 0


def test_lw_nezavisimye_aktivy_usadka_k_diagonali_silnaya_korrelyaciya_slabaya():
    rng = np.random.default_rng(1)
    T = 252
    Z = rng.standard_normal((T, 4))
    d_nez = A.lw_delta(Z)
    c = 0.9
    common = rng.standard_normal((T, 1))
    Zc = np.sqrt(c) * common + np.sqrt(1 - c) * Z
    d_kor = A.lw_delta(Zc)
    assert d_nez > 0.5
    assert d_kor < 0.1
    assert d_nez > d_kor


def test_lw_znanie_istinnoj_kovariacii_oshibka_men_she_vyborochnoj():
    rng = np.random.default_rng(7)
    n, T = 6, 30
    sig_true = np.diag(np.linspace(1.0, 2.0, n) ** 2)
    L = np.linalg.cholesky(sig_true)
    e_lw = e_s = 0.0
    for _ in range(200):
        R = rng.standard_normal((T, n)) @ L.T
        e_lw += np.linalg.norm(A.lw(R) - sig_true) ** 2
        e_s += np.linalg.norm(np.cov(R.T, bias=True) - sig_true) ** 2
    assert e_lw < 0.8 * e_s


def test_lw_ploxoj_vhod_otkaz():
    with pytest.raises(ValueError):
        A.lw(np.array([[0.1, 0.2]]))
    bad = R3.copy()
    bad[2, 1] = np.nan
    with pytest.raises(ValueError):
        A.lw(bad)
    with pytest.raises(ValueError):
        A.lw(R3[:, 0])


def test_lw_nulevaya_dispersiya_stolbca_bez_nan():
    R = R3.copy()
    R[:, 1] = 0.005
    L = A.lw(R)
    assert np.all(np.isfinite(L))
    assert abs(L[1, 1]) < 1e-30


def _vkl(w, S):
    return w * (S @ w)


def test_paritet_diagonal_ravno_obratnoj_sigma():
    sig = np.array([0.5, 1.0, 2.0, 4.0])
    w = A.paritet(np.diag(sig**2))
    exp = (1 / sig) / (1 / sig).sum()
    assert np.allclose(w, exp, atol=1e-9)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)


def test_paritet_vklady_ravny_1e8_i_polozhitelnye_vesa():
    rng = np.random.default_rng(3)
    for n in (2, 3, 5, 9):
        X = rng.standard_normal((60, n)) + 0.8 * rng.standard_normal((60, 1))
        S = np.cov(X.T, bias=True)
        w = A.paritet(S)
        assert np.all(w > 0)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        rc = _vkl(w, S)
        assert np.max(np.abs(rc - rc.mean())) < 1e-8 * rc.mean()


def test_paritet_3_aktiva_izvestnyj_otvet_korrelirovannye():
    # sigma=(1,1,2), rho12=0.5, остальные 0: проверка по определению вкладов
    S = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 4.0]])
    w = A.paritet(S)
    rc = _vkl(w, S)
    assert np.allclose(rc, rc.mean(), rtol=0, atol=1e-8 * rc.mean())
    # симметрия активов 1 и 2
    assert w[0] == pytest.approx(w[1], abs=1e-10)
    # w3: 4 w3^2 = (w1 + 0.5 w1) w1 * ... вклад 1 = w1(w1+0.5 w1)=1.5 w1^2
    assert 4 * w[2] ** 2 == pytest.approx(1.5 * w[0] ** 2, rel=1e-7)


def test_paritet_perepyutannyj_znak_menyaet_vesa():
    S = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.2], [0.3, 0.2, 1.0]])
    w = A.paritet(S)
    D = np.diag([1.0, 1.0, -1.0])
    w2 = A.paritet(D @ S @ D)
    assert not np.allclose(w, w2, atol=1e-6)
    rc = _vkl(w2, D @ S @ D)
    assert np.allclose(rc, rc.mean(), atol=1e-8 * rc.mean())


def test_paritet_nulevoj_diagonali_otkaz():
    with pytest.raises(ValueError):
        A.paritet(np.diag([1.0, 0.0]))
    with pytest.raises(ValueError):
        A.paritet(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_ravnyj_risk_obratna_sigma():
    w = A.ravnyj_risk(np.array([1.0, 2.0, 4.0]))
    assert np.allclose(w, np.array([4.0, 2.0, 1.0]) / 7.0, atol=1e-15)
    with pytest.raises(ValueError):
        A.ravnyj_risk(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        A.ravnyj_risk(np.array([1.0, -1.0]))


def _ex(T=300, S=4, seed=11):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((T, 1)) * 0.003
    return base + rng.standard_normal((T, S)) * np.array([0.004, 0.006, 0.01, 0.008])


def test_vesa_dnya_ispolzuet_tolko_stroki_men_she_i():
    ex = _ex()
    i = 280
    w0, f0 = A.vesa_dnya(ex, i)
    ex2 = ex.copy()
    ex2[i:] += 100.0  # день d и будущее
    w1, f1 = A.vesa_dnya(ex2, i)
    assert np.array_equal(w0, w1) and f0 == f1 == "ok"
    ex3 = ex.copy()
    ex3[i - 1] += 0.05  # последний допустимый день - влияет
    w2, _ = A.vesa_dnya(ex3, i)
    assert not np.allclose(w0, w2, atol=1e-6)
    ex4 = ex.copy()
    ex4[i - 252 - 1] += 0.05  # за окном - не влияет
    assert np.array_equal(A.vesa_dnya(ex4, i)[0], w0)


def _mutant_okno_s_dnem_d(ex, i, okno=252):
    sig = A.lw(ex[i - okno + 1 : i + 1])
    return A.paritet(sig)


def test_mutaciya_okno_vklyuchaet_den_d_lovitsya():
    ex = _ex()
    i = 280
    ex2 = ex.copy()
    ex2[i] += 0.5
    ok = np.array_equal(A.vesa_dnya(ex, i)[0], A.vesa_dnya(ex2, i)[0])
    mut = np.array_equal(_mutant_okno_s_dnem_d(ex, i), _mutant_okno_s_dnem_d(ex2, i))
    assert ok is True
    assert mut is False  # критерий независимости от дня d ловит мутант


def test_mutaciya_okno_sdvig_na_den_vpered_lovitsya():
    ex = _ex()
    i = 280
    ref = A.vesa_dnya(ex, i)[0]
    mut = A.paritet(A.lw(ex[i - 252 + 1 : i + 1]))
    assert not np.allclose(ref, mut, atol=1e-9)
    assert np.allclose(ref, A.paritet(A.lw(ex[i - 252 : i])), atol=1e-12)


def test_vesa_dnya_koroche_okna_ravnyj_risk_s_flagom_bez_nan():
    ex = _ex()
    w, f = A.vesa_dnya(ex, 100)
    assert f == "korotkoe_okno"
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)
    sig = ex[:100].std(0)
    assert np.allclose(w, (1 / sig) / (1 / sig).sum(), atol=1e-12)


def test_vesa_dnya_rezhim_ravnyj_risk():
    ex = _ex()
    w, f = A.vesa_dnya(ex, 280, rezhim="ravnyj_risk")
    sig = ex[280 - 252 : 280].std(0)
    assert f == "ok"
    assert np.allclose(w, (1 / sig) / (1 / sig).sum(), atol=1e-12)
    with pytest.raises(ValueError):
        A.vesa_dnya(ex, 280, rezhim="max_sharpe")


def test_vesa_dnya_pustaya_lenta_i_i_men_she_2():
    w, f = A.vesa_dnya(np.zeros((0, 3)), 0)
    assert np.allclose(w, 1 / 3) and f == "korotkoe_okno"
    w, f = A.vesa_dnya(_ex(), 1)
    assert np.allclose(w, 0.25) and f == "korotkoe_okno"
    with pytest.raises(ValueError):
        A.vesa_dnya(_ex(T=10), 11)
    with pytest.raises(ValueError):
        A.vesa_dnya(_ex(), -1)


def test_vesa_dnya_stolbec_nan_ischlyuchen_bez_nan_v_vesah():
    ex = _ex()
    ex[100:105, 2] = np.nan
    w, f = A.vesa_dnya(ex, 280)
    assert w[2] == 0.0 and f == "isklyucheny"
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)
    assert np.allclose(w[[0, 1, 3]], A.paritet(A.lw(ex[28:280][:, [0, 1, 3]])), atol=1e-12)


def test_vesa_dnya_nulevaya_sigma_ischlyuchena():
    ex = _ex()
    ex[:, 1] = 0.0
    w, f = A.vesa_dnya(ex, 280)
    assert w[1] == 0.0 and f == "isklyucheny"
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)
    ex[:, :] = 0.0
    w, f = A.vesa_dnya(ex, 280)
    assert f == "net_validnyh" and np.all(w == 0.0)


def test_vesa_dnya_vklady_ravny_na_okne():
    ex = _ex()
    i = 290
    w, _ = A.vesa_dnya(ex, i)
    S = A.lw(ex[i - 252 : i])
    rc = _vkl(w, S)
    assert np.allclose(rc, rc.mean(), atol=1e-8 * rc.mean())
