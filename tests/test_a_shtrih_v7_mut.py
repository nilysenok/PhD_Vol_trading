"""Тестировщик PR #847: свойства lw/paritet/vesa_dnya и мутации (копия модуля с правкой текста)."""

from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from fastbt import a_shtrih_v7 as A


def _mut(old: str, new: str) -> types.ModuleType:
    src = inspect.getsource(A)
    assert old in src, old
    m = types.ModuleType("mut")
    exec(compile(src.replace(old, new), "mut", "exec"), m.__dict__)
    return m


def _rv(T, N, seed=0, rho=0.0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((T, N))
    c = rho * rng.standard_normal((T, 1))
    return z * np.sqrt(1 - abs(rho)) + c if rho else z


def _prov(m):
    """Набор проверок, общий для настоящего модуля и мутантов (падает на мутанте)."""
    # delta в [0,1]: малое T при большом N и независимые активы
    for T, N, s in [(5, 30, 1), (8, 3, 2), (40, 4, 3), (500, 3, 4)]:
        assert 0.0 <= m.lw_delta(_rv(T, N, s)) <= 1.0
    for s in range(200):  # малое T: сырая интенсивность бывает > 1
        assert 0.0 <= m.lw_delta(_rv(4, 2, s)) <= 1.0
    # ковариация известного ответа: один актив = дисперсия с делением на T (ddof=0)
    x = _rv(50, 1, 5)
    assert np.isclose(m.lw(x)[0, 0], np.var(x[:, 0]), rtol=1e-12)
    # N>T: положительная определённость
    assert np.linalg.eigvalsh(m.lw(_rv(6, 20, 6, rho=0.5))).min() > 0
    # паритет при отрицательной корреляции: сумма вкладов равна, веса > 0
    S = np.array([[1.0, -0.6, 0.0], [-0.6, 4.0, 0.3], [0.0, 0.3, 9.0]])
    w = m.paritet(S)
    rc = w * (S @ w)
    assert w.min() > 0 and np.isclose(w.sum(), 1) and np.ptp(rc) < 1e-8 * rc.mean()
    assert not np.allclose(w, (1 / np.sqrt(np.diag(S))) / (1 / np.sqrt(np.diag(S))).sum(), atol=1e-3)
    # окно: ровно 252 строк строго < i
    z = _rv(400, 3, 7)
    z[:, 1] = 0.9 * z[:, 0] + 0.44 * z[:, 1]  # корреляция неоднородна: 0-1 сильная, 2 независим
    ex = z * np.array([1.0, 3.0, 9.0])
    i = 300
    wp, _ = m.vesa_dnya(ex, i)
    assert not np.allclose(wp, A.vesa_dnya(ex, i, rezhim="ravnyj_risk")[0], atol=1e-3)
    w1, f1 = m.vesa_dnya(ex, i)
    assert f1 == "ok"
    w2, _ = m.vesa_dnya(ex[:i], i)  # строки >= i не читаются
    assert np.array_equal(w1, w2)
    ex2 = ex.copy()
    ex2[i - 252] += 5.0 * np.array([1.0, -2.0, 0.5])  # первая строка окна обязана влиять
    assert not np.allclose(m.vesa_dnya(ex2, i)[0], w1)
    ex3 = ex.copy()
    ex3[i - 253] += 5.0 * np.array([1.0, -2.0, 0.5])  # строка вне окна — нет
    assert np.array_equal(m.vesa_dnya(ex3, i)[0], w1)


def test_nastoyashchij_modul_prohodit():
    _prov(A)


@pytest.mark.parametrize(
    "old,new",
    [
        ("min(1.0, max(0.0, pi[off].sum() / gam / T))", "pi[off].sum() / gam / T"),
        ("S = x.T @ x / T", "S = x.T @ x / (T - 1)"),
        ("okno: int = 252", "okno: int = 251"),
        ("w[ok] = paritet(lw(sub))", "w[ok] = ravnyj_risk(sub.std(axis=0))"),
    ],
    ids=["delta_bez_klipa", "ddof1_v_kovariacii", "okno_251", "paritet_eto_1_na_sigma"],
)
def test_mutaciya_lovitsya(old, new):
    with pytest.raises(AssertionError):
        _prov(_mut(old, new))


def test_delta_klip_nuzhen_realno():
    """Без клипа сырая интенсивность выходит за [0,1] на реальном входе (мутант не пустой)."""
    m = _mut("min(1.0, max(0.0, pi[off].sum() / gam / T))", "pi[off].sum() / gam / T")
    assert max(m.lw_delta(_rv(4, 2, s)) for s in range(200)) > 1.0


def test_delta_stremitsya_k_nulyu_pri_roste_T_dlya_korrelirovannyh():
    d = [A.lw_delta(_rv(T, 3, 11, rho=0.8)) for T in (30, 300, 3000, 30000)]
    assert d[-1] < 0.01 and d[-1] < d[1] < d[0] + 1e-9


def test_lw_N_bolshe_T_polozhitelno_opredelena_i_nevyrozhdena():
    C = A.lw(_rv(4, 25, 3))
    assert np.linalg.eigvalsh(C).min() > 0
    assert np.linalg.matrix_rank(np.cov(_rv(4, 25, 3).T)) < 25  # выборочная вырождена


def test_nan_v_okne_ne_zapolnyaetsya_stolbec_iskluchen():
    ex = _rv(300, 3, 9)
    ex[100, 1] = np.nan
    w, fl = A.vesa_dnya(ex, 290)
    assert w[1] == 0.0 and fl == "isklyucheny" and np.isfinite(w).all()
    assert np.isclose(w.sum(), 1.0)
    with pytest.raises(ValueError):
        A.lw(ex[:250])  # NaN в lw не заполняется молча


def test_konstanta_aktiv_isklyuchen_a_ne_delenie_na_nol():
    ex = _rv(300, 3, 10)
    ex[:, 2] = 0.7
    w, fl = A.vesa_dnya(ex, 290)
    assert w[2] == 0.0 and fl == "isklyucheny" and np.isfinite(w).all()
    with pytest.raises(ValueError):
        A.paritet(np.diag([1.0, 0.0]))
    ex[:, :] = 0.0
    assert A.vesa_dnya(ex, 290)[1] == "net_validnyh"


def test_lw_pri_otricatelnoj_korrelyacii_diagonal_sohranena():
    x = _rv(200, 1, 12)
    R = np.hstack([x, -0.8 * x + 0.3 * _rv(200, 1, 13)])
    C = A.lw(R)
    S = R.T @ R / 200 - np.outer(R.mean(0), R.mean(0))
    assert np.allclose(np.diag(C), np.diag(S))
    assert abs(C[0, 1]) <= abs(S[0, 1]) + 1e-15 and C[0, 1] * S[0, 1] >= 0
