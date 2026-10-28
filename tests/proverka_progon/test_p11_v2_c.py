"""П-11 (проверяющий), V6-c: ФДР t0, ранги силы, m(score), размер. Продолжение test_p11_v2.py (допущения А3 — в его шапке). ДО кода V6-c: красные.
Мутации: M10 ФДР t0 даёт ставку занятому капиталу (test_fdr_t0_*); M11 ранг по всему периоду (test_rang_*); M12 m(score) вне [0,25; 2] (test_m_score_diapazon).
Запуск: python3 -m pytest tests/proverka_progon/test_p11_v2_c.py -q --tb=short"""
import numpy as np
import pytest

from fastbt.fdr import fdr_t0
from fastbt.rang_sily import m_score, rang_obshchiy, rang_para, razmer


def test_fdr_t0_znanie_otveta():                            # M10
    r = fdr_t0(np.array([1.0, 0.5, 0.0]), np.array([0.2, 0.0, 1.0]), np.array([2e-4] * 3), np.array([1e-4] * 3))
    assert np.allclose(r, [0.0 + 0.8e-4, 1e-4 + 1e-4, 2e-4 + 0.0])


def test_fdr_t0_zanyatomu_nol_i_granitsy():
    assert np.all(fdr_t0(np.ones(4), np.ones(4), np.full(4, 2e-4), np.full(4, 1e-4)) == 0.0)   # занятому — ни ночью, ни днём
    assert fdr_t0(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)).shape == (0,)
    with pytest.raises(ValueError):
        fdr_t0(np.array([1.2]), np.array([0.0]), np.array([2e-4]), np.array([1e-4]))        # занятость > 1


def _brute(S, d, i, w=252):
    return np.mean(S[:, d - w:d] < S[i, d])


def test_rang_para_do_d_minus_1():                          # M11
    rng = np.random.default_rng(11)
    s = rng.normal(size=700)
    r = rang_para(s)
    for d in (300, 500, 699):
        assert r[d] == pytest.approx(_brute(s[None], d, 0))
    assert np.all(rang_para(np.arange(700.0))[260:] == 1.0) and np.all(rang_para(-np.arange(700.0))[260:] == 0.0)
    s2 = s.copy()
    s2[301:] += 5.0                                           # будущее не меняет ранг дня 300
    s2[:40] -= 9.0                                            # вне окна [48, 299] — тоже
    assert rang_para(s2)[300] == r[300]


def test_rang_obshchiy_puluet_tikery():
    rng = np.random.default_rng(12)
    S = rng.normal(size=(3, 600))
    r = rang_obshchiy(S)
    for i in range(3):
        assert r[i, 400] == pytest.approx(_brute(S, 400, i))
    S2 = S.copy()
    S2[:, 401:] += 5.0
    assert np.array_equal(rang_obshchiy(S2)[:, 400], r[:, 400])
    assert r[0, 400] != pytest.approx(rang_para(S[0])[400])  # общий ≠ тикерный


def test_m_score_diapazon():                                # M12
    rg = np.array([0.0, 0.5, 1.0])
    assert np.allclose(m_score(rg, "net"), 1.0)
    assert np.allclose(m_score(rg, "lin_05_15"), [0.5, 1.0, 1.5])
    kv = m_score(rg, "kvad_025_20")
    assert kv[0] == pytest.approx(0.25) and kv[2] == pytest.approx(2.0) and np.all(np.diff(kv) > 0)
    wide = np.array([-0.5, 1.5])
    assert np.allclose(m_score(wide, "lin_05_15"), [0.5, 1.5]) and np.allclose(m_score(wide, "kvad_025_20"), [0.25, 2.0])
    with pytest.raises(ValueError):
        m_score(rg, "ne_tip")


def test_razmer_formula_znak_nol():
    assert razmer(0.005, 2.0, 2.0, 0.02) == pytest.approx(0.25)
    assert razmer(0.005, 2.0, 2.0, 0.02, side=-1) == pytest.approx(-0.25)
    for bad in (0.0, -0.01, float("nan")):
        with pytest.raises(ValueError):
            razmer(0.005, 1.0, 2.0, bad)
