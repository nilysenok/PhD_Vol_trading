"""П-11 (проверяющий), V6-c: ФДР t0d, ранги силы, m(score), размер. Продолжение test_p11_v2.py. Подогнано под код V6-c (#504) тестировщиком П-11б (Ж-081).
Мутации: M10 ФДР t0d даёт ставку занятому капиталу (test_fdr_t0d_*); M11 ранг по всему периоду (test_rang_*); M12 m(score) вне [0,25; 2] (test_m_score_diapazon).
Причины правок против версии до кода (допущение А3 было неверным):
 - `fdr_t0(noch, zan_den, rf_noch, rf_den)` -> `fdr.zanyatost(noch, expo, cal, "t0d")` = занятость u; ставка строки = (1 − u)·rf·cal. Известный ответ считается тут же
   как (1−noch)·rf·(cal−S) + (1−expo)·rf·S, S = SESSIYA = 530/1440 (разложение на ночь и сессию) — независимо от формулы кода.
 - `rang_sily.rang_para/rang_obshchiy` -> `rangi_v6.rang_tikera/rang_obshchiy`; `m_score(rang, tip)` -> `mnozhitel(rang, tip)`, типы net/lin_05_15/kvad_025_20 -> "1"/"lin"/"kvadr".
 - `razmer(rho, m, k_s, sigma, side)` -> `ves(sigma, rho_pct, k_s, m)`: вес без знака (знак стороны вешает исполнение), потолок 1, σ ≤ 0 / nan -> вес 0 (а не ValueError).
 - m(score) вне [0,25; 2] при подаче ранга вне [0; 1]: код границы не режет (ранг по построению в [0; 1]); проверка вынесена на реальные ранги из rang() — тест M12 ловит ошибку в ранге.
Запуск: python3 -m pytest tests/proverka_progon/test_p11_v2_c.py -q --tb=short"""
import numpy as np
import pytest

from fastbt.fdr import SESSIYA, zanyatost
from fastbt.rangi_v6 import mnozhitel, rang, rang_obshchiy, rang_tikera, ves


def stavka(noch, expo, cal, rf):
    return (1 - zanyatost(np.asarray(noch, float), np.asarray(expo, float), np.asarray(cal, float), "t0d")) * rf * np.asarray(cal, float)


def test_fdr_t0d_znanie_otveta():                           # M10
    noch, expo, cal, rf = np.array([1.0, 0.5, 0.0]), np.array([0.2, 0.0, 1.0]), np.array([1.0, 3.0, 1.0]), 1e-4
    ans = (1 - noch) * rf * (cal - SESSIYA) + (1 - expo) * rf * SESSIYA
    assert np.allclose(stavka(noch, expo, cal, rf), ans)
    assert ans[0] == pytest.approx(0.8 * rf * SESSIYA)          # ночь занята целиком, свободно 80 % сессии


def test_fdr_t0d_zanyatomu_nol_i_granitsy():
    assert np.allclose(stavka(np.ones(4), np.ones(4), np.ones(4), 1e-4), 0.0)       # занятому — ни ночью, ни днём
    assert stavka(np.zeros(4), np.zeros(4), np.ones(4), 1e-4) == pytest.approx(np.full(4, 1e-4))   # свободному — полная ставка
    assert zanyatost(np.zeros(0), np.zeros(0), np.zeros(0), "t0d").shape == (0,)
    assert np.isfinite(zanyatost(np.array([0.5]), np.array([1.0]), np.array([0.0]), "t0d")).all()   # cal = 0 без деления на ноль
    with pytest.raises(ValueError):
        zanyatost(np.array([1.2]), np.array([0.0]), np.array([1.0]), "t0d")                       # занятость > 1


def test_fdr_t0d_otlichaetsya_ot_t0_po_dnyu():              # M10: режим не сводится к ночной занятости
    noch, expo, cal = np.array([1.0]), np.array([0.0]), np.array([1.0])
    assert zanyatost(noch, expo, cal, "t0")[0] == 1.0 and zanyatost(noch, expo, cal, "t0d")[0] == pytest.approx(1 - SESSIYA)


def _brute(S, d, i, w=252):
    return np.mean(S[:, d - w:d] < S[i, d])


def test_rang_para_do_d_minus_1():                          # M11
    rng = np.random.default_rng(11)
    s = rng.normal(size=700)
    r = rang_tikera(s[None])[0]
    for d in (300, 500, 699):
        assert r[d] == pytest.approx(_brute(s[None], d, 0))
    assert np.all(rang_tikera(np.arange(700.0)[None])[0][260:] == 1.0) and np.all(rang_tikera(-np.arange(700.0)[None])[0][260:] == 0.0)
    s2 = s.copy()
    s2[301:] += 5.0                                           # будущее не меняет ранг дня 300
    s2[:40] -= 9.0                                            # вне окна [48, 299] — тоже
    assert rang_tikera(s2[None])[0][300] == r[300]
    assert np.isnan(r[:20]).all()                              # < 20 наблюдений — рангa нет


def test_rang_obshchiy_puluet_tikery():
    rng = np.random.default_rng(12)
    S = rng.normal(size=(3, 600))
    r = rang_obshchiy(S)
    for i in range(3):
        assert r[i, 400] == pytest.approx(_brute(S, 400, i))
    S2 = S.copy()
    S2[:, 401:] += 5.0
    assert np.array_equal(rang_obshchiy(S2)[:, 400], r[:, 400])
    assert r[0, 400] != pytest.approx(rang_tikera(S[:1])[0][400])  # общий ≠ тикерный


def test_m_score_diapazon():                                # M12
    rg = np.array([0.0, 0.5, 1.0])
    assert np.allclose(mnozhitel(rg, "1"), 1.0)
    assert np.allclose(mnozhitel(rg, "lin"), [0.5, 1.0, 1.5])
    kv = mnozhitel(rg, "kvadr")
    assert kv[0] == pytest.approx(0.25) and kv[2] == pytest.approx(2.0) and np.all(np.diff(kv) > 0)
    S = np.random.default_rng(13).normal(size=(3, 500))
    for tip in ("tikernyj", "obshchiy", "srednee"):           # реальные ранги -> m строго в заявленных границах
        r = rang(S, tip)
        ok = np.isfinite(r)
        assert ok.any() and r[ok].min() >= 0.0 and r[ok].max() <= 1.0
        assert 0.5 <= mnozhitel(r[ok], "lin").min() and mnozhitel(r[ok], "lin").max() <= 1.5
        assert 0.25 <= mnozhitel(r[ok], "kvadr").min() and mnozhitel(r[ok], "kvadr").max() <= 2.0
    with pytest.raises(ValueError):
        mnozhitel(rg, "ne_tip")


def test_rang_strogo_menshe_i_ves_sigma_dnya_d():           # свои мутации: «≤» вместо «<»; σ дня d+1 в весе
    r = rang_tikera(np.ones((1, 60)))[0]
    assert np.isnan(r[:20]).all() and np.all(r[20:] == 0.0)    # ничьи не считаются «меньше»
    w = ves(np.array([0.02, 0.04, 0.08]), 0.5, 2.0, np.ones(3))
    assert np.allclose(w, [0.125, 0.0625, 0.03125])            # вес дня d по σ_d


def test_razmer_formula_nol_i_potolok():
    assert ves(np.array([0.02]), 0.5, 2.0, np.array([2.0]))[0] == pytest.approx(0.25)      # 0,005/(2·0,02)·2
    assert ves(np.array([0.001]), 0.5, 2.0, np.array([2.0]))[0] == 1.0                      # потолок веса 1
    assert np.all(ves(np.array([0.0, -0.01, np.nan]), 0.5, 1.0, np.ones(3)) == 0.0)         # σ ≤ 0 / nan -> 0
    for rho, ks in ((0.0, 1.0), (-0.1, 1.0), (0.5, 0.0), (0.5, -1.0)):
        with pytest.raises(ValueError):
            ves(np.array([0.02]), rho, ks, np.ones(1))
