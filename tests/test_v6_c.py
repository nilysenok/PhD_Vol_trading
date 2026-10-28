"""V6-c (Ж-111 пп. 4, 12, 15): ФДР t0d, ранги силы (пара / общий / среднее), размер ρ/(k_s·σ)·m(score).
Синтетика (известные ответы + подложенные ошибки) и S1 на данных ≤ 2019 (FASTBT_BARS1M=…/data/regen/D25/bars1m).
python3 -m pytest tests/test_v6_c.py -q --tb=short"""
import numpy as np
import pytest

from fastbt import fdr as F, rangi_v6 as R
from fastbt.exec_e_types import Trade


# ---- ФДР t0d: известные ответы ----
def test_zanyatost_t0_t1_eto_noch():
    n, e, c = np.array([0.0, 1.0]), np.array([0.7, 0.2]), np.array([1.0, 3.0])
    for r in ("t0", "t1"):
        assert np.array_equal(F.zanyatost(n, e, c, r), n)


def test_zanyatost_t0d_vnutridnevnaya_zanimaet_tolko_den():
    u = F.zanyatost([0.0], [1.0], [1.0], "t0d")                  # ночь свободна, весь день в рынке
    assert np.isclose(u[0], F.SESSIYA)
    assert np.isclose(F.zanyatost([1.0], [1.0], [3.0], "t0d")[0], 1.0)            # сквозная позиция: занято всё
    assert np.isclose(F.zanyatost([1.0], [0.0], [3.0], "t0d")[0], (3 - F.SESSIYA) / 3)   # занята ночь, день свободен: выходные — ночь
    assert np.isclose(F.zanyatost([0.0], [0.5], [3.0], "t0d")[0], 0.5 * F.SESSIYA / 3)


def test_zanyatost_t0d_cal_nol_ne_nan():
    assert F.zanyatost([0.4], [0.9], [0.0], "t0d").tolist() == [0.4]


def test_t0d_noch_kak_t0_i_pos_kak_t0():
    tr = [Trade("A", 1, 2, 0, 100.0, exit_day=4, side=1), Trade("A", 5, 6, 0, 100.0, exit_day=6, side=1)]
    a, b = F.noch_iz_sdelok(tr, {1: 1.0, 5: 1.0}, 8, "t0"), F.noch_iz_sdelok(tr, {1: 1.0, 5: 1.0}, 8, "t0d")
    assert np.array_equal(a, b) and b[6] == 0.0                               # внутридневная ночь не занимает
    pos = np.array([0.5, -0.2, 0.0])
    assert np.array_equal(F.noch_iz_pos(pos, np.zeros(3, bool), "t0d"), np.abs(pos))


def test_mutatsiya_zanyatost_perepytany_noch_i_den_lovitsya(monkeypatch):
    monkeypatch.setattr(F, "zanyatost", lambda n, e, c, r: np.asarray(e, float))
    with pytest.raises(AssertionError):
        assert np.isclose(F.zanyatost([1.0], [0.0], [3.0], "t0d")[0], (3 - F.SESSIYA) / 3)


# ---- ранги ----
def ryad(vals, n=300):
    m = np.full((1, n), np.nan)
    m[0, : len(vals)] = vals
    return m


def test_rang_tikera_strogo_men_she_i_bez_tekushchego_dnya():
    m = ryad(np.r_[np.arange(1.0, 30.0), 100.0])                  # день 29 — максимум, пул = 29 прошлых значений
    r = R.rang_tikera(m)
    assert r[0, 29] == 1.0 and r[0, 28] == 28 / 28 and np.isnan(r[0, 5])         # < 20 наблюдений → nan
    m2 = ryad(np.r_[np.arange(1.0, 30.0), 15.0])                   # 15 строго больше 14 значений из 29
    assert np.isclose(R.rang_tikera(m2)[0, 29], 14 / 29)
    m3 = ryad(np.r_[np.full(25, 2.0), 2.0])                         # ничья не считается «меньше»
    assert R.rang_tikera(m3)[0, 25] == 0.0


def test_rang_ne_zavisit_ot_budushchego_i_tekushchego():
    base = ryad(np.r_[np.arange(1.0, 40.0)])
    r0 = R.rang_tikera(base)[0, 35]
    f = base.copy()
    f[0, 36:39] = 1e9                                              # будущее
    assert R.rang_tikera(f)[0, 35] == r0
    c = base.copy()
    c[0, 35] = 1e9                                                 # само значение дня d не двигает свой ранг (рост — всё ещё максимум)
    assert R.rang_tikera(c)[0, 35] == 1.0


def chk_rang_d_minus_1():
    m = ryad(np.arange(1.0, 40.0))                                  # в день 35 значение 36, пул d−1 = 1…35 → ровно 1,0
    assert R.rang_tikera(m)[0, 35] == 1.0
    assert R.rang_tikera(ryad(np.full(26, 2.0)))[0, 25] == 0.0      # ничья — не «меньше»


def test_rang_d_minus_1_i_strogo():
    chk_rang_d_minus_1()


def test_mutatsiya_rang_s_tekushchim_dnem_i_nestrogo_lovyatsya(monkeypatch):
    def mutant(side, shift):
        def f(m, pul, okno, mn):
            m = np.asarray(m, float)
            out = np.full(m.shape, np.nan)
            for d in range(m.shape[1]):
                v = np.sort(m[0, max(0, d - okno):d + shift][np.isfinite(m[0, max(0, d - okno):d + shift])])
                if len(v) >= mn and np.isfinite(m[0, d]):
                    out[0, d] = np.searchsorted(v, m[0, d], side=side) / len(v)
            return out
        return f
    for side, shift in (("left", 1), ("right", 0)):                 # окно с текущим днём; «≤» вместо «<»
        monkeypatch.setattr(R, "_rang", mutant(side, shift))
        with pytest.raises(AssertionError):
            chk_rang_d_minus_1()


def test_rang_obshchiy_pul_po_tikeram_i_srednee():
    m = np.full((2, 300), np.nan)
    m[0, :25] = np.arange(1.0, 26.0)                                 # тикер 0: малые
    m[1, :25] = np.arange(101.0, 126.0)                              # тикер 1: большие
    m[0, 25], m[1, 25] = 12.0, 112.0
    t, o, s = R.rang(m, "tikernyj"), R.rang(m, "obshchiy"), R.rang(m, "srednee")
    assert np.isclose(t[0, 25], 11 / 25) and np.isclose(t[1, 25], 11 / 25)
    assert np.isclose(o[0, 25], 11 / 50) and np.isclose(o[1, 25], 36 / 50)       # пул из 50 прошлых значений
    assert np.isclose(s[0, 25], (11 / 25 + 11 / 50) / 2) and np.isnan(s[0, 24 + 40])
    with pytest.raises(ValueError):
        R.rang(m, "drugoy")


def test_rang_pust_i_odin_nabl_nan():
    assert np.isnan(R.rang(np.full((2, 50), np.nan), "srednee")).all()
    assert R.rang(np.full((2, 0), np.nan), "obshchiy").shape == (2, 0)


# ---- m(score) и вес ----
def test_mnozhitel_znacheniya_i_nan():
    r = np.array([0.0, 0.5, 1.0, np.nan])
    assert np.allclose(R.mnozhitel(r, "lin")[:3], [0.5, 1.0, 1.5]) and np.isnan(R.mnozhitel(r, "lin")[3])
    assert np.allclose(R.mnozhitel(r, "kvadr")[:3], [0.25, 0.25 + 1.75 * 0.25, 2.0])
    assert np.array_equal(R.mnozhitel(r, "1"), np.ones(4))
    with pytest.raises(ValueError):
        R.mnozhitel(r, "x")


def chk_ves():
    s = np.full(10, 0.01)
    assert np.allclose(R.ves(s, 0.5, 1.0, np.ones(10))[:9], 0.5)                  # 0,5 % / (1·1 %) = 0,5
    assert np.allclose(R.ves(s, 1.0, 2.0, np.ones(10))[:9], 0.5)                  # k_s = 2 вдвое уменьшает
    assert np.allclose(R.ves(s, 0.5, 1.0, np.full(10, 1.5))[:9], 0.75)            # × m
    assert np.allclose(R.ves(s, 1.0, 1.0, np.full(10, 2.0))[:9], 1.0)             # 2,0 → потолок 1
    assert R.ves(s, 0.5, 1.0, np.full(10, np.nan)).tolist() == [0.0] * 10          # нет ранга → вес 0
    z = R.ves(np.r_[0.01, 0.0, 0.01, 0.01], 0.5, 1.0, np.ones(4))
    assert z[0] == 0.5 and z[1] == 0.0 and np.isfinite(z).all()                    # σ[d] = 0 → вес 0, без деления на нуль


def test_ves_formula_potolok_i_nuli():
    chk_ves()
    with pytest.raises(ValueError):
        R.ves(np.full(10, 0.01), 0.0, 1.0, np.ones(10))


def test_mutatsiya_ves_bez_k_s_lovitsya(monkeypatch):
    monkeypatch.setattr(R, "ves", lambda sg, rho, ks, m: np.minimum(1.0, rho / 100.0 / np.asarray(sg, float) * np.asarray(m)))
    with pytest.raises(AssertionError):
        chk_ves()


# ---- оценщик (S1, данные ≤ 2019) ----
PT = {"vhod": 0.5, "k_s": 2.0, "R": "net", "tau": 3}


@pytest.fixture(scope="module")
def ev():
    from test_etap1_eval_v2 import make
    return make("S1")


def rows(ev, **nov):
    from test_etap1_eval_v2 import rows_of
    return rows_of(ev, PT, nov)


def g(rs):
    return np.concatenate([r.g for r in rs])


def test_rho_linejno_po_vesu(ev):
    a, b = rows(ev, rho=0.2), rows(ev, rho=0.4)
    assert np.abs(g(a)).sum() > 0 and np.allclose(2 * g(a), g(b), atol=1e-12)


def test_m_tip_ranzhirovanie_menyaet_ves_i_otsekaet_bez_ranga(ev):
    one = rows(ev, rho=0.3, m_tip="1")
    lin = rows(ev, rho=0.3, m_tip="lin", rang_tip="obshchiy")
    assert not np.allclose(g(one), g(lin))
    assert sum(r.cnt.sum() for r in lin) <= sum(r.cnt.sum() for r in one)          # кандидат без ранга не входит
    tik = rows(ev, rho=0.3, m_tip="lin", rang_tip="tikernyj")
    assert not np.allclose(g(tik), g(lin))                                         # ось rang_tip действует


def test_rho_trebuet_k_s_i_edinogo_razmera(ev):
    from test_etap1_eval_v2 import rows_of
    with pytest.raises(ValueError):
        rows_of(ev, {**PT, "k_s": "net"}, {"rho": 0.3})
    with pytest.raises(ValueError):
        rows_of(ev, {**PT, "razmer": 20}, {"rho": 0.3})


def test_novye_polya_po_umolchaniyu_vyklyucheny():
    from fastbt import etap1_eval_v2 as v2, fast_etap1
    assert v2.NOVYE["rho"] == "net" and v2.NOVYE["m_tip"] == "1"
    pt = {**PT, "vhod": 0.5}
    assert fast_etap1.pokryto(pt) and not fast_etap1.pokryto({**pt, "rho": 0.3})
    assert not fast_etap1.pokryto({**pt, "m_tip": "lin"})


def test_t0d_yakor_kak_t0_a_seteka_otlichaetsya(ev):
    from fastbt import etap1, etap1_eval_v2 as v2
    inp, base = ev.inp, etap1.base_point("S1")
    e0, e1 = v2.EvaluatorV2("S1", inp, base, 0.0004, fdr="t0"), v2.EvaluatorV2.novaya_setka("S1", inp, base)
    assert e1.fdr == "t0d"
    xa0, xa1 = e0.evaluate_v2({}, "train")[1], e1.evaluate_v2({}, "train")[1]
    assert np.array_equal(xa0, xa1)                                                  # якорь не тронут
    pt = {"vhod": 0.5, "k_s": 2.0, "R": "net", "tau": 3}
    x0, x1 = e0.evaluate_v2(pt, "train")[1], e1.evaluate_v2(pt, "train")[1]
    assert not np.array_equal(x0, x1)                                                # внутридневная занятость учтена
