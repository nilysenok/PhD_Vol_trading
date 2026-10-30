"""Л1.1 kross_v7 (Ж-146): панель скоров S7–S9, ребалансировка день/неделя, гистерезис оборота, плацебо сечения.
Тесты написаны ДО кода. Мутации в тесте: скор дня d с c[d+1] (M-d1); решение недели с данными после дня решения;
бафер не сводится к vesa при bufer=0 (выкл.); плацебо теряет число ног. Каждая подложена и обязана быть поймана."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kross_seksiya as ks
from fastbt import kross_v7 as kv

N, T = 8, 320
IDX = pd.bdate_range("2018-01-01", periods=T)
COLS = [f"T{i}" for i in range(N)]


def _ceny(seed=1):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, N)), axis=0)), index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, T))), index=IDX)
    metki = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * T, dtype=object).reshape(T, N)
    return c, m, metki


def _eq(a, b):
    return np.array_equal(np.asarray(a, float), np.asarray(b, float), equal_nan=True)


# ---- G1: известный ответ на 6 тикерах ----
def _rost():
    n, t = 6, 12
    idx = pd.bdate_range("2018-01-01", periods=t)
    g = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])      # i-й тикер растёт быстрее (i+1)-го -> ранг растёт с i
    c = pd.DataFrame(100 * (1 + g) ** np.arange(t)[:, None], index=idx, columns=list("abcdef"))
    return c, pd.Series(1000.0, index=idx), np.array(["X", "X", "X", "Y", "Y", "Y"], dtype=object)


def test_g1_skor_s7_izvestnyj_otvet():
    c, m, _ = _rost()
    sk = kv.panel(c, m, np.full(c.shape, "", dtype=object), "S7")
    assert sk.shape == c.shape
    assert np.isnan(sk[:5]).all()                                   # нет 5 дней истории
    ozhid = np.array([1, 0.6, 0.2, -0.2, -0.6, -1.0])              # разворот: i=5 растёт быстрее всех -> скор -1
    assert np.allclose(sk[7], ozhid)


def test_g1_storona_izvestnyj_otvet_i_cap():
    c, m, _ = _rost()
    sk = kv.panel(c, m, np.full(c.shape, "", dtype=object), "S7")
    t = sk.shape[0]
    metki = np.tile(np.array(["X", "X", "X", "Y", "Y", "Y"], dtype=object), (t, 1))
    sg, ok = np.ones(sk.shape), np.ones(sk.shape, bool)
    s = kv.storona(sk, sg, metki, ok)
    assert s.dtype == np.int8
    assert list(s[7]) == [1, 1, 0, 0, -1, -1]
    assert (s[:5] == 0).all()
    one = np.full(sk.shape, "Z", dtype=object)                      # весь сектор один, cap=1 -> по одному с каждой стороны
    s1 = kv.storona(sk, sg, one, ok, cap=1)
    assert np.abs(s1[7]).sum() == 2 and s1[7].max() == 1 and s1[7].min() == -1


def test_g1_histerezis_bufer():
    sk = np.array([[0.6, -0.9, 0], [0.4, -0.9, 0], [0.2, -0.9, 0], [0.4, -0.9, 0]])
    sg, ok = np.ones(sk.shape), np.ones(sk.shape, bool)
    metki = np.full(sk.shape, "A", dtype=object)
    s = kv.storona(sk, sg, metki, ok, porog=0.5, bufer=0.2)
    assert list(s[:, 0]) == [1, 1, 0, 0] and list(s[:, 1]) == [-1, -1, 0, 0]   # держим при 0,4 ≥ 0,3; на 0,2 выход
    s0 = kv.storona(sk, sg, metki, ok, porog=0.5, bufer=0.0)
    assert (s0[1:] == 0).all() and s0[0, 0] == 1                    # без бафера 0,4 < порога -> выход сразу


def test_storona_zapret_bufer():
    sk = np.zeros((2, 3))
    a = (sk, np.ones(sk.shape), np.full(sk.shape, "A", dtype=object), np.ones(sk.shape, bool))
    with pytest.raises(ValueError):
        kv.storona(*a, porog=0.5, bufer=0.5)
    with pytest.raises(ValueError):
        kv.storona(*a, bufer=-0.1)


# ---- G2: выкл. == vesa побайтно ----
def test_g2_vykl_ravno_vesa():
    r = np.random.default_rng(7)
    t, n = 200, 12
    sk = r.uniform(-1, 1, (t, n))
    sk[r.random((t, n)) < 0.1] = np.nan
    sg = r.uniform(0.5, 2, (t, n))
    sg[r.random((t, n)) < 0.05] = 0.0
    ok = r.random((t, n)) < 0.9
    metki = np.array(list("AAABBBCCCDDD") * t, dtype=object).reshape(t, n)
    for cap in (1, 2, 3):
        w = ks.vesa(sk, sg, metki, ok, 0.5, cap)
        s = kv.storona(sk, sg, metki, ok, 0.5, 0.0, cap)
        assert np.array_equal(s != 0, w != 0)
        assert np.array_equal(s, np.sign(w).astype(np.int8))
        assert (s != 0).any()


def test_g2_mutatsiya_bufer_ne_vykl_lovitsya():
    r = np.random.default_rng(8)
    t, n = 150, 10
    sk = r.uniform(-1, 1, (t, n))
    sg, ok = np.ones((t, n)), np.ones((t, n), bool)
    metki = np.array(list("AABBCCDDEE") * t, dtype=object).reshape(t, n)
    w = ks.vesa(sk, sg, metki, ok)
    mut = kv.storona(sk, sg, metki, ok, 0.5, 0.2)
    assert not np.array_equal(mut != 0, w != 0)                     # тест G2 обязан падать при bufer>0


# ---- G3: M-d1 ----
def _prichinno(fn, c, m, metki, d=270):
    """Скор до дня d включительно не меняется, если испортить цены после d."""
    a = fn(c, m, metki)
    c2, m2 = c.copy(), m.copy()
    c2.iloc[d + 1:] *= np.linspace(0.3, 3.7, c.shape[1])[None, :] ** 2    # разные множители: ранги меняются
    m2.iloc[d + 1:] *= 0.4
    b = fn(c2, m2, metki)
    return _eq(a[:d + 1], b[:d + 1])


@pytest.mark.parametrize("sid", ["S7", "S8", "S9"])
def test_g3_skor_prichinen(sid):
    c, m, metki = _ceny()
    assert _prichinno(lambda c_, m_, mk: kv.panel(c_, m_, mk, sid), c, m, metki)
    assert np.isfinite(kv.panel(c, m, metki, sid)[260:]).any()


@pytest.mark.parametrize("sid", ["S7", "S8", "S9"])
def test_g3_mutatsiya_d_plus_1_lovitsya(sid):
    c, m, metki = _ceny()
    # скор дня d берёт будущее: у S8 последние 21 день окна не используются, поэтому для неё опережение 25 дней
    k = -25 if sid == "S8" else -1
    mut = lambda c_, m_, mk: kv.panel(c_.shift(k), m_.shift(k), mk, sid)   # noqa: E731
    assert not _prichinno(mut, c, m, metki)


def test_panel_raznye_sid_i_oshibki():
    c, m, metki = _ceny()
    assert _eq(kv.panel(c, m, metki, "S7"), ks.skor_s7(c, m).to_numpy())
    assert _eq(kv.panel(c, m, metki, "S8"), ks.skor_s8(c).to_numpy())
    assert _eq(kv.panel(c, m, metki, "S9"), ks.skor_s9(c, m, metki).to_numpy())
    with pytest.raises(ValueError):
        kv.panel(c, m, metki, "S1")
    assert np.isnan(kv.panel(c.iloc[:, :4], m, metki[:, :4], "S7")).all()    # меньше MIN_TIK тикеров -> входа нет


# ---- G4: неделя ----
def _nedeli():
    r = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=30)                  # 6 полных недель, пн-пт
    return r.uniform(-1, 1, (30, 5)), idx


def test_g4_nedelya_ves_postoyanen_i_znachenie():
    sk, idx = _nedeli()
    out = kv.rebal(sk, idx, "nedelya")
    assert out.shape == sk.shape
    for k in range(1, 30):
        pt = k // 5 * 5 - 1                                          # пятница предыдущей недели
        if k % 5 == 4:
            assert np.array_equal(out[k], sk[k])                     # в последний день недели — своё решение
        elif pt >= 0:
            assert np.array_equal(out[k], sk[pt])
        else:
            assert np.isnan(out[k]).all()                            # до первого решения решения нет
    assert np.array_equal(out[5:9], np.tile(sk[4], (4, 1)))          # пн-чт недели 2 держат пятницу недели 1


def test_g4_nedelya_ne_ispolzuet_budushchee():
    sk, idx = _nedeli()
    a = kv.rebal(sk, idx, "nedelya")
    for d in (7, 12, 19):
        sk2 = sk.copy()
        sk2[d + 1:] = 99.0
        b = kv.rebal(sk2, idx, "nedelya")
        assert _eq(a[:d + 1], b[:d + 1])


def test_g4_mutatsiya_nedelya_s_budushchim_lovitsya():
    sk, idx = _nedeli()
    wk = pd.Series(np.arange(30) // 5)
    mut = pd.DataFrame(sk).groupby(wk.values).transform("last").to_numpy()      # вся неделя берёт пятничное решение
    a = mut
    sk2 = sk.copy()
    sk2[8 + 1:] = 99.0
    b = pd.DataFrame(sk2).groupby(wk.values).transform("last").to_numpy()
    assert not _eq(a[:9], b[:9])                                      # мутация видна проверкой причинности


def test_g4_den_tozhdestvenno_i_oshibki():
    sk, idx = _nedeli()
    assert _eq(kv.rebal(sk, idx, "den"), sk)
    with pytest.raises(ValueError):
        kv.rebal(sk, idx, "mesyac")
    with pytest.raises(ValueError):
        kv.rebal(sk, idx[:-1], "nedelya")
    assert kv.rebal(np.zeros((0, 3)), pd.DatetimeIndex([]), "nedelya").shape == (0, 3)     # пустая лента


def test_g4_nedelya_dyra_v_dnyakh():
    idx = pd.DatetimeIndex(["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-09", "2018-01-10", "2018-01-12"])
    sk = np.arange(6, dtype=float)[:, None] * np.ones((1, 2))
    out = kv.rebal(sk, idx, "nedelya")            # недели: [1,2,3] (посл. 3), [9,10,12] (посл. 12)
    assert np.isnan(out[:2]).all() and out[2, 0] == 2.0               # 3 янв. — последний торговый день недели
    assert (out[3:5, 0] == 2.0).all() and out[5, 0] == 5.0


# ---- G5: плацебо ----
def test_g5_placebo_sohranyaet_chislo_nog():
    r = np.random.default_rng(5)
    sk = r.uniform(-1, 1, (60, 12))
    sk[r.random(sk.shape) < 0.1] = np.nan
    p = kv.placebo_sechenie(sk, seed=11)
    assert p.shape == sk.shape
    for sg in (1, -1):
        a = np.nansum(sg * sk >= 0.5, axis=1)
        b = np.nansum(sg * p >= 0.5, axis=1)
        assert np.array_equal(a, b)
    assert np.array_equal(np.sort(sk, axis=1), np.sort(p, axis=1), equal_nan=True)     # множество значений дня цело (NaN в конце)
    assert not _eq(sk, p)                                              # перестановка не тождественна
    assert (np.isnan(sk).sum(1) == np.isnan(p).sum(1)).all()


def test_g5_placebo_seed_i_nezavisimost_dnej():
    sk = np.tile(np.linspace(-1, 1, 10), (40, 1))
    a, b, c = (kv.placebo_sechenie(sk, seed=s) for s in (1, 1, 2))
    assert _eq(a, b) and not _eq(a, c)
    assert len({tuple(row) for row in a}) > 20                         # каждый день — своя перестановка
    assert np.array_equal(sk, np.tile(np.linspace(-1, 1, 10), (40, 1)))   # вход не изменён
    assert kv.placebo_sechenie(np.zeros((0, 4)), seed=1).shape == (0, 4)


# ---- G6: к2 ----
def test_g6_k2_ravno_k2_skor_i_lag0():
    c, m, metki = _ceny()
    sk = kv.panel(c, m, metki, "S8")
    for lag in (0, 63, 100):
        assert _eq(kv.k2(sk, lag), ks.k2_skor(sk, lag))
    assert _eq(kv.k2(sk, 0), sk)                                       # mutations.md № 9
    assert not _eq(kv.k2(sk, 63), sk)
