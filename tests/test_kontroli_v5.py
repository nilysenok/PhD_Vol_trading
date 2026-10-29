"""Контроли процедуры v5 (лист П9): fastbt/kontroli_v5.py поверх kalibr_v5.Pul, синтетика, данных нет.
Каждая мутация подложена в самом тесте и обязана ловиться:
  лаг 0 ≠ исходный (№ 9); к3 оставляет не ту ось (№ 10); окно с годом t; лаг < 21; плато → пик (№ 27).
python3 -m pytest tests/test_kontroli_v5.py -q
"""
import itertools

import numpy as np
import pytest
from scipy import stats

from fastbt import fast_kontrol as fk
from fastbt import kalibr_v5 as K
from fastbt import kontroli_v5 as kv
from fastbt.data import year_bounds

_D = np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]")
DATY = _D[np.is_busday(_D)]
NT = 2


def _pul(seed=3, **kw):
    return kv.sintez_pul(seed, DATY, **kw)


def _dni(pul, gody=kv.GODY):
    return np.flatnonzero(np.isin(kv._god(pul), gody))


def _oracle(blok=60):
    """Известный ответ: точка 0 растёт в режиме +1 и падает в −1; оракул держит её только в режиме +1."""
    pul = _pul(amp=0.0, m=4)
    dni = _dni(pul)
    s = np.where(np.sign(np.sin(np.arange(len(dni)) // blok * 2.3)) > 0, 1.0, -1.0)
    ex = pul.ex.copy()
    ex[0][:, dni] += 0.004 * s
    w = np.zeros((len(dni), NT, ex.shape[0]))
    w[:, :, 0] = (s > 0)[:, None]
    return kv.podpul_ex(pul, ex), kv.Vyhod(dni, w)


def _rand_proc(seed, blok=20):
    def proc(pul, gody):
        dni = _dni(pul, gody)
        rng = np.random.default_rng(seed)
        w = np.zeros((len(dni), NT, pul.ex.shape[0]))
        for ti in range(NT):
            for a in range(0, len(dni), blok):
                w[a:a + blok, ti, rng.integers(pul.ex.shape[0])] = 1.0
        return kv.Vyhod(dni, w)
    return proc


# ---------------- к2 ----------------

def test_k2_lag0_gives_original():
    pul, v = _oracle()
    zero = np.zeros((1, NT), int)
    assert np.array_equal(kv.sdvig_vesov(v.w, zero[0]), v.w)
    base = kv.k2(pul, v, lagi=zero)
    assert base["perm"][0] == base["sharpe"]
    assert kv.k2(pul, v, lagi=np.full((1, NT), 77))["perm"][0] != base["sharpe"]   # «лаг 0 ≠ тождество» ловится


def test_k2_shift_keeps_weights_and_runs():
    _, v = _oracle()
    sh = kv.sdvig_vesov(v.w, np.array([30, 100]))
    for ti in range(NT):
        a, b = v.w[:, ti, 0], sh[:, ti, 0]
        assert sorted(a) == sorted(b)                                # доля дней и уровни
        assert np.sum(a != np.roll(a, 1)) == np.sum(b != np.roll(b, 1))   # оборот (круг)


def test_p_perm_uniform_for_independent_weights():
    ps = [kv.k2(p, _rand_proc(s)(p, kv.GODY), n_shift=59)["p"]
          for s in range(40) for p in [_pul(amp=0.0, m=4, rseed=100 + s)]]
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert np.mean(np.array(ps) < 0.05) <= 0.15


def test_p_perm_known_answer_and_sign_flip():
    pul, v = _oracle()
    assert kv.k2(pul, v, n_shift=99)["p"] < 0.05                     # оракул режима — p мал
    flip = v._replace(w=np.where(v.w > 0, 0.0, 1.0) * (np.arange(v.w.shape[2]) == 0))
    assert kv.k2(pul, flip, n_shift=99)["p"] > 0.5                   # перепутанный знак — p не мал


def test_p_perm_not_below_nominal_for_selection_under_null():
    """Выбор K.god по прошлому, чистый шум: размер критерия не выше номинала (сдвиг только консервативен)."""
    ps = np.array([kv.k2(p, kv.kalibr(p), n_shift=59)["p"] for s in range(10)
                   for p in [_pul(amp=0.0, m=6, rseed=300 + s)]])
    assert np.mean(ps < 0.05) <= 0.10


def test_mutatsiya_lag_menshe_21_lovitsya():
    pul, v = _oracle()
    n = len(v.dni)
    with pytest.raises(AssertionError):
        kv.proverit_lagi(fk.draw_lags(np.full(NT, n), 50, min_lag=1), n)
    with pytest.raises(AssertionError):
        kv.k2(pul, v, n_shift=50, min_lag=1)
    kv.proverit_lagi(fk.draw_lags(np.full(NT, n), 50), n)


# ---------------- к1, к3 ----------------

def test_k1_is_unconditional_mean_of_all_points():
    pul = _pul(amp=1.5)
    dni = _dni(pul)
    p = pul.ex.shape[0]
    v = kv.Vyhod(dni, np.full((len(dni), NT, p), 1.0 / p))
    ref = pul.ex[:, :, dni].mean((0, 1))
    assert np.allclose(kv.portfel(pul, v), ref, atol=1e-15)
    assert abs(kv.k1(pul) - kv.sharpe(ref)) < 1e-12                  # порядок суммирования — единицы 1e-16


def test_k3_zeroes_exactly_named_axis():
    pul = _pul(m=9)
    k = kv.k3(pul, ("a",), {"a": 0})
    keep = np.array([pt["a"] == 0 for pt in pul.tochki])
    assert 0 < keep.sum() < len(keep) and len(k.tochki) == keep.sum()  # убраны только точки с a ≠ якоря
    assert all(pt["a"] == 0 for pt in k.tochki) and np.array_equal(k.ex, pul.ex[keep])
    assert k.osi["a"] == ["0.0"] and k.osi["b"] == pul.osi["b"] and k.osi["c"] == pul.osi["c"]
    bad = kv.k3(pul, ("b",), {"b": 0})                               # мутация: не та ось
    assert any(pt["a"] != 0 for pt in bad.tochki) and all(pt["b"] == 0 for pt in bad.tochki)
    assert kv.kalibr(k, (2018,)).w.shape[2] == len(k.tochki)         # процедура на базе v4 работает


# ---------------- фиксированная точка, seed Соболя ----------------

def test_plato_not_peak_and_edge_excluded():
    osi = {"a": list(range(5)), "b": list(range(5)), "c": list(range(5))}
    g = list(itertools.product(range(5), repeat=3))
    pul = K.Pul("S1", DATY[:10], osi, [dict(zip("abc", c)) for c in g], np.zeros((125, 1, 10)))
    gg = np.array(g)
    sh = 2.0 * np.exp(-((gg - 1) ** 2).sum(1) / 8.0)                 # плавный холм около (1,1,1)
    spike = g.index((3, 3, 3))
    sh[spike] = 6.0                                                  # одиночный пик
    pl = kv.plato(pul, sh)
    assert np.argmax(sh) == spike and np.nanargmax(pl) != spike      # мутация «плато → пик» выбрала бы spike
    osi2 = {"a": list(range(5)), "b": list(range(5))}
    g2 = list(itertools.product(range(5), repeat=2))
    pul2 = K.Pul("S1", DATY[:10], osi2, [dict(zip("ab", c)) for c in g2], np.zeros((25, 1, 10)))
    assert np.isnan(kv.plato(pul2, np.ones(25))[0])                  # угол 2D — 2 соседа: край


def test_fiksirovannaya_finds_plateau_on_other_years():
    pul = _pul(amp=4.0, m=8)
    f = kv.fiksirovannaya(pul, (2016, 2017), (2018, 2019))
    t = pul.tochki[f["tochka"]]
    assert abs(t["a"] - 2) + abs(t["b"] - 2) + abs(t["c"] - 1.5) <= 3.5 and f["sharpe"] > 0.5


def test_seed2_reports_per_year_and_seeds_differ():
    a, b = _pul(seed=1, amp=2.0, m=6), _pul(seed=2, amp=2.0, m=6)
    assert a.tochki != b.tochki                                      # выборки Соболя разные
    out = kv.seed2(kv.kalibr, a, b, (2018, 2019))
    assert set(out) == {2018, 2019} and all(np.isfinite(x["rashozhdenie"]) for x in out.values())
    assert all(x["rashozhdenie"] == 0 for x in kv.seed2(kv.kalibr, a, a, (2018, 2019)).values())


# ---------------- причинность ----------------

def test_mutatsiya_okno_s_godom_t_lovitsya():
    pul = _pul(amp=1.5, m=6)
    kv.prichinnost(kv.kalibr, pul, (2018, 2019))                     # боевой калибратор проходит

    def bad(p, gody):                                                # окно ≤ конца года t вместо ≤ 31.12(t−1)
        s, e = year_bounds(p.daty, gody[0])
        w = np.zeros((e - s, NT, p.ex.shape[0]))
        w[:, np.arange(NT), K.sharp_okna(p.ex[:, :, :e]).argmax(0)] = 1.0
        return kv.Vyhod(np.arange(s, e), w)
    with pytest.raises(AssertionError):
        kv.prichinnost(bad, pul, (2018, 2019))
