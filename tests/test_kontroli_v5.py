"""Контроли процедуры v5 (лист П9): fastbt/kontroli_v5.py на синтетическом ledger, данных не читает.
Каждая мутация подложена в самом тесте и обязана ловиться:
  лаг 0 ≠ исходный (№ 9); к3 оставляет не ту ось (№ 10); окно с годом t; лаг < 21; плато → пик (№ 27).
python3 -m pytest tests/test_kontroli_v5.py -q
"""
import numpy as np
import pytest
from scipy import stats

from fastbt import fast_kontrol as fk
from fastbt import kontroli_v5 as kv

_D = np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]")
DATY = _D[np.is_busday(_D)]
GODY = (2018, 2019)
NT = 3


def _led(seed=3, **kw):
    return kv.sintez_ledger(seed, DATY, **kw)


def _oracle(blok=60):
    """Известный ответ: точка 0 растёт в режиме +1 и падает в −1; оракул держит её только в режиме +1."""
    led = _led(amp=0.0, m=4)
    god = kv._god(led)
    dni = np.flatnonzero(np.isin(god, GODY))
    s = np.where(np.sign(np.sin(np.arange(len(dni)) // blok * 2.3)) > 0, 1.0, -1.0)
    r = led.r.copy()
    r[dni, :, 0] += 0.004 * s[:, None]
    w = np.zeros((len(dni), NT, r.shape[2]))
    w[:, :, 0] = (s > 0)[:, None]
    return led._replace(r=r), kv.Vyhod(dni, w), s


def _rand_proc(seed, blok=20):
    def proc(led, gody):
        dni = np.flatnonzero(np.isin(kv._god(led), gody))
        rng = np.random.default_rng(seed)
        w = np.zeros((len(dni), NT, led.r.shape[2]))
        for ti in range(NT):
            for a in range(0, len(dni), blok):
                w[a:a + blok, ti, rng.integers(led.r.shape[2])] = 1.0
        return kv.Vyhod(dni, w)
    return proc


# ---------------- к2 ----------------

def test_k2_lag0_gives_original():
    led, v, _ = _oracle()
    zero = np.zeros((1, NT), int)
    assert np.array_equal(kv.sdvig_vesov(v.w, zero[0]), v.w)
    assert kv.k2(led, v, lagi=zero)["perm"][0] == kv.k2(led, v, lagi=zero)["sharpe"]
    one = kv.k2(led, v, lagi=np.full((1, NT), 77))["perm"][0]
    assert one != kv.k2(led, v, lagi=zero)["sharpe"]                 # мутация «лаг 0 ≠ тождество» ловится


def test_k2_shift_keeps_weights_and_runs():
    _, v, _ = _oracle()
    sh = kv.sdvig_vesov(v.w, np.array([30, 100, 200]))
    for ti in range(NT):
        a, b = v.w[:, ti, 0], sh[:, ti, 0]
        assert sorted(a) == sorted(b)                                # доля дней и уровни
        assert np.sum(a != np.roll(a, 1)) == np.sum(b != np.roll(b, 1))   # оборот (круг)


def test_p_perm_uniform_for_independent_weights():
    ps = [kv.k2(led, _rand_proc(s)(led, GODY), n_shift=59)["p"]
          for s in range(40) for led in [_led(rseed=100 + s, amp=0.0, m=4)]]
    assert stats.kstest(ps, "uniform").pvalue > 0.01
    assert np.mean(np.array(ps) < 0.05) <= 0.15


def test_p_perm_known_answer_and_sign_flip():
    led, v, s = _oracle()
    n = len(s)
    assert kv.k2(led, v, n_shift=99)["p"] < 0.05                     # оракул режима — p мал
    flip = v._replace(w=np.where(v.w > 0, 0.0, 1.0) * (np.arange(v.w.shape[2]) == 0))
    assert kv.k2(led, flip, n_shift=99)["p"] > 0.5                   # перепутанный знак — p не мал
    assert n > 2 * fk.MIN_LAG


def test_p_perm_conservative_for_selection_procedure_under_null():
    """Выбор по прошлому, шум: сдвиг весов 2019 на 2018 вносит выбор «по этим же дням» — p не меньше номинала."""
    ps = np.array([kv.k2(led, kv.prostaya_kalibr(led, GODY), n_shift=59)["p"]
                   for s in range(30) for led in [_led(rseed=300 + s, amp=0.0, m=4)]])
    assert np.mean(ps < 0.05) <= 0.10


def test_mutatsiya_lag_menshe_21_lovitsya():
    led, v, _ = _oracle()
    bad = fk.draw_lags(np.full(NT, len(v.dni)), 50, min_lag=1)
    with pytest.raises(AssertionError):
        kv.proverit_lagi(bad, len(v.dni))
    with pytest.raises(AssertionError):
        kv.k2(led, v, n_shift=50, min_lag=1)
    kv.proverit_lagi(fk.draw_lags(np.full(NT, len(v.dni)), 50), len(v.dni))


# ---------------- к1, к3 ----------------

def test_k1_is_unconditional_mean_of_all_points():
    led = _led(amp=1.5)
    v = kv.prostaya_kalibr(led, GODY, bez_vybora=True)
    assert np.allclose(v.w.sum(2), 1.0)
    god = kv._god(led)
    ref = led.r[np.isin(god, GODY)].mean((1, 2))
    assert np.allclose(kv.portfel(led, v), ref, atol=1e-15)
    assert abs(kv.k1(led, GODY) - kv.sharpe(ref)) < 1e-12              # порядок суммирования — единицы 1e-16


def test_k3_zeroes_exactly_named_axis():
    led = _led(m=7)
    k = kv.k3(led, ("a",), {"a": 0})
    full = led.idx[led.idx[:, 0] == 0]
    assert np.array_equal(k.idx, full) and len(k.idx) < len(led.idx)     # убраны только точки с a ≠ якоря
    assert np.array_equal(k.r, led.r[:, :, led.idx[:, 0] == 0])
    bad = kv.k3(led, ("b",), {"b": 0})                                 # мутация: не та ось
    assert not (bad.idx[:, 0] == 0).all() and (bad.idx[:, 1] == 0).all()
    two = kv.k3(led, ("a", "b"), {"a": 0, "b": 0})
    assert ((two.idx[:, :2] == 0).all()) and len(two.idx) <= len(k.idx)


# ---------------- фиксированная точка, seed Соболя ----------------

def test_plato_not_peak_and_edge_excluded():
    g = np.array(list(np.ndindex(5, 5, 5)))
    led = kv.Ledger(np.zeros((1, 1, len(g))), DATY[:1], g, ("a", "b", "c"))
    sh = 2.0 * np.exp(-((g - 1) ** 2).sum(1) / 8.0)                   # плавный холм около (1,1,1)
    spike = int(np.flatnonzero((g == [3, 3, 3]).all(1))[0])
    sh[spike] = 6.0                                                    # одиночный пик
    pl = kv.plato(led, sh)
    assert np.nanargmax(pl) != spike and np.argmax(sh) == spike        # плато → пик (мутация) выбрала бы spike
    assert np.nanargmax(pl) != np.argmax(sh)
    g2 = np.array(list(np.ndindex(5, 5)))
    led2 = kv.Ledger(np.zeros((1, 1, 25)), DATY[:1], g2, ("a", "b"))
    assert np.isnan(kv.plato(led2, np.ones(25))[0])                    # угол 2D — 2 соседа: край


def test_fiksirovannaya_finds_plateau_on_other_years():
    led = _led(amp=4.0, m=8)
    f = kv.fiksirovannaya(led, (2016, 2017), GODY)
    assert np.abs(led.idx[f["tochka"]] - 2).sum() <= 3 and f["sharpe"] > 0.5


def test_seed2_reports_per_year_and_seeds_differ():
    a, b = _led(seed=1, amp=2.0, m=7), _led(seed=2, amp=2.0, m=7)
    assert a.idx.shape != b.idx.shape or not np.array_equal(a.idx, b.idx)
    out = kv.seed2(kv.prostaya_kalibr, a, b, GODY)
    assert set(out) == set(GODY) and all(np.isfinite(x["rashozhdenie"]) for x in out.values())
    same = kv.seed2(kv.prostaya_kalibr, a, a, GODY)
    assert all(x["rashozhdenie"] == 0 for x in same.values())


# ---------------- причинность ----------------

def test_mutatsiya_okno_s_godom_t_lovitsya():
    led = _led(amp=1.5, m=5)
    kv.prichinnost(kv.prostaya_kalibr, led, GODY)                     # честная процедура проходит

    def bad(le, gody):                                                # окно ≤ t вместо ≤ t−1
        god = kv._god(le)
        dni = np.flatnonzero(np.isin(god, gody))
        x = le.r[god <= max(gody)].mean(0)
        w = np.zeros((len(dni), NT, le.r.shape[2]))
        w[:, np.arange(NT), np.argmax(x, axis=1)] = 1.0
        return kv.Vyhod(dni, w)
    with pytest.raises(AssertionError):
        kv.prichinnost(bad, led, GODY)
