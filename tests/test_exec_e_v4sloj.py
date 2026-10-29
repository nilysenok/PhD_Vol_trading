"""П2 плана сетки v5: слой исполнения v4 как фильтр сигнала до автомата (fastbt/exec_e_v4sloj.py).
Ожидания в тестах посчитаны вручную по V4:211-275 (fastbt/execlayer.py); мутации подменяют execlayer.exec_layer."""
import numpy as np
import pytest

from fastbt import execlayer
from fastbt.exec_e import Day, Params, run
from fastbt.exec_e_v4sloj import filtr_signala, run_v4

ORIG = execlayer.exec_layer


def _eq(got, want):
    assert np.array_equal(np.asarray(got, float), np.asarray(want, float)), (list(got), list(want))


def proverka_pravil():
    """Ручные эталоны; падает на любой из подложенных мутаций."""
    _eq(filtr_signala([1, 0, 1, 1, 1, 0], consec=2), [0, 0, 0, 1, 1, 0])
    _eq(filtr_signala([1, 1, -1, 0, 0, 0], min_hold=2), [1, 1, -1, 0, 0, 0])
    _eq(filtr_signala([1, 0, 1, 1, 0, 1, 1], cooldown=2), [1, 0, 0, 0, 0, 1, 1])
    _eq(filtr_signala([1, 1, -1, 1], strength=[.5, .2, .4, .3], min_sily=0.3), [1, 0, -1, 1])
    _eq(filtr_signala([-1, -1, 0, -1, -1], consec=2), [0, -1, 0, 0, -1])   # шорт: знак один раз, не |sig|


def test_ruchnye_etalony():
    proverka_pravil()


def test_vyklyucheno_pobaytno_tot_zhe_sig():
    s = np.array([1, -1, 0, 1, 1, 0, -1.0])
    assert filtr_signala(s) is s
    assert filtr_signala(s, consec=1, min_hold=0, cooldown=0, min_sily=0.0).tobytes() == s.tobytes()


def test_sovpadaet_s_execlayer_na_sintetike():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = 60
        raw = rng.integers(-1, 2, n).astype(float)
        st = rng.random(n)
        c, h, cd, ms = int(rng.integers(1, 4)), int(rng.integers(0, 6)), int(rng.integers(0, 11)), float(rng.choice([0, .3, .5]))
        _eq(filtr_signala(raw, st, c, h, cd, ms), ORIG(raw, st, c, h, cd, ms, n))


def _days(n, seed=3):
    rng = np.random.default_rng(seed)
    px, out = 100.0, []
    for _ in range(n):
        o = px * (1 + 0.003 * rng.standard_normal(6).cumsum() / 3)
        c = np.append(o[1:], o[-1] * (1 + 0.002 * rng.standard_normal()))
        out.append(Day(o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c))
        px = c[-1]
    return out


def test_avtomat_pobaytno_pri_vyklyuchennyh_osyah():
    n = 80
    days = _days(n)
    sig = np.random.default_rng(1).integers(-1, 2, n)
    sg = np.full(n, 0.02)
    p = Params(a=1.0, b=1.0)
    a, b = run("T", days, sig, sg, p), run_v4("T", days, sig, sg, p)
    for f in ("gross", "cost", "pos", "free"):
        assert getattr(a, f).tobytes() == getattr(b, f).tobytes()
    assert [(t.entry_day, t.exit_day, t.side, t.reason) for t in a.trades] == [(t.entry_day, t.exit_day, t.side, t.reason) for t in b.trades]


def test_avtomat_shorty_ostayutsya_shortami():
    sig = [-1, -1, -1, 0, 0]
    days = [Day(*[np.array([100.0, 100.0])] * 4) for _ in sig]
    r = run_v4("T", days, sig, np.full(5, 0.02), Params(), consec=2)
    assert r.trades and all(t.side == -1 for t in r.trades)
    assert len(r.trades) == len(run_v4("T", days, sig, np.full(5, 0.02), Params(), consec=2).trades)


def test_filtr_vliyaet_na_avtomat():
    n = 80
    days, sig, sg = _days(n), np.random.default_rng(1).integers(-1, 2, n), np.full(n, 0.02)
    a = run("T", days, sig, sg, Params())
    b = run_v4("T", days, sig, sg, Params(), consec=3, cooldown=5)
    assert len(b.trades) < len(a.trades)


def test_oshibki_vhoda():
    with pytest.raises(ValueError):
        filtr_signala([1, 0], min_sily=0.3)
    with pytest.raises(ValueError):
        filtr_signala([1, 0], consec=0)


# --- мутации ---
def _mut_dvoynoy_znak(raw, st, c, h, cd, ms, end):               # № 20: знак умножен повторно после слоя
    return ORIG(raw, st, c, h, cd, ms, end) * np.sign(raw)


def _mut_cooldown_ot_vhoda(raw, st, c, h, cd, ms, end):          # cooldown стартует в день входа, не выхода
    b = np.abs(ORIG(raw, st, c, h, 0, ms, end))
    res, cl = np.zeros(len(raw)), 0
    for i in range(end):
        if cl > 0:
            cl -= 1
            continue
        if b[i] > 0:
            res[i] = 1.0
            if i == 0 or res[i - 1] == 0:
                cl = cd
    return res * np.sign(raw)


def _mut_consec_d_plus_1(raw, st, c, h, cd, ms, end):            # подтверждение смотрит на сигнал d+1
    o = ORIG(raw, st, c, h, cd, ms, end)
    return np.append(o[1:], 0.0)


@pytest.mark.parametrize("mut", [_mut_dvoynoy_znak, _mut_cooldown_ot_vhoda, _mut_consec_d_plus_1])
def test_mutatsii_lovyatsya(monkeypatch, mut):
    monkeypatch.setattr(execlayer, "exec_layer", mut)
    with pytest.raises(AssertionError):
        proverka_pravil()


def test_mut20_dvoynoy_znak_menyaet_storonu_sdelki(monkeypatch):
    sig, days = [-1, -1, -1, 0, 0], [Day(*[np.array([100.0, 100.0])] * 4) for _ in range(5)]
    monkeypatch.setattr(execlayer, "exec_layer", _mut_dvoynoy_znak)
    r = run_v4("T", days, sig, np.full(5, 0.02), Params(), consec=2)
    assert any(t.side == 1 for t in r.trades)          # мутант: шорт стал лонгом
