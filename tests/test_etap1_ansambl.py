"""Ж-051 итерация 2: ансамбль по плато, чистая часть (без данных). Синтетика с известным ответом + мутации."""
import numpy as np
import pytest

from fastbt import etap1_ansambl as A
from fastbt.etap1_poisk_vorota import Ledger, matrix

AXES = {"z": [1, 2, 3], "w": [10, 20], "k_s": [1.0, 2.0, "net"], "sobytiya": ["net", "cbr", "div"]}
C = {"z": 2, "w": 10, "k_s": 2.0, "sobytiya": "net"}


def test_tochki_po_osyam_radius_1():
    pts = A.tochki(C, ["z", "w"], AXES)
    assert pts[0] == C and len(pts) == 4                  # центр + z=1, z=3 + w=20 (у края w один сосед)
    assert all(sum(p[a] != C[a] for a in C) <= 1 for p in pts)      # углов куба нет
    assert len(A.tochki(C, ["z", "w", "k_s"], AXES)) == 6


def test_tochki_propuskayut_ne_schitannye_sobytiya():
    pts = A.tochki({**C, "sobytiya": "cbr"}, ["sobytiya"], AXES)
    assert [p["sobytiya"] for p in pts] == ["cbr", "net"]            # div не считается (❓5)


def test_osi_nabora():
    assert A.osi_nabora("top2", ["w", "z", "k_s"], AXES) == ["w", "z"]
    assert A.osi_nabora("vse", [], AXES) == list(AXES)
    with pytest.raises(ValueError):
        A.osi_nabora("top9", [], AXES)


def test_signal_points_skhlopyvayut_osi_ispolneniya():
    sk = A.signal_points(A.tochki(C, ["z", "k_s"], AXES), ["z", "w"])
    assert [p["z"] for p in sk] == [2, 1, 3]                         # k_s сигнал не меняет: дубли центра убраны


P = np.array([[1, 1, 0, -1, 0], [1, 0, 0, -1, 1], [1, 0, -1, 1, 0]], float)


def test_srednee_i_golos_izvestnyy_otvet():
    d, w = A.combine(P, "srednee")
    assert list(d) == [1, 1, -1, -1, 1] and np.allclose(w, [1, 1 / 3, 1 / 3, 1 / 3, 1 / 3])
    d, w = A.combine(P, "golos")                                      # строгое большинство из 3: нужно ≥ 2
    assert list(d) == [1, 0, 0, -1, 0] and list(w) == [1, 0, 0, 1, 0]


@pytest.mark.parametrize("mode", A.MODES)
def test_golden_odna_tochka_pobaytno(mode):
    pos = np.array([0, 1, 1, 0, -1, -1, 0.0])
    d, w = A.combine(pos[None, :], mode)
    assert d.tobytes() == pos.tobytes() and w.tobytes() == np.abs(pos).tobytes()


def test_combine_ne_smotrit_v_budushchee():
    rng = np.random.default_rng(1)
    X = rng.choice([-1.0, 0.0, 1.0], size=(5, 40))
    Y = X.copy()
    Y[:, 20:] = -Y[:, 20:]                                            # будущее с дня 20 перевёрнуто
    for mode in A.MODES:
        a, b = A.combine(X, mode), A.combine(Y, mode)
        assert a[0][:20].tobytes() == b[0][:20].tobytes() and a[1][:20].tobytes() == b[1][:20].tobytes()


def _lookahead(P, mode):
    d, w = A.combine(P, mode)
    d2, w2 = A.combine(np.roll(P, -1, axis=1), mode)                  # мутация: усреднение с точкой d+1
    return (d + d2) / 2, (w + w2) / 2


def test_mutatsiya_srednee_s_budushchikh_tochek_lovitsya():
    X = np.random.default_rng(2).choice([-1.0, 0.0, 1.0], size=(5, 40))
    X[:, 19:21] = 1.0                                                 # день 19 и 20 определённо ненулевые
    Y = X.copy()
    Y[:, 20:] = -Y[:, 20:]
    a, b = _lookahead(X, "srednee"), _lookahead(Y, "srednee")
    assert a[0][:20].tobytes() != b[0][:20].tobytes() or a[1][:20].tobytes() != b[1][:20].tobytes()    # день 19 видит день 20


def test_mutatsiya_golos_vmesto_srednee_lovitsya(monkeypatch):
    ok = A.combine(P, "srednee")
    real = A.combine
    monkeypatch.setattr(A, "combine", lambda M, mode: real(M, "golos"))      # мутация: режим подменён
    bad = A.combine(P, "srednee")
    assert not (np.array_equal(ok[0], bad[0]) and np.allclose(ok[1], bad[1]))   # ожидаемые 1/3 против 0/1


def test_combine_validaciya():
    with pytest.raises(ValueError):
        A.combine(np.empty((0, 3)), "srednee")
    with pytest.raises(ValueError):
        A.combine(P, "mediana")


def test_plato_ansambl():
    o = {(n, m): 0.5 for n in A.NABORY for m in A.MODES}
    assert A.plato_ansambl(o, ("top3", "golos")) == dict(ok=True, n_sosedey=3, sosedi=[("top2", "golos"), ("vse", "golos"), ("top3", "srednee")])
    o[("top2", "golos")] = 0.3                                        # провал соседа > 0,1
    assert not A.plato_ansambl(o, ("top3", "golos"))["ok"]
    o[("top2", "golos")] = float("nan")
    assert not A.plato_ansambl(o, ("top3", "golos"))["ok"]            # nan -> ложь
    assert A.plato_ansambl(o, ("top2", "srednee"))["n_sosedey"] == 2


def test_obshchaya_matrica_i_N():
    led = Ledger(lambda p, per: {"excess_d": {2018: [0.0] * 3, 2019: [0.0] * 3}})
    led({"a": 1}, "train")
    assert matrix(led).shape == (6, 1) and led.n_unique == 1
    r = {"excess_d": {2018: [1.0, 2.0, 3.0], 2019: [1.0, 0.0, 1.0]}}
    A.dobavit_v_ledger(led, "top2", "golos", r)
    assert matrix(led).shape == (6, 2) and led.n_unique == 2
    with pytest.raises(ValueError):
        A.dobavit_v_ledger(led, "top2", "golos", r)                   # только дополнение
