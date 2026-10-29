"""Ж-129 п. 7: min_hold при нулевом сигнале держит знак входа (по умолчанию); "v4" — прежнее обнуление, побитово."""
import inspect

import numpy as np
import pytest

from fastbt import exec_e_v4sloj as M
from fastbt import execlayer
from fastbt.exec_e_v4sloj import filtr_signala

ORIG = execlayer.exec_layer
ZV = M._sloj_znak_vhoda


def _znak(raw, h, **kw):
    return [float(x) for x in filtr_signala(raw, min_hold=h, **kw)]


def test_v4_pobaytno_50_naborov():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = 60
        raw = rng.integers(-1, 2, n).astype(float)
        st = rng.random(n)
        c, h, cd, ms = int(rng.integers(1, 4)), int(rng.integers(0, 6)), int(rng.integers(0, 11)), float(rng.choice([0, .3, .5]))
        got = filtr_signala(raw, st, c, h, cd, ms, min_hold_rezhim="v4")
        assert got.tobytes() == ORIG(raw, st, c, h, cd, ms, n).tobytes()


def test_znak_vhoda_derzhit_do_sroka_potom_vyhod():
    assert _znak([1, 0, 0, 0, 0], 3) == [1, 1, 1, 0, 0]
    assert _znak([-1, 0, 0, 0], 2) == [-1, -1, 0, 0]
    assert _znak([1, 0, 0, 0, 0], 3, min_hold_rezhim="v4") == [1, 0, 0, 0, 0]


def test_razvorot_vnutri_min_hold_kak_v4():
    # знак из сырого сигнала, срок не перезапускается; нуль после разворота — знак входа до конца срока
    assert _znak([1, -1, 0, 0], 3) == [1, -1, 1, 0]
    assert _znak([1, -1, 0, 0], 3, min_hold_rezhim="v4") == [1, -1, 0, 0]


def test_sovpadaet_s_v4_gde_net_nulei_v_derzhanii():
    rng = np.random.default_rng(5)
    for _ in range(50):
        raw = rng.choice([-1.0, 1.0], 50)
        h, c = int(rng.integers(1, 6)), int(rng.integers(1, 3))
        assert _znak(raw, h, consec=c) == _znak(raw, h, consec=c, min_hold_rezhim="v4")


def test_umolchanie_znak_vhoda_a_ne_v4():
    """Падает, если умолчание вернуть на "v4" (мутация): и явный "v4" в чужих тестах нужен, т.к. умолчание отличается."""
    assert inspect.signature(filtr_signala).parameters["min_hold_rezhim"].default == "znak_vhoda"
    assert _znak([1, 0, 0, 0], 2) != _znak([1, 0, 0, 0], 2, min_hold_rezhim="v4")
    with pytest.raises(ValueError):
        filtr_signala([1, 0], min_hold=2, min_hold_rezhim="x")


def test_mutatsiya_obratnyj_znak_lovitsya(monkeypatch):
    monkeypatch.setattr(M, "_sloj_znak_vhoda", lambda *a: -ZV(*a))
    assert _znak([1, 0, 0, 0, 0], 3) != [1, 1, 1, 0, 0]


def test_mutatsiya_derzhanie_dolshe_lovitsya(monkeypatch):
    monkeypatch.setattr(M, "_sloj_znak_vhoda", lambda r, s, c, h, cd, m: ZV(r, s, c, h + 1, cd, m))
    assert _znak([1, 0, 0, 0, 0], 3) != [1, 1, 1, 0, 0]
