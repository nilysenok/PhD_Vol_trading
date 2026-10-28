"""CE-9: пересчёт поверх A′ — только замороженные точки, отбор не вызывается (синтетика)."""
import pytest

from fastbt import perescet_ce_aprim as pc

FROZEN = [dict(family="C1", W=126, label="a"), dict(family="D2", W=252, label="b"), dict(family="E1", W=63, label="c")]


def boom(*a, **k):
    raise AssertionError("отбор вызван")


def test_mnozhestvo_i_poryadok_zamorozheny():
    out = pc.perescet(FROZEN, {"sha": "abc"}, boom)
    assert [{k: o[k] for k in pc.POLYA} for o in out] == FROZEN


def test_pometki_perescheta_i_ne_v_N():
    out = pc.perescet(FROZEN, {"sha": "abc"}, boom)
    assert all(o["istochnik"] == "perescet" and o["baza"] == "A_prim:abc" and o["v_N"] is False for o in out)
    assert pc.perescet(FROZEN, None, boom)[0]["baza"] == "A_strogaya"


def test_vhod_ne_menyaetsya():
    src = [dict(t) for t in FROZEN]
    pc.perescet(src, None, boom)
    assert src == FROZEN


def test_plohoy_vhod():
    with pytest.raises(ValueError):
        pc.perescet([], None, boom)
    with pytest.raises(ValueError):
        pc.perescet([dict(family="C1", W=1)], None, boom)
    with pytest.raises(ValueError):
        pc.perescet(FROZEN, {}, boom)


def test_mutaciya_perescet_vyzyvaet_otbor_lovitsya(monkeypatch):
    """Подложенная версия, заново вызывающая отбор, обязана упасть на boom."""
    def bad(zam, aprim, otbor):
        otbor(zam)
        return pc.perescet(zam, aprim, otbor)
    with pytest.raises(AssertionError):
        bad(FROZEN, None, boom)
    assert pc.perescet(FROZEN, None, boom)
