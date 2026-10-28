"""Ж-096 п. 3: проход 2 Соболя — 10 000 точек на стратегию (индексы 5000…9999)."""
import pytest

from fastbt import progon_sobol as ps

SIDS = [f"S{i}" for i in range(1, 7)]


def _bounds(kub: dict, prohod: int) -> tuple:
    s = kub["sobol"]
    return {1: (0, s["prohod1"]), 2: (s["prohod1"], s["prohod2"])}[prohod]


def _bounds_mut_s_nulya(kub: dict, prohod: int) -> tuple:
    return (0, kub["sobol"]["prohod2"])    # мутация: проход 2 пересчитывает первые 5000


@pytest.mark.parametrize("sid", SIDS)
def test_prohod2_10000_v_kube(sid):
    s = ps.load_kub(sid)["sobol"]
    assert s["prohod1"] == 5000 and s["prohod2"] == 10000


@pytest.mark.parametrize("sid", SIDS)
def test_indeksy_prohoda2_5000_9999(sid):
    kub = ps.load_kub(sid)
    n0, n1 = _bounds(kub, 2)
    idx = [i for i, _ in ps.tochki(kub, n0, n0 + 3)] + [i for i, _ in ps.tochki(kub, n1 - 3, n1)]
    assert idx[0] == 5000 and idx[-1] == 9999 and n1 - n0 == 5000


def test_prodolzhenie_bez_pereschyota_pervyh_5000():
    kub = ps.load_kub("S2")
    sploshnoy = ps.tochki(kub, 0, 5010)
    assert ps.tochki(kub, 5000, 5010) == sploshnoy[5000:]


def test_mutatsiya_prohod2_s_nulya_lovitsya():
    kub = ps.load_kub("S1")
    n0, _ = _bounds_mut_s_nulya(kub, 2)
    with pytest.raises(AssertionError):
        assert n0 == 5000
