"""S8/S9: полная решётка 108 точек ровно по разу, независимо от --n1 (Ж-161 пп. 1–2). Данных не читает; Соболь — только S10 на 100 точек."""
from __future__ import annotations

import json

import pytest

from fastbt import canon_v7b as b, ledger_v2 as lm, progon_v7 as p

SECH = ("S8", "S9")


class RZ:
    def __init__(self, seed="seed"):
        self.seed = seed


def _kuby():
    return {s: (p.kub_put7(s), b.load_kub(s, p.kub_put7(s).parent)) for s in SECH + ("S10",)}


def _kl(sid, pt):
    return lm.klyuch_n(sid, {"tochka": pt})


def _prover(sid, arg, seed="seed"):
    kubs = _kuby()
    kub = kubs[sid][1]
    b.proverit(kub)
    pts = p.tochki_semya(kub, 0, p.n_po_sid(kubs, arg)[sid], RZ(seed))
    assert len(pts) == 108
    assert len({_kl(sid, pt) for _, pt in pts}) == 108
    assert len({i for i, _ in pts}) == 108
    return pts


@pytest.mark.parametrize("sid", SECH)
@pytest.mark.parametrize("arg", ["1000", "50"])
@pytest.mark.parametrize("seed", ["seed", "seed2"])
def test_108_razlichnyh_pri_lyubom_n1(sid, arg, seed):
    _prover(sid, arg, seed)


@pytest.mark.parametrize("sid", SECH)
def test_poryadok_po_klyuchu_i_seed_ne_vazhen(sid):
    a, c = _prover(sid, "1000", "seed"), _prover(sid, "50", "seed2")
    assert json.dumps(a, sort_keys=True) == json.dumps(c, sort_keys=True)
    kl = [_kl(sid, pt) for _, pt in a]
    assert kl == sorted(kl)


@pytest.mark.parametrize("sid", SECH)
def test_kuski_po_100_skladyvayutsya_v_reshetku(sid):
    kub = _kuby()[sid][1]
    kuski = [x for a in range(0, 108, 100) for x in p.tochki_semya(kub, a, min(a + 100, 108), RZ())]
    assert kuski == _prover(sid, "1000")


def test_mut_sobol_pervye_N_lovitsya(monkeypatch):
    """Мутация: сэмплер S8 = первые N индексов Соболя (как до правки) → проверка 108 различных падает."""
    monkeypatch.setattr(b, "reshetka", lambda k, g=None: b.tochki_v7b(k, 0, 50, "seed", g)[0], raising=False)
    monkeypatch.setattr(p, "tochki_semya", lambda k, n0, n1, rz: b.tochki_v7b(k, n0, n1, rz.seed, p.canon7().GOTOVY_V7)[0])
    with pytest.raises(AssertionError):
        _prover("S8", "50")


def test_mut_dubl_tochki_lovitsya(monkeypatch):
    real = b.reshetka
    monkeypatch.setattr(b, "reshetka", lambda k, g=None: (lambda r: r[:-1] + [(r[-1][0], r[0][1])])(real(k, g)))
    with pytest.raises(AssertionError):
        _prover("S9", "1000")


def test_s10_kak_do_pravki():
    kubs = _kuby()
    kub = kubs["S10"][1]
    do = b.tochki_v7b(kub, 0, 100, "seed", p.canon7().GOTOVY_V7)[0]
    posle = p.tochki_semya(kub, 0, 100, RZ())
    assert do and [(i, _kl("S10", pt)) for i, pt in posle] == [(i, _kl("S10", pt)) for i, pt in do]
    assert p.n_po_sid(kubs, "50")["S10"] == 50
