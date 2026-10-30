"""Свёртка осей ведения стопа при пивот/VWAP-стопе (вместо None): трейл, безубыток, stop_v_plyus, chast → «выкл.»; тейк не трогается."""
from unittest import mock

import pytest

from fastbt import canon_v6 as c6, canon_v6_nesovm as nsv
from fastbt.eval_v5_osi import yakor


def _g(sid, os, pref):
    return next(v for v in dict((n, g) for n, g, _ in c6.osi(c6._kub(sid)))[os] if str(v).startswith(pref))


def kanon(sid, **kw):
    pt = {n: g[0] for n, g, _ in c6.osi(c6._kub(sid))}
    pt.update({k: v for k, v in yakor(sid).items() if k in pt})
    return c6.canon_v6(sid, {**pt, **kw})


ZAKR = dict(trail="atr:1.0", bezubytok="atr:1.0", stop_v_plyus="2:1", chast="1/2:1", pir=0.5, razmer_tip="risk_sigma")
SV = {"trail": "net", "bezubytok": "net", "stop_v_plyus": "net", "chast": "net", "pir": 0, "razmer_tip": "ravnyj"}


@pytest.mark.parametrize("sid,ax", [("S5", "stop_nach"), ("S6", "stop_nach")])
def test_svertka_vmesto_none(sid, ax):
    st = _g(sid, ax, "piv:next" if sid == "S5" else "vwap")
    for n, v in ZAKR.items():
        c = kanon(sid, stop_nach=st, **{n: v})
        assert c is not None and c.get(n, SV[n]) == SV[n], (sid, n)
    c = kanon(sid, stop_nach=st, **ZAKR)
    assert c is not None and all(c.get(n, SV[n]) == SV[n] for n in ZAKR)
    assert kanon(sid, stop_nach=st, tejk=_g(sid, "tejk", "rr")) is None          # rr — тейк от стопа: не сворачивается, отказ


def test_tejk_ne_svorachivaetsya_i_pol_puti_ostayotsya_none():
    st = _g("S5", "stop_nach", "piv:next")
    assert kanon("S5", stop_nach=st, tejk=_g("S5", "tejk", "piv:R"), **ZAKR)["tejk"] == "piv:R"
    assert kanon("S5", stop_nach=st, tejk="pct:0.01")["tejk"] == "pct:0.01"      # другие тейки пивот-точку не выбрасывают
    assert kanon("S5", stop_nach=st, tejk=_g("S5", "tejk", "pol_puti")) is None  # правило 6


def test_ne_pivot_ne_svorachivaetsya():
    c = kanon("S5", stop_nach="atr:2.0", **ZAKR)
    assert all(c[n] == v for n, v in ZAKR.items())


def test_mutaciya_svertka_snyata_lovitsya():
    st = _g("S5", "stop_nach", "piv:next")
    assert kanon("S5", stop_nach=st, trail="atr:1.0").get("trail", "net") == "net"
    with mock.patch.object(nsv, "svernut", lambda q: q):
        c = kanon("S5", stop_nach=st, trail="atr:1.0")
    assert c.get("trail", "net") != "net"                                        # мутант оставляет трейл у пивот-точки: тест выше падает


def test_mutaciya_svertka_trogaet_tejk_lovitsya():
    st = _g("S5", "stop_nach", "piv:next")
    ok = kanon("S5", stop_nach=st, tejk="pct:0.01")
    real = nsv.svernut

    def mut(q):
        q = real(q)
        if nsv.edinica(q.get("stop_nach")) in ("piv", "vwap"):
            q["tejk"] = "net"
        return q
    with mock.patch.object(nsv, "svernut", mut):
        bad = kanon("S5", stop_nach=st, tejk="pct:0.01")
    assert ok["tejk"] == "pct:0.01" and bad["tejk"] != "pct:0.01"                # тест выше (== pct:0.01) на мутанте падает
