"""V6-12c, синтетика (данные не читает): оси исполнения lestnica, gap, t_vyhod, vyhod_lim, f_neisp (fastbt/eval_v6_ispoln.py) и свёртки canon_v6
(piv_buf без стопа piv:next|half; time_decay без tau или при тейке ≠ net). Реальные данные — test_eval_v6c_real.py."""
from types import SimpleNamespace

import pytest

from fastbt import canon_v5, canon_v6, eval_v6, eval_v6_ispoln as isp
from fastbt.eval_v6 import fkey, razdelit
from fastbt.etap1_eval_v2 import NOVYE, NeSchitana

SIDS = canon_v5.SIDS


def _r(vhod_tip="podtverzhdenie", k=0.5):
    return SimpleNamespace(pt4={"vhod": k}, nov={**NOVYE, "vhod_tip": vhod_tip}, ext={})


def test_primenit_dohodit_do_razbora():
    r = _r()
    isp.primenit(r, {"gap": "propusk", "t_vyhod": "open2", "vyhod_lim": "da", "f_neisp": "zakrytie_d1", "lestnica": 0.5})
    assert (r.nov["gap"], r.nov["t_vyhod"], r.ext) == ("propusk", "open2", {"vyhod_lim": True, "f": "zakrytie_d1"})
    r = _r()
    isp.primenit(r, {})
    assert r.nov == _r().nov and r.ext == {}


def test_proverit_nesovmestimoe():
    r = _r()
    isp.proverit(r, {"vyhod_lim": "da"})
    r.nov["t_vyhod"] = "1500"
    isp.proverit(r, {"t_vyhod": "1500"})
    with pytest.raises(NeSchitana, match="vyhod_lim"):
        isp.proverit(r, {"vyhod_lim": "da", "t_vyhod": "1500"})
    for k in ("gap", "vyhod_lim", "lestnica"):
        with pytest.raises(NeSchitana, match="стоп-заявка"):
            isp.proverit(_r("stop_zayavka"), {k: 1})


# ---------------- лесенка: ноги, веса, восстановление k ----------------
class _Baza:
    def __init__(self):
        self.r5, self.v6, self.log = SimpleNamespace(pt4={"vhod": 0.5}), {"lestnica": 1.0}, []

    def _ves(self, sg, rg):
        return 1.0

    def _rows(self, pt, period, anchor):
        self.log.append((self.r5.pt4["vhod"], self._ves(None, None)))
        if getattr(self, "padat", False) and len(self.log) == 2:
            raise RuntimeError("нога 2")
        return [(0, SimpleNamespace(k=self.r5.pt4["vhod"]), 0, 9)]


class _E(isp.IspolnMixin, _Baza):
    pass


def test_lestnica_dve_nogi_po_polovine(monkeypatch):
    monkeypatch.setattr(isp, "sum_rows", lambda rs: [r.k for r in rs])
    e = _E()
    assert e._legs({}, "train4", False) == [(0, [0.5, 1.5], 0, 9)]
    assert e.log == [(0.5, 0.5), (1.5, 0.5)] and e.r5.pt4["vhod"] == 0.5 and e._ves(None, None) == 1.0   # w/2 на ногу; k и вес восстановлены
    e.v6, e.log = {}, []
    assert e._legs({}, "train4", False)[0][1].k == 0.5 and e.log == [(0.5, 1.0)]                           # без оси — один прогон, вес 1


def test_lestnica_vosstanavlivaet_k_pri_sboe(monkeypatch):
    monkeypatch.setattr(isp, "sum_rows", lambda rs: rs)
    e = _E()
    e.padat = True
    with pytest.raises(RuntimeError):
        e._legs({}, "train4", False)
    assert e.r5.pt4["vhod"] == 0.5 and e._ves(None, None) == 1.0


# ---------------- GOTOVY, ключ кэша, Соболь ----------------
def test_gotovy_otkryvaet_pyat_osej():
    for n in isp.ISPOLN:
        assert n in canon_v6.V12 and n in eval_v6.PODKLYUCHENO
    for sid in SIDS:
        kub = canon_v6.load_kub(sid)
        v_kube = [n for n in isp.ISPOLN if n in canon_v6.novye(kub)]
        assert v_kube and set(v_kube) <= {n for n, _, _ in canon_v6.osi(kub)}
        assert not set(v_kube) & {n for n, _, _ in canon_v6.osi(kub, canon_v6.GOTOVY - {f"V6-12:{o}" for o in isp.ISPOLN})}


def test_fkey_razlichaet_nogi_i_znacheniya():
    ks = {fkey({}, {k: v}) for k, v in (("lestnica", 0.5), ("lestnica", 1.0), ("gap", "propusk"), ("t_vyhod", "open2"), ("t_vyhod", "1500"),
                                         ("t_vyhod", "1700"), ("vyhod_lim", "da"), ("f_neisp", "zakrytie_d1"))}
    assert len(ks) == 8 and fkey({}, {}) == ()


@pytest.mark.parametrize("sid", SIDS)
def test_sobol_variruet_novye_osi_bez_ne_podklyucheno(sid):
    kub = canon_v6.load_kub(sid)
    pts = [c for _, c in canon_v6.tochki_v6(kub, 0, 1500)[0]]
    nov = canon_v6.novye(kub)
    for n in (x for x in isp.ISPOLN if x in nov):
        assert any(n in c for c in pts), f"{sid}.{n}: ось ни разу не выпала"
    for c in pts[:400]:
        razdelit(sid, c)


# ---------------- свёртки canon_v6 ----------------
def _q(sid, **kw):
    kub = canon_v6.load_kub(sid)
    base = {n: g[0] for n, g in canon_v5.osi(kub["v5"])}
    return kub, {**base, **{n: v["setka"][0] for n, v in canon_v6.novye(kub).items()}, **kw}


def test_time_decay_vybros_bez_tau_ili_pri_tejke():
    """V6-12d (решение дирижёра 29.10): без tau / при тейке ≠ net — не выброс точки, а свёртка в «выкл.» (точки Соболя сохраняются)."""
    kub, q = _q("S1", time_decay="da", tau=10, tejk="net")
    assert canon_v6.canon_v6("S1", q, kub)["time_decay"] == "da"
    for bad in (dict(tau="net"), dict(tejk="pct:0.05")):
        c = canon_v6.canon_v6("S1", {**q, **bad}, kub)
        assert c is not None and "time_decay" not in c, bad
        assert c == canon_v6.canon_v6("S1", {**q, **bad, "time_decay": "net"}, kub), bad       # ≡ точке с выключенной осью
    assert "time_decay" not in canon_v6.canon_v6("S1", {**q, "time_decay": "net", "tau": "net"}, kub)   # «выкл.» — не выброс


def test_piv_buf_svertka_h4():
    kub, q = _q("S5", piv_buf=0.1)
    for kw in ({}, dict(stop_nach="sigma:1.0", tejk="piv:R"), dict(tejk="piv:half")):
        assert "piv_buf" not in canon_v6.canon_v6("S5", {**q, **kw}, kub), kw
    assert canon_v6.canon_v6("S5", {**q, "stop_nach": "piv:half"}, kub)["piv_buf"] == 0.1
