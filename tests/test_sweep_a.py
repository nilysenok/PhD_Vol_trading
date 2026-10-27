"""Э1-г, PR 2: ядро сборок A (`fastbt.sweep_a_core`) на синтетике с известным ответом. Данных проекта не читает.
Детекторы `ok_*` возвращают True, если свойство выполнено; мутации — `test_sweep_a_mut.py`."""
import numpy as np
import pandas as pd

from fastbt import sweep_a_core as core
from fastbt.sweep_a import SynthPanelSource
from fastbt.sweep_a_core import Panel

NET = {"vesa": "ravnye", "limit_pozitsiy": "net", "limit_na_strategiyu": "net"}


def mkpanel(pos, lr=None, st=None, sid=None, rfc=None):
    pos = np.array(pos, float)
    D, P = pos.shape
    return Panel(pd.bdate_range("2018-01-01", periods=D), np.array(sid or ["S1"] * P), pos,
                 np.zeros((D, P)) if lr is None else np.array(lr, float), np.ones((D, P), bool),
                 np.zeros((D, P)) if st is None else np.array(st, float), np.zeros(D) if rfc is None else np.array(rfc, float))


def manual_a(pn, comm):
    """Расчёт A по определению: на день d вес 1/cnt; комиссия за каждую смену позиции и за выход на последнем баре."""
    D, P = pn.pos.shape
    out = []
    for d in range(D):
        net = 0.0
        for p in range(P):
            prev = pn.pos[d - 1, p] if d else 0.0
            side = abs(pn.pos[d, p] - prev) + (abs(pn.pos[d, p]) if d == D - 1 else 0.0)
            net += pn.pos[d, p] * pn.lr[d, p] - side * comm
        expo = sum(abs(pn.pos[d, p]) for p in range(P)) / P
        out.append(net / P - expo * pn.rfc[d])
    return np.array(out)


def ok_etalon() -> bool:
    pn = mkpanel([[1, 0, 0], [1, -1, 0], [0, -1, 1], [0, -1, 1]], lr=[[.01, .02, -.01], [-.02, .01, .03], [.01, -.02, .01], [.02, .01, -.03]],
                 rfc=[.0001, .0002, .0001, .0002])
    got = core.assemble(pn, NET, 0.0004)["excess"].values
    return bool(np.allclose(got, manual_a(pn, 0.0004), atol=1e-12))


def ok_ves_bez_tekushchego_dnya() -> bool:
    pn = SynthPanelSource(2, ("S1", "S2"), seed=3).panel("2018-06-01")
    d0, ok = 50, True
    for sch in ("obratnaya_vol", "paritet_riska"):
        w1 = core.full_weights(pn, sch)
        pn2 = Panel(pn.dates, pn.sid, pn.pos, pn.lr.copy(), pn.act, pn.strength, pn.rfc)
        pn2.lr[d0] *= 10                                  # шок в день d0: вес d0 не меняется, вес d0+1 — меняется
        w2 = core.full_weights(pn2, sch)
        ok &= bool(np.array_equal(w1[:d0 + 1], w2[:d0 + 1]) and not np.allclose(w1[d0 + 1], w2[d0 + 1]))
    return ok


def ok_obratnaya_vol_izvestnyy_otvet() -> bool:
    base = np.random.default_rng(1).normal(size=(40, 1))
    h = np.hstack([base, 4 * base])
    w = core.slot_weights("obratnaya_vol", h)
    return bool(np.allclose(w, [0.8, 0.2]))


def ok_paritet_diagonal() -> bool:
    w = core.erc(np.diag([1.0, 4.0, 9.0]))
    return bool(np.allclose(w, np.array([1, 1 / 2, 1 / 3]) / (1 + 1 / 2 + 1 / 3), atol=1e-8))


def ok_limit_soblyudaetsya() -> bool:
    pn = SynthPanelSource(4, ("S1", "S2", "S3"), seed=5).panel("2019-01-01")
    g = core.gate(pn, 3, 2)
    per = [(np.abs(g[:, pn.sid == s]) > 0).sum(axis=1).max() for s in ("S1", "S2", "S3")]
    sub = bool(np.all((g == 0) | (g == pn.pos)))
    return bool((np.abs(g) > 0).sum(axis=1).max() <= 3 and max(per) <= 2 and (np.abs(g) > 0).sum(axis=1).max() == 3 and sub)


def ok_sila_pri_ravenstve() -> bool:
    """L=1: двое входят в один день — берётся сильнейший; отвергнутая сделка не входит позже, когда слот освободился."""
    pn = mkpanel([[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 1]], st=[[1, 2, 0]] * 4)
    g = core.gate(pn, 1, None)
    return bool(np.array_equal(g, [[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]))


def ok_fifo() -> bool:
    """L=1: первый вошедший держит слот, более сильный новичок отвергается."""
    pn = mkpanel([[1, 0], [1, 1], [1, 1]], st=[[1, 0], [1, 9], [1, 9]])
    return bool(np.array_equal(core.gate(pn, 1, None), [[1, 0], [1, 0], [1, 0]]))


def ok_slot_1_cherez_L() -> bool:
    pn = mkpanel([[1, 1, 1, 0]], lr=[[0, 0, 0, 0]])
    g = core.gate(pn, 4, None)
    w = core.weights(pn, "ravnye", g, 4)
    return bool(np.allclose(w[0, :3], 0.25))


def test_etalon_ravnye_net_net_raven_ruchnomu_A():
    assert ok_etalon()


def test_ves_bez_tekushchego_dnya():
    assert ok_ves_bez_tekushchego_dnya()


def test_ves_obratnaya_vol_izvestnyy_otvet():
    assert ok_obratnaya_vol_izvestnyy_otvet()


def test_ves_paritet_diagonal_daet_odin_na_sigma():
    assert ok_paritet_diagonal()


def test_limit_pozitsiy_i_na_strategiyu_soblyudayutsya():
    assert ok_limit_soblyudaetsya()


def test_limit_sila_pri_ravenstve_i_otverg_ne_vhodit_pozzhe():
    assert ok_sila_pri_ravenstve()


def test_limit_fifo():
    assert ok_fifo()


def test_limit_slot_ves_1_na_L():
    assert ok_slot_1_cherez_L()


def test_short_zarabatyvaet_minus_lr_a_ne_lonh():
    """Шорт при росте цены теряет, при падении зарабатывает (ошибка знака в sweep_post.py:162 превращала шорты в лонги)."""
    pn = mkpanel([[-1], [-1]], lr=[[0.02], [-0.01]])
    r = core.assemble(pn, NET, 0.0)["excess"].values
    assert np.allclose(r, [-0.02, 0.01])
    assert (core.assemble(mkpanel([[1], [1]], lr=[[0.02], [-0.01]]), NET, 0.0)["excess"].values == [0.02, -0.01]).all()


def test_plecho_zapreshcheno():
    pn = mkpanel([[1, 1]])
    w = np.array([[1.0, 1.0]])
    try:
        core.portfolio(pn, pn.pos, w, 0.0)
    except AssertionError:
        return
    raise AssertionError("плечо прошло")


def test_pustaya_panel_nulevye_pozitsii():
    pn = mkpanel(np.zeros((5, 3)))
    for sch in core.VESA:
        r = core.assemble(pn, {**NET, "vesa": sch, "limit_pozitsiy": 3}, 0.0004)
        assert (r["excess"] == 0).all() and (r["exposure"] == 0).all()
