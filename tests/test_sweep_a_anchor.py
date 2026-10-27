"""Э1-г, PR 2: якорь. Сборка (равные, нет, нет) == A побайтно и там, где число пар в act меняется; смена 1/cnt
комиссию не платит, пересмотр по схеме (обр. волатильность) — платит. Данных проекта не читает."""
import numpy as np

from fastbt import sweep_a_core as core
from fastbt.sweep_a import SynthPanelSource
from fastbt.sweep_a_core import Panel
from test_sweep_a import NET, mkpanel


def panel_menyayushchiysya_act() -> Panel:
    pn = mkpanel([[1, 1, 0], [1, 1, 0], [1, 0, -1], [1, 0, -1]],
                 lr=[[.01, .02, 0], [-.02, .01, 0], [.01, 0, .03], [.02, 0, -.03]], rfc=[.0001, .0002, .0001, .0002])
    pn.act[:] = [[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]      # cnt по дням: 2, 2, 3, 2 — у пары 0 вес 1/cnt меняется
    pn.lr[~pn.act] = 0.0
    return pn


def manual_a_act(pn, comm):
    """A по определению: вес 1/cnt_d; комиссия за каждую смену позиции и выход на последнем баре пары; смена cnt не платится."""
    D, P = pn.pos.shape
    last = [np.flatnonzero(pn.act[:, p])[-1] for p in range(P)]
    out = []
    for d in range(D):
        cnt = pn.act[d].sum()
        net = expo = 0.0
        for p in range(P):
            prev = pn.pos[d - 1, p] if d else 0.0
            side = abs(pn.pos[d, p] - prev) + (abs(pn.pos[d, p]) if d == last[p] else 0.0)
            net += pn.pos[d, p] * pn.lr[d, p] - side * comm
            expo += abs(pn.pos[d, p])
        out.append(net / cnt - expo / cnt * pn.rfc[d])
    return np.array(out)


def ok_etalon_act() -> bool:
    pn = panel_menyayushchiysya_act()
    got = core.assemble(pn, NET, 0.0004)["excess"].values
    return bool(np.array_equal(got, manual_a_act(pn, 0.0004)) or np.allclose(got, manual_a_act(pn, 0.0004), atol=1e-15, rtol=0))


def ok_obr_vol_peresmotr_platit() -> bool:
    pn = SynthPanelSource(2, ("S1", "S2"), seed=3).panel("2018-06-01")
    pn.pos[:] = 1.0                                               # все пары держатся весь период: сделок нет, веса дрейфуют
    p = {**NET, "vesa": "obratnaya_vol"}
    pay = core.assemble(pn, p, 0.0004)
    free = core.assemble(pn, p, 0.0)
    eq = core.assemble(pn, NET, 0.0004)
    reb_turn = pay["turnover"] - eq["turnover"]
    return bool(reb_turn.sum() > 0 and (pay["excess"] - free["excess"]).sum() < (eq["excess"] - core.assemble(pn, NET, 0.0)["excess"]).sum())


def test_etalon_ravnye_net_net_raven_ruchnomu_A_pri_menyayushchemsya_act():
    assert ok_etalon_act()


def test_obratnaya_vol_peresmotr_vesov_dayet_izderzhki():
    assert ok_obr_vol_peresmotr_platit()


def test_mut_platit_za_smenu_1_cnt(monkeypatch):
    orig = core.portfolio
    monkeypatch.setattr(core, "portfolio", lambda pn, g, w, comm, we=None: orig(pn, g, w, comm, np.zeros_like(w)))
    assert not ok_etalon_act()


def test_s_limitom_ravnye_peresmotra_net():
    """С лимитом L равные веса = 1/L при любом n_open: пересмотра нет, на днях без входов и выходов оборот 0."""
    pn = panel_menyayushchiysya_act()
    r = core.assemble(pn, {**NET, "limit_pozitsiy": 3}, 0.0004)
    assert r["turnover"].iloc[1] == 0.0
