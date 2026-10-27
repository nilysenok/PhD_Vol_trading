"""Мутации PR 2 (сборки A): подложенная ошибка -> детектор из `test_sweep_a.py` / `test_sweep_a_run.py` обязан упасть."""
import numpy as np

from fastbt import sweep_a as sa
from fastbt import sweep_a_core as core
from fastbt.sweep_a import SynthPanelSource
from test_sweep_a import (ok_etalon, ok_fifo, ok_limit_soblyudaetsya, ok_obratnaya_vol_izvestnyy_otvet,
                          ok_peresmotr_vesov_platit, ok_sila_pri_ravenstve, ok_stavka_bez_bfill, ok_ves_bez_tekushchego_dnya,
                          ok_ves_limit_bez_tekushchego_dnya)
from test_sweep_a_run import ok_n36


def test_mut_1_okno_vesov_s_tekushchim_dnem(monkeypatch):
    def fw(pn, scheme):
        w = np.zeros(pn.pos.shape)
        for d in range(len(pn.dates)):
            idx = np.flatnonzero(pn.act[d])
            w[d, idx] = core.slot_weights(scheme, pn.lr[max(0, d - core.OKNO_VESOV):d + 1][:, idx])   # d + 1: текущий день
        return w
    monkeypatch.setattr(core, "full_weights", fw)
    assert not ok_ves_bez_tekushchego_dnya()


def test_mut_2_limit_ne_soblyudaetsya(monkeypatch):
    monkeypatch.setattr(core, "gate", lambda pn, limit, ks: pn.pos.copy())
    assert not ok_limit_soblyudaetsya()


def test_mut_3_net_komissii_vyhoda_na_poslednem_bare(monkeypatch):
    orig = core.portfolio

    def pf(pn, g, w, comm, we=None):
        df = orig(pn, g, w, comm, we)
        df.iloc[-1, 0] += (np.abs(g[-1]) * w[-1] * comm).sum()       # вернули плату за выход
        return df
    monkeypatch.setattr(core, "portfolio", pf)
    assert not ok_etalon()


def test_mut_4_komissiya_tolko_s_odnoy_storony(monkeypatch):
    orig = core.portfolio
    monkeypatch.setattr(core, "portfolio", lambda pn, g, w, comm, we=None: orig(pn, g, w, comm / 2, we))
    assert not ok_etalon()


def test_mut_5_sila_pri_ravenstve_naoborot(monkeypatch):
    orig = core.Panel

    def inv_gate(pn, limit, ks):
        pn2 = orig(pn.dates, pn.sid, pn.pos, pn.lr, pn.act, -pn.strength, pn.rfc)
        return real(pn2, limit, ks)
    real = core.gate
    monkeypatch.setattr(core, "gate", inv_gate)
    assert not ok_sila_pri_ravenstve()


def test_mut_6_pozdniy_vhod_otvergnutoy_sdelki(monkeypatch):
    """Без FIFO: слот отдаётся самому сильному сигналу дня, в том числе вместо держателя."""
    def gate(pn, limit, ks):
        g = np.zeros(pn.pos.shape)
        for d in range(len(pn.dates)):
            for p in sorted(np.flatnonzero(pn.pos[d]), key=lambda q: (-pn.strength[d, q], q))[:limit]:
                g[d, p] = pn.pos[d, p]
        return g
    monkeypatch.setattr(core, "gate", gate)
    assert not ok_fifo()


def test_mut_7_ves_volatilnost_vmesto_obratnoy(monkeypatch):
    orig = core.slot_weights

    def sw(scheme, hist):
        w = orig(scheme, hist)
        if scheme == "obratnaya_vol" and len(hist) >= core.MIN_ISTORIYA:
            w = hist.std(axis=0, ddof=1)
            w = w / w.sum()
        return w
    monkeypatch.setattr(core, "slot_weights", sw)
    assert not ok_obratnaya_vol_izvestnyy_otvet()


def test_mut_9_okno_vesov_s_limitom_sdvinuto_na_d_plus_1(monkeypatch):
    def weights(pn, scheme, g, limit, full=None):
        w = np.zeros(g.shape)
        for d in range(len(pn.dates)):
            idx = np.flatnonzero(g[d] != 0)
            if len(idx):
                w[d, idx] = core.slot_weights(scheme, pn.lr[max(0, d - core.OKNO_VESOV):d + 1][:, idx]) * len(idx) / limit
        return w
    monkeypatch.setattr(core, "weights", weights)
    assert not ok_ves_limit_bez_tekushchego_dnya()


def test_mut_10_peresmotr_vesov_besplaten(monkeypatch):
    orig = core.portfolio
    monkeypatch.setattr(core, "portfolio", lambda pn, g, w, comm: _no_reb(orig, pn, g, w, comm))
    assert not ok_peresmotr_vesov_platit()


def _no_reb(orig, pn, g, w, comm):
    """Комиссия на |Δw| не берётся: добавляем её обратно по формуле."""
    df = orig(pn, g, w, comm)
    z = np.zeros((1, g.shape[1]))
    dev = w - pn.act / np.maximum(pn.act.sum(axis=1, keepdims=True), 1)
    gp, devp = np.vstack([z, g[:-1]]), np.vstack([z, dev[:-1]])
    reb = np.where((g != 0) & (g == gp), np.abs(g) * np.abs(dev - devp), 0.0).sum(axis=1)
    df["excess"] += reb * comm
    df["turnover"] -= reb
    return df


def test_mut_11_stavka_s_bfill(monkeypatch):
    def rate_on(rate, dts):
        return rate.reindex(rate.index.union(dts)).sort_index().ffill().bfill().reindex(dts).values
    monkeypatch.setattr(sa, "rate_on", rate_on)
    assert not ok_stavka_bez_bfill()


def test_mut_8_sbornok_ne_36(monkeypatch):
    g = sa.load_grid()
    g["vesa"] = g["vesa"][:2]
    sw = sa.make_sweep(SynthPanelSource(2, ("S1", "S2"), seed=2))
    res = sa.run_sborki(sw, g)
    assert not ok_n36(res, sw)
