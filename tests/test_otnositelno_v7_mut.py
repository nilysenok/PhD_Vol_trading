"""Тестировщик PR #865: дополнительные мутации оси otnositelno (синтетика, известный ответ)."""
import numpy as np
import pandas as pd
import pytest

from fastbt import otnositelno_v7 as ot
from test_otnositelno_v7 import C, IM, METKI, T, mk_td

M3 = np.array([["X", "X", "X", ""]] * T, dtype=object)        # A,B,C — один сектор, D — без


def r(c):
    return c / c.shift(1) - 1


def test_tiker_ne_v_sobstvennom_fone_znanyj_otvet():
    fon, flag = ot.fon_ryad(C, IM, M3, "sektor")
    R = r(C)
    assert np.allclose(fon["A"].to_numpy()[1:], ((R["B"] + R["C"]) / 2).to_numpy()[1:])
    assert not np.allclose(fon["A"].to_numpy()[1:], ((R["A"] + R["B"] + R["C"]) / 3).to_numpy()[1:])
    assert not flag[["A", "B", "C"]].to_numpy().any() and flag["D"].iloc[1:].all()


def fon_mut_svoj(c, imoex, metki, rezhim):
    """Мутация: тикер входит в свой фон (среднее по всему сектору)."""
    R = r(c)
    m = pd.DataFrame({t: R[["A", "B", "C"]].mean(axis=1) for t in c.columns}, index=c.index)
    return m


def test_mutaciya_tiker_v_svoem_fone_lovitsya():
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    mut = fon_mut_svoj(C, IM, M3, "sektor")
    assert not np.allclose(fon["A"].to_numpy()[1:], mut["A"].to_numpy()[1:])


def test_propusk_kollegi_fon_po_ostavshimsya_bez_ffill():
    c = C.copy()
    c.iloc[100, c.columns.get_loc("B")] = np.nan
    fon, flag = ot.fon_ryad(c, IM, M3, "sektor")
    R = r(C)
    assert np.isclose(fon["A"].iloc[100], R["C"].iloc[100])           # только C: B отсутствует
    assert not flag["A"].iloc[100]                                    # коллега остался -> не IMOEX
    mut = r(c.ffill())                                                # мутация ffill: B с нулевой доходностью
    assert not np.isclose((mut["B"].iloc[100] + mut["C"].iloc[100]) / 2, fon["A"].iloc[100])
    # день 101: у B доходность по пропущенному close d-1 — нет (NaN), не пересчёт от d-2
    assert np.isclose(fon["A"].iloc[101], R["C"].iloc[101])


def test_pervyj_den_net_dohodnosti_i_flag_false():
    for rez in ("sektor", "imoex"):
        fon, flag = ot.fon_ryad(C, IM, M3, rez)
        assert np.isnan(fon.iloc[0]).all() and not flag.iloc[0].any()
    td, _ = mk_td(C)
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    rel = ot.td_rel(td, fon["A"].to_numpy())
    assert rel.close[0] == td.close[0] and rel.open[0] == td.open[0]


def test_ohlc_svojstva_sohranyayutsya():
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    td, _ = mk_td(C)
    rel = ot.td_rel(td, fon["A"].to_numpy())
    assert (rel.high >= np.maximum(rel.open, rel.close) - 1e-12).all()
    assert (rel.low <= np.minimum(rel.open, rel.close) + 1e-12).all()
    assert (rel.high >= rel.low).all()


def test_mutaciya_mnozhitel_ohl_ne_tot_lovitsya():
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    td, _ = mk_td(C)
    rel = ot.td_rel(td, fon["A"].to_numpy())
    m = rel.close / td.close
    # мутация: множитель предыдущего дня — ломает high >= max(open, close) на части дней
    bad = td.high * np.r_[1.0, m[:-1]]
    assert not (bad >= np.maximum(rel.open, rel.close) - 1e-12).all()
    assert np.allclose(rel.high, td.high * m, rtol=1e-12)


def test_signal_dnya_d_ne_zavisit_ot_fona_d_plus_1():
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    td, _ = mk_td(C)
    f2 = fon["A"].to_numpy().copy()
    f2[251:] += 0.01
    a, b = ot.td_rel(td, fon["A"].to_numpy()), ot.td_rel(td, f2)
    assert np.array_equal(a.close[:251], b.close[:251]) and not np.allclose(a.close[251:], b.close[251:])


def test_fon_s_sdvigom_plus_odin_menyaet_rezultat():
    fon, _ = ot.fon_ryad(C, IM, M3, "sektor")
    td, _ = mk_td(C)
    a = ot.td_rel(td, fon["A"].to_numpy())
    sh = fon["A"].shift(-1).fillna(0.0).to_numpy()
    assert not np.allclose(a.close, ot.td_rel(td, sh).close)


def test_otricatelnyj_mnozhitel_otkaz():
    td, _ = mk_td(C)
    fon = np.zeros(T)
    fon[5] = 2.0                                                      # 1 + r - 2 <= 0
    with pytest.raises(ValueError):
        ot.td_rel(td, fon)
