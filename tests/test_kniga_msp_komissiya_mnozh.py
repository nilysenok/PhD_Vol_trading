"""Комиссия книги МСП (#857, Ж-149, #861): ex_kniga = u·R_brutto − ставка·Σ|u|·T − ставка·|Δ хеджа|. Синтетика."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kniga_msp as km
from fastbt import msp_v5, tf_os_v5

C = 0.0004
T, TIK = 60, ["A", "B"]
IDX = pd.bdate_range("2016-01-04", periods=T)


def test_stavka_ta_zhe_chto_v_dvizhke():
    assert km.KOMISSIYA == tf_os_v5.KOMISSIYA == C


def test_komissiya_oborota_izvestnyj_otvet():
    """C·Σ|Δw| руками: |0,5−0|+|−1−(−0,5)|+|0,2−0,2| = 1,0."""
    assert km.komissiya_oborota([0.5, -1.0, 0.2], [0.0, -0.5, 0.2]) == pytest.approx(C * 1.0)
    assert km.komissiya_oborota([-0.4], [-0.1]) == pytest.approx(C * 0.3)


def test_mutaciya_modul_w_vmesto_delta_lovitsya():
    w = [0.5, -1.0]
    assert km.komissiya_oborota(w, w) == 0.0
    assert km.komissiya_oborota(w, w) != pytest.approx(C * np.abs(w).sum())
    assert km.komissiya_oborota([1.0], [0.0]) == pytest.approx(C)


def _f(v):
    return pd.DataFrame(v, index=IDX, columns=TIK)


def _sborshchik(seed=11):
    """(в) Сборщик R_brutto/T по tf_os_v5.dohod: единица лонга, один бар в день -> вход и выход каждый день."""
    rng = np.random.default_rng(seed)
    o = np.full((T, 2), 100.0)
    c = o * (1 + 0.01 * rng.standard_normal((T, 2)))
    rb, tt = np.zeros((T, 2)), np.zeros((T, 2))
    for j in range(2):
        g, _, turn = tf_os_v5.dohod(np.ones(T), o[:, j], c[:, j], np.arange(T))
        rb[:, j], tt[:, j] = g, turn
    return rb, tt, rng


def test_v_sborshchik_na_trekh_dnyah_izvestnyj_otvet():
    o, c = np.full(3, 100.0), np.array([101.0, 99.0, 100.0])
    g, cost, turn = tf_os_v5.dohod(np.ones(3), o, c, np.arange(3))
    assert g.tolist() == pytest.approx([0.01, -0.01, 0.0]) and turn.tolist() == [2, 2, 2]
    assert cost.tolist() == pytest.approx([2 * C] * 3)


def _progon(rb, tt, rng, storona=(1.0, -1.0), im=None):
    sig = {"S1": {"storona": _f(np.array([list(storona)] * T)), "sila": _f(rng.uniform(0, 1, (T, 2))),
                  "g": _f(np.ones((T, 2)))}}
    imoex = pd.Series(0.0, index=IDX) if im is None else im
    return msp_v5.progon(sig, _f(np.full((T, 2), 0.02)), {"A": "X", "B": "Y"},
                         {"R_brutto": _f(rb), "T": _f(tt)}, imoex, 0.0, k=2, rho=0.001)


def test_a_short_i_long_platyat_odinakovo():
    """R_brutto = 0: ex_kniga = −C·Σ|u|·T. Лонг и шорт равного размера платят одинаково; мутант «u @ R с комиссией
    внутри» (R = −C·T) дал бы Σu = 0 для сбалансированной книги, т. е. возврат шорту."""
    rb, tt, rng = _sborshchik()
    r = _progon(np.zeros((T, 2)), tt, rng)
    u = r.u.to_numpy()
    ex = r.ex_kniga.to_numpy()
    assert np.allclose(ex, -C * (np.abs(u) * tt).sum(axis=1)) and (ex[-1] < 0)
    assert np.allclose(u[-1, 0], -u[-1, 1]) and u[-1, 0] != 0
    mutant = (u * -(C * tt)).sum(axis=1)
    assert abs(mutant[-1]) < 1e-12 and ex[-1] < mutant[-1]


def test_b_den_tolko_mnozhitel_spisyvaet_stavku_na_novyj_ves_i_T():
    """Тест владельца (Ж-149) через msp_v5: сигналы все дни те же, меняется только множитель; в каждый день
    ex_kniga − u·R_brutto = −C·Σ|u_новый|·T (через оборот дня, не через Δw)."""
    rb, tt, rng = _sborshchik()
    r = _progon(rb, tt, rng)
    u, ex = r.u.to_numpy(), r.ex_kniga.to_numpy()
    assert np.allclose(ex - (u * rb).sum(axis=1), -C * (np.abs(u) * tt).sum(axis=1))
    m = np.abs(u).sum(axis=1)
    assert (np.abs(np.diff(m[30:])) > 1e-9).any()          # множитель менялся между днями без смены сигналов


def test_staryj_kontrakt_otkaz_s_prichinoj():
    rb, tt, rng = _sborshchik()
    with pytest.raises(ValueError, match="возврат комиссии"):
        _progon_staryj = msp_v5.progon(
            {"S1": {"storona": _f(np.ones((T, 2))), "sila": _f(np.ones((T, 2))), "g": _f(np.ones((T, 2)))}},
            _f(np.full((T, 2), 0.02)), {"A": "X", "B": "Y"}, _f(rb), pd.Series(0.0, index=IDX), 0.0)


def _ist(T=40):
    return {"ex_sk": np.zeros((T, 2)), "ex_port": np.zeros(T), "ex_kniga": np.zeros(T), "imoex": np.zeros(T),
            "kvartal": np.zeros(T, int)}


ARGS = (np.array([5, 4, 3, -2, -1, 0.5]), np.full(6, 0.02), np.arange(6), np.array([0, 0, 0, 1, 1, 1]))


def test_kniga_dnya_komissiya_tolko_hedzh():
    """Множитель акций komissiya не даёт (она в T); Itog.komissiya — только хедж."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    assert b.m != a.m and not np.allclose(a.u, b.u) and b.komissiya == 0.0


def test_kniga_dnya_hedzh_platit_po_izmeneniyu_bety():
    rng = np.random.default_rng(5)
    n = 90
    ist = _ist(n)
    ist["imoex"] = rng.normal(0, 0.01, n)
    ist["ex_kniga"] = 1.5 * ist["imoex"] + rng.normal(0, 0.002, n)
    ist["ex_kniga"][:64] = 0.5 * ist["imoex"][:64]
    a = km.kniga_dnya(70, *ARGS, ist, k=6)
    b = km.kniga_dnya(71, *ARGS, ist, k=6)
    assert a.hedzh != b.hedzh
    assert b.komissiya == pytest.approx(C * abs(b.hedzh - a.hedzh)) and b.komissiya > 0
