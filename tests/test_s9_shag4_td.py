"""4.D по тикер-дням (Ъ 27.10 15:10, (в)): разбиение по рангу σ̂, знак выигрыша, DM, портфельные определения.
Синтетика; данные не читаются. python3 -m pytest tests/test_s9_shag4_td.py -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts" / "analysis"))
from d16_eval import nw_t  # noqa: E402

from fastbt import shag4_td as td  # noqa: E402


def test_open_by_rank_porog_strogo_nizhe():
    R = np.array([[0.1, 0.4999, 0.5, 0.5001, 0.9, np.nan]])
    defined, op = td.open_by_rank(R)
    assert op.tolist() == [[True, True, False, False, False, False]]          # R = 0,5 — закрыт
    assert defined.tolist() == [[True] * 5 + [False]]


def test_mutatsii_porog_i_neravenstvo_lovyatsya():
    R = np.array([[0.3, 0.47, 0.5, 0.53, 0.6]])
    _, op = td.open_by_rank(R)
    assert not np.array_equal(op, R <= 0.5)                                     # «≤» / «≥» вместо «<» — другая маска
    assert not np.array_equal(op, R < 0.55) and not np.array_equal(op, R < 0.45)
    assert td.THRESH == 0.5


def test_rank_matrix_sovpadaet_s_zatvorom_kontrtrenda():
    from fastbt.s9 import _c_gate
    rng = np.random.default_rng(0)
    S = np.exp(np.cumsum(rng.normal(0, 0.1, (2, 400)), axis=1))
    R = td.rank_matrix(S, [400, 400], 252)
    G = _c_gate(R, 0.5)
    defined, op = td.open_by_rank(R)
    assert (G[1][defined] == op[defined]).all()                                 # при p50 G[1] = «R < 0,5»
    assert (G[0][defined & (R != 0.5)] == (~op)[defined & (R != 0.5)]).all()    # затвор тренда — дополнение


def _data(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    y = np.exp(rng.normal(-9, 0.5, (2, n)))
    f_base = y * np.exp(rng.normal(0, 0.4, (2, n)))
    op = rng.random((2, n)) < 0.4
    f_hyb = np.where(op, y * np.exp(rng.normal(0, 0.05, (2, n))), f_base)   # гибрид лучше только в открытых
    days = np.tile(np.arange(n), (2, 1))
    years = np.tile(np.repeat([2018, 2019, 2020, 2021], n // 4), (2, 1))
    return y, f_base, f_hyb, op, days, years


def _by(rows, per="2018–2021"):
    return {r["gruppa"]: r for r in rows if r["period"] == per}


def test_znak_vyigrysha_i_summa_grupp():
    y, fb, fh, op, days, years = _data()
    valid = np.ones_like(op)
    rows = td.qlike_td(y, fb, fh, {"открыт": valid & op, "закрыт": valid & ~op}, days, years, 5)
    q = _by(rows)
    assert q["открыт"]["vyigrysh"] > 0.05 and q["закрыт"]["vyigrysh"] == pytest.approx(0, abs=1e-12)
    assert q["открыт"]["otn_vyigrysh_pct"] > 10 and q["открыт"]["p_dm"] < 1e-6
    assert q["открыт"]["n"] + q["закрыт"]["n"] == q["все"]["n"] == y.size
    assert q["открыт"]["dolya"] + q["закрыт"]["dolya"] == pytest.approx(1)
    swapped = td.qlike_td(y, fh, fb, {"открыт": op, "закрыт": ~op}, days, years, 5)       # мутация: база и гибрид наоборот
    assert _by(swapped)["открыт"]["vyigrysh"] < -0.05
    ys = sum(_by(rows, str(p))["все"]["n"] for p in (2018, 2019, 2020, 2021))
    assert ys == q["все"]["n"]


def test_nan_celi_ischezaet_iz_grupp_i_schitaetsya():
    y, fb, fh, op, days, years = _data(400)
    y[:, -5:] = np.nan
    rows = td.qlike_td(y, fb, fh, {"открыт": op, "закрыт": ~op}, days, years, 5)
    q = _by(rows)
    assert q["все"]["n"] == 2 * 395 and q["все"]["n_nan_celi"] == 10
    assert q["открыт"]["n"] + q["закрыт"]["n"] == q["все"]["n"]


def test_2022_ne_popadaet_v_periody():
    y, fb, fh, op, days, years = _data(400)
    years[:, -100:] = 2022
    q = _by(td.qlike_td(y, fb, fh, {"открыт": op, "закрыт": ~op}, days, years, 5))
    assert q["все"]["n"] == 2 * 300


def test_dm_sovpadaet_s_gl3_i_mala_gruppa():
    rng = np.random.default_rng(5)
    d = rng.normal(0.1, 1, (3, 300))
    days = np.tile(np.arange(300), (3, 1))
    p, nd = td.dm_nw(days.ravel(), d.ravel(), 4)
    t = nw_t(d.mean(axis=0), 4)
    assert nd == 300 and p == pytest.approx(__import__("math").erfc(abs(t) / 2 ** 0.5), rel=1e-9)
    p2, n2 = td.dm_nw(days[:, :10].ravel(), d[:, :10].ravel(), 4)
    assert np.isnan(p2) and n2 == 10                                           # мало дней — p не считается
    assert td.dm_nw(days.ravel(), d.ravel(), 0)[0] != p                         # лаг h−1 действует


def test_ravno_050_ucheno():
    y, fb, fh, op, days, years = _data(400)
    half = np.zeros_like(op)
    half[0, :7] = True
    q = _by(td.qlike_td(y, fb, fh, {"открыт": op & ~half, "закрыт": ~op | half}, days, years, 1, half=half))
    assert q["закрыт"]["n_R_050"] == 7 and q["открыт"]["n_R_050"] == 0


def test_portfelnoe_ne_ravno_tikernomu():
    shape, n = (2, 6), 6
    tick = np.array([0, 0, 0, 1, 1, 1])
    bar = np.array([0, 0, 1, 0, 1, 1])
    cat = np.array([0, 1, 0, 0, 0, 1])
    A = np.array([1.0, 1, 1, 1, 0, 1])
    G = np.ones((3, *shape))
    G[0, 0, 0] = 0                                                              # тикер 0, день 0: тренд закрыт, контртренд открыт
    G[1, 1, 1] = 0
    valid = np.ones(shape, bool)
    gr = td.portfolio_groups(A, G, tick, bar, cat, shape, valid)
    assert gr["хотя бы одна позиция A"]["открыт"][0, 0] and not gr["все позиции против ни одной"]["открыт"][0, 0]
    assert not gr["затвор тренда"]["открыт"][0, 0]
    R = np.array([[0.2, 0.7, 0.9, 0.1, 0.5, 0.3]] * 2)                           # тикерное определение — по рангу
    _, op = td.open_by_rank(R)
    assert not np.array_equal(op, gr["хотя бы одна позиция A"]["открыт"])        # мутация «портфельное вместо тикерного» видна
