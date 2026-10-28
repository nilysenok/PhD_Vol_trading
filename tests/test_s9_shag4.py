"""Шаг 4 (v3): fastbt.shag4 на синтетике. Каждая подложенная ошибка (в памяти) обязана ловиться проверкой:
перепутанная σ в «одной точке», сдвиг σ на +1 (и −1), разрыв лестницы, инверсия маски открыт/закрыт."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from fastbt import shag4, shag4_gl3 as g3  # noqa: E402

IDX = pd.bdate_range("2018-01-01", "2021-12-31")


def _port(seed, drift=0.0):
    x = np.random.default_rng(seed).normal(drift, 0.01, len(IDX))
    return pd.DataFrame(dict(excess=x, total=x, exposure=1.0, turnover=0.0), index=IDX)


def _steps():
    return {s: _port(i, 0.0004 * i) for i, s in enumerate(shag4.STEPS)}


def test_lestnitsa_skladyvaetsya():
    rows = shag4.lestnitsa({0.0004: _steps(), 0.00035: _steps()}, 50)
    shag4.check_lestnitsa(rows)
    assert {r["comm_pct"] for r in rows} == {0.035, 0.04}
    for c in (0.035, 0.04):
        rs = [r for r in rows if r["comm_pct"] == c]
        assert len(rs) == len(shag4.STEPS) and all(f"znak_{y}" in rs[1] for y in shag4.YEARS)
        assert sum(r.get("d_sharpe", 0) for r in rs) == pytest.approx(rs[-1]["sharpe"] - rs[0]["sharpe"], abs=1e-12)


def test_lestnitsa_razryv_lovitsya():
    rows = shag4.lestnitsa({0.0004: _steps()}, 50)
    bad = [dict(r) for r in rows]
    bad[2]["d_sharpe"] += 0.05                         # ступень посчитана не к предыдущей
    with pytest.raises(AssertionError):
        shag4.check_lestnitsa(bad)
    wrong = [dict(r) for r in rows]
    wrong[-1]["d_k_A"] = wrong[1]["d_sharpe"]          # итог от другого ряда
    with pytest.raises(AssertionError):
        shag4.check_lestnitsa(wrong)


def _odna():
    P_l = {0.0004: {nm: _port(10 + i) for i, nm in enumerate(shag4.NAMES)}}
    P_o = {0.0004: {(nm, sg): _port(100 + 3 * i + j) for i, nm in enumerate(shag4.NAMES)
                    for j, sg in enumerate(shag4.SIGMAS)}}
    for nm, sg in zip(shag4.NAMES, shag4.SIGMAS):
        P_o[0.0004][(nm, sg)] = P_l[0.0004][nm]
    return P_l, P_o


def test_odna_tochka_perepytannaya_sigma():
    P_l, P_o = _odna()
    shag4.check_odna_tochka(P_o, P_l)
    d = P_o[0.0004]
    d[("C_hyb", "hyb")], d[("C_harj", "harj")] = d[("C_harj", "harj")], d[("C_hyb", "hyb")]
    with pytest.raises(AssertionError):
        shag4.check_odna_tochka(P_o, P_l)
    P_l, P_o = _odna()
    P_o[0.0004][("C_naive", "naive")] = None
    with pytest.raises(AssertionError):
        shag4.check_odna_tochka(P_o, P_l)


def _rv_sigma(seed=0, n=1000, tk=3):
    """RV — лог-AR(1); σ_d честная: сглаживание RV_d, RV_{d−1} + шум (знает день d, но не d+1)."""
    rng = np.random.default_rng(seed)
    lr = np.zeros((tk, n))
    for t in range(1, n):
        lr[:, t] = 0.8 * lr[:, t - 1] + rng.normal(0, 0.3, tk)
    RV = np.exp(lr) * 1e-4
    prev = np.concatenate([RV[:, :1], RV[:, :-1]], axis=1)
    S = np.sqrt((0.7 * RV + 0.3 * prev) * np.exp(rng.normal(0, 0.1, (tk, n))))
    return S, RV, np.ones((tk, n), bool)


def test_sdvig_sigma_plus_odin_lovitsya():
    S, RV, valid = _rv_sigma()
    g3.check_shift(g3.shift_profile(S, RV, valid))
    g3.check_shift(g3.shift_profile(S, RV, valid, diff=True))
    leak = g3.shifted(S, 1)                         # σ_d := σ_{d+1} — утечка будущего
    stale = g3.shifted(S, -1)                       # σ_d := σ_{d−1} — отставание
    for bad in (leak, stale):
        with pytest.raises(AssertionError):
            g3.check_shift(g3.shift_profile(bad, RV, valid))
    with pytest.raises(AssertionError):
        g3.check_shift(g3.shift_profile(leak, RV, valid, diff=True))


def test_qlike_sovpadaet_s_gl3():
    from d26_lib import qlike as ql3
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(-9, 0.5, 500))
    f1, f2 = y * np.exp(rng.normal(0, 0.3, 500)), y * np.exp(rng.normal(0, 0.1, 500))
    op, valid = rng.random(500) < 0.5, np.ones(500, bool)
    rows = {r["filtr"]: r for r in g3.qlike_split(y, f1, f2, op, valid)}
    assert rows["все"]["qlike_harj"] == pytest.approx(ql3(y, f1), abs=1e-12)
    assert rows["все"]["qlike_gibrid"] == pytest.approx(ql3(y, f2), abs=1e-12)
    assert rows["открыт"]["qlike_harj"] == pytest.approx(ql3(y[op], f1[op]), abs=1e-12)
    assert rows["закрыт"]["n"] + rows["открыт"]["n"] == rows["все"]["n"] == 500
    r = y / f1                                         # прямая формула
    assert rows["все"]["qlike_harj"] == pytest.approx(np.mean(r - np.log(r) - 1), abs=1e-12)


def test_otkryt_zakryt_razdeleny_verno():
    rng = np.random.default_rng(2)
    n = 4000
    G = np.ones((3, 1, n))
    G[:, 0, ::2] = 0.0                                  # чётные дни — фильтр закрыт во всех категориях
    op = g3.open_mask(G)[0]
    assert (~op[::2]).all() and op[1::2].all()
    y = np.exp(rng.normal(-9, 0.5, n))
    f_harj = y * np.exp(rng.normal(0, 0.4, n))
    f_hyb = np.where(op, f_harj, y * np.exp(rng.normal(0, 0.05, n)))   # выигрыш гибрида — только в закрытых днях
    valid = np.ones(n, bool)
    res = {r["filtr"]: r for r in g3.qlike_split(y, f_harj, f_hyb, op, valid)}
    assert res["открыт"]["vyigrysh"] == pytest.approx(0, abs=1e-12) and res["закрыт"]["vyigrysh"] > 0.05
    inv = {r["filtr"]: r for r in g3.qlike_split(y, f_harj, f_hyb, ~op, valid)}     # подложенная инверсия маски
    assert inv["открыт"]["vyigrysh"] > 0.05 and inv["закрыт"]["vyigrysh"] == pytest.approx(0, abs=1e-12)
    assert res["открыт"]["n"] == n // 2 and inv["открыт"]["n"] == n // 2


def test_exposure_open_p50_razdelyaet_dni():
    """На затворе p50 (тренд R>0.5, контртренд R<0.5, диапазон закрыт) open_mask ≡ «никогда» — ловится;
    exposure_open делит дни по тому, пропускает ли затвор позицию A."""
    n = 10
    R = np.tile(np.array([0.2, 0.8]), n // 2)[None, :]
    G = np.ones((3, 1, n))
    G[0] = R > 0.5
    G[1] = R < 0.5
    G[2] = (R >= 0.5) & (R <= 0.5)
    assert not g3.open_mask(G).any()                    # вырожденное определение
    tick, bar = np.zeros(2 * n, int), np.tile(np.arange(n), 2)
    cat = np.repeat([0, 2], n)                          # строка тренда и строка диапазона
    A = np.r_[np.ones(n), np.ones(n)]
    A[n + 5:] = 0.0                                     # у диапазонной строки позиция только в днях 0–4
    has, op = g3.exposure_open(A, G, tick, bar, cat, (1, n))
    assert has.all()                                    # в каждом дне есть позиция
    assert op[0].tolist() == [False, True] * 5          # пропускает только тренд на высоком ранге
    A[:n] = 0.0                                         # остался только диапазон (затвор закрыт)
    has, op = g3.exposure_open(A, G, tick, bar, cat, (1, n))
    assert has[0, :5].all() and not has[0, 5:].any() and not op.any()


def test_okno_tolko_2018_2021():
    n = 300
    yrs = np.repeat([2017, 2018, 2022], 100)[None, :]
    valid = g3.dev_mask(yrs)
    assert valid.sum() == 100 and valid[0, 100:200].all()
    y = np.full(n, 1e-4)
    rows = g3.qlike_split(y, y * 1.2, y * 1.1, np.ones(n, bool), valid[0])
    assert {r["filtr"]: r["n"] for r in rows}["все"] == 100


def test_share_diff_i_label():
    Ga, Gb = np.ones((3, 2, 10)), np.ones((3, 2, 10))
    Gb[1, 0, :4] = 0.0                                  # различие хотя бы в одной категории
    valid = np.ones((2, 10), bool)
    assert g3.share_diff(Ga, Gb, valid) == pytest.approx(4 / 20)
    assert g3.share_diff(Ga, Ga, valid) == 0.0
    assert shag4.parse_label("W126_p50") == (126, 0.5)
    with pytest.raises(ValueError):
        shag4.parse_label("p70")


def _p(x):
    return pd.DataFrame(dict(excess=x, total=x, exposure=1.0, turnover=0.0), index=IDX)


def test_znak_po_godam_i_comm_pct():
    base = np.random.default_rng(0).normal(0, 0.01, len(IDX))
    sg = np.where(IDX.year <= 2019, 0.002, -0.002)
    c = shag4.compare(_p(base + sg), _p(base), 30)
    assert [c[f"znak_{y}"] for y in shag4.YEARS] == ["+", "+", "−", "−"] and c["n_let_plyus"] == 2
    r = shag4.compare(_p(base), _p(base + sg), 30)
    assert [r[f"znak_{y}"] for y in shag4.YEARS] == ["−", "−", "+", "+"]
    P = {cm: {nm: _p(base + 0.0003 * i) for i, nm in enumerate(shag4.NAMES)} for cm in shag4.COMMS}
    L = shag4.luchshiy(P, 20)
    assert len(L) == 18 and {r["comm_pct"] for r in L} == {0.035, 0.04}
    assert sum("d_sharpe" in r for r in L) == 12 and all(r["pometka"] == shag4.POMETKA for r in L)
    O = shag4.odna_tochka({cm: {("C_naive", "naive"): _p(base), ("C_naive", "hyb"): None} for cm in shag4.COMMS},
                          {cm: _p(base - 0.001) for cm in shag4.COMMS}, 20)
    assert {r["comm_pct"] for r in O} == {0.035, 0.04}
    assert [("primechanie" in r) for r in O[:2]] == [False, True] and O[0]["d_sharpe"] > 0


def test_share_diff_valid_rezhet_vne_okna():
    Ga, Gb = np.ones((3, 1, 10)), np.ones((3, 1, 10))
    Gb[0, 0, 5:] = 0.0                                     # различия только в «не окне»
    valid = np.zeros((1, 10), bool)
    valid[0, :5] = True
    assert g3.share_diff(Ga, Gb, valid) == 0.0
    valid[0, 5:7] = True
    assert g3.share_diff(Ga, Gb, valid) == pytest.approx(2 / 7)
    assert np.isnan(g3.share_diff(Ga, Gb, np.zeros((1, 10), bool)))

