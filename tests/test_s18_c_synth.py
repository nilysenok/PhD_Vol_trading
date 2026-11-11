"""S-C-DATA, синтетика: тождество разложения, известный ответ, PIT нагрузок, мутации (mutations.md № 1, 8, 142, 167, 168)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import c_decomp as DC, c_factor as CF
from models_v2.s18.c_data import Inputs

ND, NB, NP = 160, 20, 16


def make(seed=0, nd=ND) -> Inputs:
    """Синтетика: 16 бумаг = нагрузка·общий фактор + шум; SBER = β·фактор + собственный шум; rv — сумма квадратов ×1e4."""
    g = np.random.default_rng(seed)
    n = nd * NB
    f = g.normal(size=n)
    load = g.uniform(0.5, 1.5, NP)
    R = np.outer(f, load) + 0.7 * g.normal(size=(n, NP))
    r = 0.8 * f + 0.5 * g.normal(size=n)
    day = np.repeat(np.arange(nd), NB)
    off = np.arange(nd + 1) * NB
    rv = np.add.reduceat(r ** 2, off[:-1]) * 1e4
    days = pd.bdate_range("2020-01-01", periods=nd)
    return Inputs(days, rv, np.full(nd, NB), off, day, np.tile(np.arange(NB), nd), r, R, np.full(n, np.nan), {})


def corrupt(inp: Inputs, from_day: int, seed=9, what="RR") -> Inputs:
    """Копия, где бары сессий >= from_day заменены мусором (what: R — бумаги, r — SBER)."""
    g = np.random.default_rng(seed)
    a = inp.off[from_day]
    R, r, rv = inp.R.copy(), inp.r.copy(), inp.rv.copy()
    if "R" in what:
        R[a:] = 5 * g.normal(size=R[a:].shape)
    if "r" in what:
        r[a:] = 5 * g.normal(size=r[a:].shape)
        rv = np.add.reduceat(r ** 2, inp.off[:-1]) * 1e4
    return Inputs(inp.days, rv, inp.nbars, inp.off, inp.day, inp.slot, r, R, inp.ri, inp.stats)


def test_parts_known_answer_and_orthogonality():
    g = np.random.default_rng(1)
    F = g.normal(size=(50, 2))
    b = np.array([1.5, -0.7])
    e = g.normal(size=50)
    e -= F @ np.linalg.lstsq(F, e, rcond=None)[0]
    p = DC.parts(F @ b + e, F)
    assert p["common"] == pytest.approx(np.sum((F @ b) ** 2), rel=1e-12)
    assert p["specific"] == pytest.approx(np.sum(e ** 2), rel=1e-12)
    assert abs(p["cross"]) < 1e-12 and p["unc"] == 0.0


def test_identity_every_session_and_both_k():
    df = DC.run(make(), range(60, 100))
    assert (df["ident_err"].abs() / df["rv"] < 1e-13).all() and (df["ident_err_pit"].abs() / df["rv"] < 1e-13).all()   # относительно: rv ~ 1e5 %²
    assert (df["cross"].abs() / df["rv"] < 1e-13).all()
    assert (df["rv_common"] > 0).all() and (df["rv_specific"] > 0).all()


def test_pit_variant_cross_term_does_not_vanish_but_identity_holds():
    df = DC.run(make(), range(60, 100))
    assert df["cross_pit"].abs().max() > 1e-3            # перекрёстный член PIT-варианта не нулевой
    assert (df["ident_err_pit"].abs() / df["rv"] < 1e-13).all()      # а сумма с ним сходится


def test_uncovered_bars_and_empty_tape():
    inp = make()
    R = inp.R.copy()
    R[inp.off[70]:inp.off[70] + 5] = np.nan               # 5 баров сессии 70 без бумаг
    R[inp.off[71]:inp.off[72]] = np.nan                   # вся сессия 71 без бумаг: пустая лента
    df = DC.run(Inputs(inp.days, inp.rv, inp.nbars, inp.off, inp.day, inp.slot, inp.r, R, inp.ri, {}), [70, 71])
    a, b = df[df["d"] == inp.days[70]].iloc[0], df[df["d"] == inp.days[71]].iloc[0]
    assert a["n_cov"] == NB - 5 and a["rv_uncovered"] == pytest.approx(1e4 * np.sum(inp.r[inp.off[70]:inp.off[70] + 5] ** 2), rel=1e-12)
    assert b["n_cov"] == 0 and b["rv_uncovered"] == pytest.approx(b["rv"], rel=1e-12) and b["rv_common"] == 0 and b["flag_fewcov"]
    assert abs(a["ident_err"]) / a["rv"] < 1e-13 and abs(b["ident_err"]) / b["rv"] < 1e-13


def test_mutation_dropped_bar_breaks_identity():
    inp = make()
    r = np.delete(inp.r, inp.off[80] + 3)                 # бар выброшен из r, rv — прежняя
    off = inp.off.copy()
    off[81:] -= 1
    m = Inputs(inp.days, inp.rv, inp.nbars, off, np.delete(inp.day, inp.off[80] + 3), np.delete(inp.slot, inp.off[80] + 3), r,
               np.delete(inp.R, inp.off[80] + 3, 0), inp.ri, {})
    assert abs(DC.session_rows(m, 80)[0]["ident_err"]) > 1e-3


def test_mutation_next_session_bar_in_session_s_caught_and_honest_ignores_it():
    inp = make()
    base = DC.session_rows(inp, 90)
    ch = corrupt(inp, 91)
    assert DC.session_rows(ch, 90) == base                # будущие сессии сессию 90 не меняют
    off = inp.off.copy()
    off[91] += 1                                           # мутация: первый бар сессии 91 попал в сессию 90
    mut = Inputs(inp.days, inp.rv, inp.nbars, off, inp.day, inp.slot, inp.r, inp.R, inp.ri, {})
    assert abs(DC.session_rows(mut, 90)[0]["ident_err"]) > 1e-3


def test_pit_loadings_ignore_future_bars_bitwise():
    inp, s = make(), 100
    lo, hi, _ = CF.window(inp.off, s)
    a, b = CF.loadings(inp.R[lo:hi]), CF.loadings(corrupt(inp, s).R[lo:hi])
    assert np.array_equal(a[0], b[0], equal_nan=True) and np.array_equal(a[1], b[1])
    f1 = CF.scores(inp.R[inp.off[s]:inp.off[s + 1]], *a)
    f2 = CF.scores(corrupt(inp, s + 1).R[inp.off[s]:inp.off[s + 1]], *a)
    assert np.array_equal(f1, f2, equal_nan=True)


def test_mutation_window_with_current_session_is_caught(monkeypatch):
    inp, s = make(), 100
    ch = corrupt(inp, s, what="R")
    ok = lambda i: CF.loadings(i.R[slice(*CF.window(i.off, s)[:2])])[1]  # noqa: E731
    assert np.array_equal(ok(inp), ok(ch))
    monkeypatch.setattr(CF, "LAG", 0)                      # окно захватило сессию s
    assert not np.array_equal(ok(inp), ok(ch))


def test_row_of_session_s_ignores_all_later_bars_pit_variant_too():
    inp = make()
    rows = DC.session_rows(inp, 110)
    assert DC.session_rows(corrupt(inp, 111, what="Rr"), 110) == rows


def test_sber_not_in_factor_and_mutation_caught():
    inp, s = make(), 105
    a, b = int(inp.off[s]), int(inp.off[s + 1])

    def fs(i):
        lo, hi, _ = CF.window(i.off, s)
        sig, W = CF.loadings(i.R[lo:hi])
        return CF.scores(i.R[a:b], sig, W)

    ch = corrupt(inp, s, what="r")                         # собственные доходности SBER сессий >= s испорчены
    assert np.array_equal(fs(inp), fs(Inputs(ch.days, ch.rv, ch.nbars, ch.off, ch.day, ch.slot, ch.r, inp.R, ch.ri, {})), equal_nan=True)
    mut = lambda i: Inputs(i.days, i.rv, i.nbars, i.off, i.day, i.slot, i.r, np.column_stack([i.R[:, 1:], i.r]), i.ri, {})  # SBER — в пул
    assert not np.array_equal(fs(mut(inp)), fs(mut(Inputs(ch.days, ch.rv, ch.nbars, ch.off, ch.day, ch.slot, ch.r, inp.R, ch.ri, {}))), equal_nan=True)
