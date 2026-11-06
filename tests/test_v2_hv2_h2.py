"""Ж-214, лист 2, V6 (остаток): причинность винзоризации и порога стресса (мутант «порог по всей выборке» ловится), известный ответ расширений H2/H3/H6 и
BPQ в уровнях H9 (реакция >= 0 на историческом диапазоне √RQ), главный тест обрезки на строках H2/H3/H5/H6/H8/H9/H10. Синтетика, без данных и сети."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s9 import hv2_eval as E, hv2_harq2 as HR, hv2_hstage2 as S2, hv2_world as W
from tests.test_v2_hv2_q import synth_world


def with_mkt(w, seed=9):
    return dataclasses.replace(w, mkt=np.exp(np.random.default_rng(seed).normal(1, 0.5, len(w.p.rv))))


def test_winsor_known_answer_and_causal():
    x = np.arange(200, dtype=float)
    o = HR.winsor(x, 1, 99)
    assert np.isclose(o[150], np.percentile(x[:150], 99)) and o[10] == x[10] and (o[:60] == x[:60]).all()
    y = x.copy()
    y[151:] = 1e6
    assert HR.winsor(y, 1, 99)[150] == o[150]


def test_stress_causal_and_full_sample_mutant_differs():
    m = np.exp(np.random.default_rng(1).normal(size=300))
    s = HR.stress(m, 80)
    m2 = m.copy()
    m2[201:] *= 50
    assert (HR.stress(m2, 80)[:201] == s[:201]).all() and (s[:60] == 0).all()
    assert (HR.stress(m, 80, full=True)[60:200] != s[60:200]).any()          # МУТАНТ: порог по всей выборке


def test_qbar5_known_answer():
    q = np.arange(10, dtype=float)
    assert np.allclose(HR.qbar5(q)[4:], [2, 3, 4, 5, 6, 7]) and np.isnan(HR.qbar5(q)[3])


@pytest.mark.parametrize("spec", [{"q": "rq", "react": ("J",)}, {"q": "rq", "react": ("w5",)}, {"q": "rq", "stress": 80}, {"q": "rq", "react": ("J", "w5"), "stress": 70}])
def test_extended_ols_recovers_known_answer(spec):
    rng = np.random.default_rng(8)
    n = 330
    X = rng.normal(size=(n, 4))
    p = H.Panel(pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]"), np.zeros(n, int), np.ones(n), X)
    w = W.World(p, np.exp(rng.normal(size=n)), np.ones(n), H.eligible(p)[-30:], np.exp(rng.normal(size=n)))
    Xd, names = HR.design(w, spec)
    b = np.r_[0.2, rng.uniform(-0.5, 0.5, Xd.shape[1])]
    ok = np.isfinite(Xd).all(axis=1)
    rv = np.ones(n)
    rv[1:] = np.exp(b[0] + np.einsum("ij,j->i", np.where(ok[:-1, None], Xd[:-1], 0.0), b[1:]))
    w = dataclasses.replace(w, p=H.Panel(p.dates, p.seg, rv, X))
    s2, B, nm, npair = HR.run(w, spec)
    assert nm == ["b0"] + names and np.allclose(B[-1], b, atol=1e-6) and (s2 > 0).all()


def levels_world(r_fn, n=330, seed=11):
    rng = np.random.default_rng(seed)
    X = rng.normal(1.0, 0.1, size=(n, 4))
    X[:, 3] = np.log1p(rng.uniform(0, 0.5, n))
    x = rng.uniform(0.5, 3.0, n)
    rv, rvw, rvm, jj = np.ones(n), np.exp(X[:, 1]), np.exp(X[:, 2]), np.expm1(X[:, 3])
    for t in range(n - 1):
        rv[t + 1] = 0.3 + r_fn(x[t]) * rv[t] + 0.3 * rvw[t] + 0.3 * rvm[t] + 0.2 * jj[t]
    X[:, 0] = np.log(rv)
    p = H.Panel(pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]"), np.zeros(n, int), rv, X)
    return W.World(p, x ** 2, np.ones(n), H.eligible(p)[-20:])


def test_h9_recovers_nonneg_reaction_and_binds_when_true_is_negative():
    s2, B, ab = HR.run_h9(levels_world(lambda x: 0.9 - 0.05 * x))
    a, b = ab[-1]
    assert np.isclose(B[-1, 1], 0.9 - 0.05 * a, atol=5e-3) and np.isclose(B[-1, 2], 0.9 - 0.05 * b, atol=5e-3) and (s2 > 0).all()
    s2n, Bn, abn = HR.run_h9(levels_world(lambda x: 0.8 - 0.4 * x))      # r(3) < 0: нужна граница
    assert (Bn[:, 1:3] >= 0).all() and (Bn[:, 2] <= 1e-9).all() and Bn[-1, 1] > 0.1 and (s2n > 0).all()


def test_kinds_and_neighbors_cover_registered_names():
    assert set(S2.neighbors("H5")) == {"H10", "H11"} and S2.neighbors("H5")["H10"]["wins"] == (0.5, 99.5)
    assert S2.neighbors("H6")["H11"]["stress"] == 90 and S2.neighbors("H1")["H10"]["qwin"] == 504 and S2.neighbors("H9")["H11"]["qwin"] == 1008


RUNS = {n: (lambda x, s=s: HR.run(x, s)) for n, s in {**S2.SPEC, "H10": S2.neighbors("H1")["H10"], "H5w": {"q": "rq", "wins": (0.5, 99.5)}}.items()}
RUNS["H9"] = lambda x: HR.run_h9(x)


@pytest.mark.parametrize("name", list(RUNS))
def test_main_truncation_h_rest_rows_unchanged_by_corrupting_future(name):
    w = with_mkt(synth_world(n=360))
    assert E.trunc_diff(RUNS[name], w, int(w.el[25])) == 0.0


def test_truncation_catches_full_sample_stress_threshold(monkeypatch):
    w = with_mkt(synth_world(n=360))
    orig = HR.stress
    monkeypatch.setattr(HR, "stress", lambda m, pct, min_obs=60, full=False: orig(m, pct, min_obs, full=True))
    assert E.trunc_diff(RUNS["H6"], w, int(w.el[25])) > 1e-9
