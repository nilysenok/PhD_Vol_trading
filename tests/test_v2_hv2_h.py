"""Ж-214, лист 2, V6 (HARQ-J): z-оценка качества измерения по дням s < t (мутант «день d+1» ловится), известный ответ HARQ-OLS и QLIKE-подгонки,
знак реакции b_dQ (мутант «перепутанный знак» ловится), таблица реакции, главный тест обрезки на строках H1/H4/H7. Синтетика, без данных и сети."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import har as H
from models_v2.s9 import hv2_eval as E, hv2_harq as HQ, hv2_world as W
from tests.test_v2_hv2_q import synth_world


def test_zscore_uses_only_days_before_t_known_answer():
    x = np.arange(100, dtype=float) ** 1.5
    z = W.zscore(x)
    past = x[:70]
    assert np.isnan(z[59]) and np.isclose(z[70], (x[70] - past.mean()) / past.std(ddof=1))
    x2 = x.copy()
    x2[71:] = 1e9
    assert z[70] == W.zscore(x2)[70]                         # будущее на z_t не влияет


def test_zscore_lead_mutant_differs():
    x = np.random.default_rng(1).normal(size=200)
    assert not np.allclose(W.zscore(x)[100:150], W.zscore(x, lead=1)[100:150])


def known_world(n=260, seed=6, bd=-0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    q = rng.normal(size=n)
    b = np.array([0.1, 0.45, bd, 0.3, 0.1, -0.05])
    lnrv = np.zeros(n)
    for t in range(n - 1):
        X[t, 0] = lnrv[t]                                   # ln RV_d строки t = ln RV_t
        lnrv[t + 1] = b[0] + (b[1] + b[2] * q[t]) * X[t, 0] + b[3] * X[t, 1] + b[4] * X[t, 2] + b[5] * X[t, 3]
    X[n - 1, 0] = lnrv[n - 1]
    p = H.Panel(pd.bdate_range("2018-01-01", periods=n).to_numpy("datetime64[ns]"), np.zeros(n, int), np.exp(lnrv), X)
    rq = np.exp(2 * q)
    return W.World(p, rq, np.ones(n), H.eligible(p)[-30:]), q, b


def test_harq_ols_recovers_known_answer_and_negative_sign():
    w, q, b = known_world()
    s2, B, npair = HQ.run(w, q)
    assert np.allclose(B[-1], b, atol=1e-6) and B[-1, 2] < 0
    assert (s2 > 0).all() and npair[-1] > npair[0]


def test_harq_qlike_recovers_same_known_answer():
    w, q, b = known_world()
    s2, B, _ = HQ.run(w, q, qlike=True)
    assert np.allclose(B[-1], b, atol=1e-4) and (s2 > 0).all()


def test_sign_mutant_caught(monkeypatch):
    w, q, b = known_world()
    orig = HQ.design
    monkeypatch.setattr(HQ, "design", lambda ww, qq: orig(ww, -qq))   # МУТАНТ: перепутанный знак q
    _, B, _ = HQ.run(w, q)
    assert B[-1, 2] > 0                                              # знак b_dQ перевернулся: тест знака b_dQ < 0 его ловит


def test_no_defined_q_raises():
    w, _, _ = known_world()
    with pytest.raises(ValueError):
        HQ.run(w, np.full(len(w.p.rv), np.nan))


def test_reaction_table_known_values():
    w, q, b = known_world()
    s2, B, _ = HQ.run(w, q)
    t = HQ.reaction("H1", w, B)
    r = t.iloc[-1]
    assert np.isclose(r["R_q_0"], 0.45, atol=1e-6) and np.isclose(r["R_q_minus1"], 0.60, atol=1e-6) and np.isclose(r["R_q_plus1"], 0.30, atol=1e-6)
    assert (t["sign_b_dQ"] == "−").all() and t["n_days"].sum() == len(w.el)


RUNS = {"H1": lambda w: HQ.run(w, HQ.q_series(w, "rq")), "H4": lambda w: HQ.run(w, HQ.q_series(w, "rel")),
        "H7": lambda w: HQ.run(w, HQ.q_series(w, "rq"), qlike=True)}


@pytest.mark.parametrize("name", list(RUNS))
def test_main_truncation_h_rows_unchanged_by_corrupting_future(name):
    w = synth_world(n=360)
    i0 = int(w.el[25])
    assert E.trunc_diff(RUNS[name], w, i0) == 0.0


def test_truncation_catches_q_with_day_d_plus_1():
    w = synth_world(n=360)
    i0 = int(w.el[25])
    assert E.trunc_diff(lambda x: HQ.run(x, HQ.q_series(x, "rq", lead=1)), w, i0) > 1e-9      # МУТАНТ: q_t с RQ дня t+1
