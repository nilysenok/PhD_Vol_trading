"""E3 (X3): PIT — окно g_a ровно 250 закрытых пар, пара «закрыто + 1» и CUSUM по будущим парам ловятся (мутации через боевой путь)."""
import numpy as np
import pandas as pd

from models_v2.s15 import de_e3_core as C
from tests.test_v2_de_e3_core import synth


def _plus_one(tp, d0):
    return np.flatnonzero(np.asarray(tp, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(1, "D") * 3)       # МУТАНТ: пара с t⁺ = d0 + 1 торговый день «закрыта»


def _cut(d):
    pos = [p for p in C.refit_positions(d) if d[p] > pd.Timestamp("2018-08-01")]
    return pos[0], pos[1]


def test_window_is_exactly_250_last_closed_pairs_and_plus_one_is_caught(monkeypatch):
    d, tp, F0, rv, Z = synth()
    a, _ = _cut(d)
    r = C.refit(F0, rv, tp, Z, d[a])
    assert r["n_rows"] == C.WIN == 250 == len(r["rows"])
    assert (tp[r["rows"]] <= np.datetime64(d[a])).all()
    assert np.array_equal(r["rows"], np.flatnonzero(tp <= np.datetime64(d[a]))[-250:])
    monkeypatch.setattr(C, "closed_rows", _plus_one)
    bad = C.refit(F0, rv, tp, Z, d[a])
    assert (tp[bad["rows"]] > np.datetime64(d[a])).any()                              # мутант берёт пару t⁺ > d0 — тест окна его ловит


def test_future_pairs_do_not_change_ga_honest_path_and_plus_one_mutation_changes_it(monkeypatch):
    d, tp, F0, rv, Z = synth()
    a, b = _cut(d)
    rv2 = rv.copy()
    rv2[a:] *= 1000.0                                                                  # пары j ≥ a: t⁺_j > d0 = d[a], не закрыты
    w1, w2 = C.walk_forward_ga(F0, rv, tp, d, Z), C.walk_forward_ga(F0, rv2, tp, d, Z)
    assert np.array_equal(w1["ga"][:b], w2["ga"][:b])
    monkeypatch.setattr(C, "closed_rows", _plus_one)
    m1, m2 = C.walk_forward_ga(F0, rv, tp, d, Z), C.walk_forward_ga(F0, rv2, tp, d, Z)
    assert not np.array_equal(m1["ga"][:b], m2["ga"][:b])


def test_cusum_uses_only_closed_pairs_and_future_mutation_is_caught(monkeypatch):
    d, tp, F0, rv, Z = synth()
    w = C.walk_forward_ga(F0, rv, tp, d, Z)
    i = 500
    rv2 = rv.copy()
    rv2[i:] *= 1000.0                                                                  # пары j ≥ i не закрыты на дне d[i]
    a1 = C.adaptive_weights(rv, F0, w["ga"], w["fitted"], tp, d)
    a2 = C.adaptive_weights(rv2, F0, w["ga"], w["fitted"], tp, d)
    assert np.array_equal(a1[:i + 1], a2[:i + 1]) and a1[i] > 0
    monkeypatch.setattr(C, "closed_rows", _plus_one)
    m1, m2 = (C.adaptive_weights(r, F0, w["ga"], w["fitted"], tp, d) for r in (rv, rv2))
    assert not np.array_equal(m1[:i + 1], m2[:i + 1])


def test_cusum_window_is_last_250_closed_pairs_with_ga():
    d, tp, F0, rv, Z = synth()
    w = C.walk_forward_ga(F0, rv, tp, d, Z)
    i = 600
    rv2 = rv.copy()
    first = np.flatnonzero(w["fitted"] & (tp <= np.datetime64(d[i])))[-C.WIN:][0]
    rv2[:first] *= 1000.0                                                              # до окна CUSUM — на a_t влиять не должно
    a1 = C.adaptive_weights(rv, F0, w["ga"], w["fitted"], tp, d)
    a2 = C.adaptive_weights(rv2, F0, w["ga"], w["fitted"], tp, d)
    assert a1[i] == a2[i]
