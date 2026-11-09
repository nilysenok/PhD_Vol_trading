"""E2 (X2): walk-forward PIT — будущие пары, порог метки, метка во входе (мутации через боевой путь), F = H без модели."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s15 import de_e2_feat as FE, de_e2_wf as WF

KW = dict(seeds=(0, 1), betas=(0.1, 1.0), maxiter=12)
FIRST = "2018-06-01"


def _synth(seed=3, n=560):
    d = pd.bdate_range("2017-01-02", periods=n)
    rng = np.random.default_rng(seed)
    tp = d[1:].append(pd.DatetimeIndex([d[-1] + pd.offsets.BDay(1)]))
    H = np.exp(0.2 * rng.standard_normal(n))
    rv = H * np.exp(0.4 * rng.standard_normal(n))
    Z, e, rsn = rng.standard_normal((n, FE.NZ)), (rng.random(n) < 0.1).astype(float), rng.random(n)
    return d, tp.to_numpy("datetime64[ns]"), H, rv, Z, e, rsn


def _run(d, tp, H, rv, Z, e, rsn, **kw):
    return WF.walk_forward(H, rv, tp, d, FE.day_inputs(Z, e, rsn), FIRST, **KW), kw


def _cut(d):
    pos = [p for p in WF.refit_positions(d) if d[p] > pd.Timestamp("2018-08-01")]
    return pos[0], pos[1]                                                           # a_k и следующая переоценка


def _future_perturbed(d, tp, H, rv, Z, e, rsn, a):
    rv2 = rv.copy()
    rv2[a:] *= 1000.0                                                               # пары, не закрытые к d0 = d[a]
    return rv2


def _same_before(r1, r2, b):
    return all(np.array_equal(r1[k][:b], r2[k][:b], equal_nan=True) for k in ("F", "p", "cn", "cs", "beta", "thr"))


def test_future_pairs_do_not_change_forecasts_honest_path():
    d, tp, H, rv, Z, e, rsn = _synth()
    a, b = _cut(d)
    r1, _ = _run(d, tp, H, rv, Z, e, rsn)
    r2, _ = _run(d, tp, H, _future_perturbed(d, tp, H, rv, Z, e, rsn, a), Z, e, rsn)
    assert r1["fitted"][a:b].all() and _same_before(r1, r2, b)


def test_mutation_closed_plus_one_is_caught(monkeypatch):
    d, tp, H, rv, Z, e, rsn = _synth()
    a, b = _cut(d)
    monkeypatch.setattr(WF, "closed_rows", lambda tp_, d0: np.flatnonzero(np.asarray(tp_, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(1, "D") * 3))
    r1, _ = _run(d, tp, H, rv, Z, e, rsn)
    r2, _ = _run(d, tp, H, _future_perturbed(d, tp, H, rv, Z, e, rsn, a), Z, e, rsn)
    assert not _same_before(r1, r2, b)


def test_mutation_threshold_with_future_pairs_is_caught(monkeypatch):
    d, tp, H, rv, Z, e, rsn = _synth()
    a, b = _cut(d)
    monkeypatch.setattr(FE, "stress_threshold", lambda rv_, closed: float(np.quantile(rv_[np.isfinite(rv_)], FE.Q_STRESS)))
    r1, _ = _run(d, tp, H, rv, Z, e, rsn)
    r2, _ = _run(d, tp, H, _future_perturbed(d, tp, H, rv, Z, e, rsn, a), Z, e, rsn)
    assert not _same_before(r1, r2, b)


def test_mutation_label_in_input_is_caught(monkeypatch):
    d, tp, H, rv, Z, e, rsn = _synth()
    a, b = _cut(d)
    rv2 = _future_perturbed(d, tp, H, rv, Z, e, rsn, a)
    honest = [FE.day_inputs(Z, e, rsn) for _ in (0, 1)]
    assert np.array_equal(honest[0], honest[1]) and not np.isin(rv2, honest[0]).any()   # боевой вход не содержит RV(t⁺) ни в каком виде
    cur = {"rv": rv}
    monkeypatch.setattr(FE, "day_inputs", lambda Z_, e_, rsn_: np.column_stack([Z_, e_, (cur["rv"] > np.median(cur["rv"])).astype(float)]))    # метка собственной пары вместо rs_neg_share
    r1, _ = _run(d, tp, H, rv, Z, e, rsn)
    cur["rv"] = rv2
    r2, _ = _run(d, tp, H, rv2, Z, e, rsn)
    assert not _same_before(r1, r2, b)


def test_no_model_before_first_eval_gives_F_equals_H_and_p_nan():
    d, tp, H, rv, Z, e, rsn = _synth()
    r, _ = _run(d, tp, H, rv, Z, e, rsn)
    i0 = int(np.searchsorted(d.to_numpy(), np.datetime64(FIRST)))
    assert np.array_equal(r["F"][:i0], H[:i0]) and np.isnan(r["p"][:i0]).all() and not r["fitted"][:i0].any()
    assert r["fitted"][i0:].all() and np.isfinite(r["p"][i0:]).all() and (r["F"] > 0).all() and ((r["p"][i0:] >= 0) & (r["p"][i0:] <= 1)).all()


def test_validation_rows_are_last_126_closed_usable_pairs_only():
    d, tp, H, rv, Z, e, rsn = _synth()
    d0 = d[400]
    usable = np.ones(len(d), bool)
    usable[[390, 395]] = False
    va = WF.val_rows(tp, d0, usable)
    assert len(va) == WF.HOLD and (tp[va] <= np.datetime64(d0)).all() and 390 not in va and 395 not in va and va.max() == WF.closed_rows(tp, d0).max()


def test_beta_chosen_from_grid_and_stored():
    d, tp, H, rv, Z, e, rsn = _synth()
    r, _ = _run(d, tp, H, rv, Z, e, rsn)
    b = r["beta"][np.isfinite(r["beta"])]
    assert len(b) > 100 and set(np.unique(b)) <= {0.1, 1.0}
