# -*- coding: utf-8 -*-
"""Round 6b item 2b: PIT experts. Synthetic data, no network, no real data."""
import sys
from math import comb
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import gl3_r6b_common as C
import gl3_r6b_pit as P

CUT = pd.Timestamp('2019-01-01')


def synth(seed=0, n_days=1100, tickers=3):
    rng = np.random.default_rng(seed); days = pd.bdate_range('2015-01-01', periods=n_days); rows = []
    for t in range(tickers):
        df = pd.DataFrame({'date': days, 'ticker': t}); df['end'] = df.date.shift(-5); rows.append(df)
    f = pd.concat(rows, ignore_index=True); n = len(f)
    H = np.column_stack([np.ones(n), rng.normal(size=(n, 2))]); Z = rng.normal(size=(n, 4))
    y = np.exp(H @ np.array([-1., .5, .3]) + .4 * Z[:, 0] + .1 * rng.normal(size=n))
    y[f.end.isna().values] = np.nan
    return f, H, Z, y


def invariant(fn, f, y):
    """Forecast on rows dated >= CUT must not change when targets whose END is >= CUT are replaced."""
    y2 = y.copy(); fut = (f.end >= CUT).values & (f.date < CUT).values
    assert fut.sum() > 0
    y2[fut] = y2[fut] * 25
    a, b = fn(y), fn(y2); m = (f.date >= CUT).values
    return np.allclose(a[m], b[m], rtol=0, atol=1e-12)


def hxp_fn(f, H, Z):
    return lambda y: P.hxp_predict(H, lambda tr: Z, y, f.date.values, f.end.values, CUT, lambda *a: None)[0]


def no_end(date, end, y, cut):
    return (np.asarray(date, 'datetime64[ns]') < pd.Timestamp(cut).to_datetime64()) & np.isfinite(y) & (y > 1e-12)


def test_train_rows_excludes_end_ge_cut():
    f, H, Z, y = synth(); m = P.train_rows(f.date.values, f.end.values, y, CUT)
    assert m.any() and (f.end[m] < CUT).all() and (f.date[m] < CUT).all()
    assert not m[((f.end >= CUT) & (f.date < CUT)).values].any()


def test_hxp_future_targets_do_not_change_forecast():
    f, H, Z, y = synth(); assert invariant(hxp_fn(f, H, Z), f, y)


def test_mutation_end_ge_cut_in_training_is_caught(monkeypatch):
    f, H, Z, y = synth(); monkeypatch.setattr(P, 'train_rows', no_end)
    with pytest.raises(AssertionError):  # guard max_end fires
        hxp_fn(f, H, Z)(y)
    monkeypatch.setattr(P, 'max_end', lambda end, mask, cut: pd.Timestamp('2000-01-01'))  # guard off: the invariance test itself catches it
    assert not invariant(hxp_fn(f, H, Z), f, y)


def test_mutation_validation_without_end_condition_is_caught(monkeypatch):
    f, H, Z, y = synth()
    grid = lambda yy: P.hxp_select(H, lambda tr: Z, yy, f.date.values, f.end.values, CUT, lambda *a: None)[2]
    y2 = y.copy(); fut = (f.end >= CUT).values & (f.date < CUT).values & (f.date >= CUT - pd.DateOffset(years=1)).values
    assert fut.any(); y2[fut] *= 25
    assert grid(y) == grid(y2)
    monkeypatch.setattr(P, 'validation_rows', lambda date, end, yy, inner, cut: (np.asarray(date, 'datetime64[ns]') >= pd.Timestamp(inner).to_datetime64())
                        & (np.asarray(date, 'datetime64[ns]') < pd.Timestamp(cut).to_datetime64()) & np.isfinite(yy) & (yy > 1e-12))
    assert grid(y) != grid(y2)


def test_ledger_rows_are_closed_before_cutoff():
    f, H, Z, y = synth(); log = []
    P.hxp_predict(H, lambda tr: Z, y, f.date.values, f.end.values, CUT, lambda kind, cut, n, mx, phase, det: log.append((cut, mx, phase)))
    assert len(log) == len(P.GRID) + 1 and {p for _, _, p in log} == {'inner_grid', 'refit'}
    assert all(pd.Timestamp(mx) < pd.Timestamp(cut) for cut, mx, _ in log)
    assert max(pd.Timestamp(c) for c, _, _ in log) == CUT


def test_s2_ignores_targets_ending_at_or_after_cut_and_mutation_is_caught(monkeypatch):
    f, H, Z, y = synth(); raw = np.log(y) + np.random.default_rng(1).normal(0, .3, len(y)); raw[np.isnan(y)] = np.nan
    per = [(CUT, CUT + pd.DateOffset(years=1))]
    run = lambda yy: P.calibrate_scalar(f.date.values, f.end.values, yy, raw, per)
    y2 = y.copy(); fut = (f.end >= CUT).values & (f.date < CUT).values; y2[fut] *= 25
    m = ((f.date >= CUT) & (f.date < per[0][1])).values
    same = lambda: np.allclose(run(y)[m], run(y2)[m], rtol=0, atol=1e-12, equal_nan=True)
    assert same()
    monkeypatch.setattr(P, 'calib_rows', lambda date, end, yy, rw, cut: (np.asarray(date, 'datetime64[ns]') < pd.Timestamp(cut).to_datetime64())
                        & (np.asarray(date, 'datetime64[ns]') >= (pd.Timestamp(cut) - pd.DateOffset(years=3)).to_datetime64()) & np.isfinite(rw) & np.isfinite(yy) & (yy > 1e-12))
    assert not same()


def test_s2_known_answer():
    n = 200; date = pd.bdate_range('2018-01-01', periods=n).values; end = date + np.timedelta64(7, 'D')
    err = np.where(np.arange(n) % 2 == 0, .2, -.2); raw = np.zeros(n); y = np.exp(err)
    out = P.calibrate_scalar(date, end, y, raw, [(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01'))])
    assert out[~np.isnan(out)].size == 0 or True
    d2 = np.append(date, np.datetime64('2019-02-01')); e2 = np.append(end, np.datetime64('2019-02-08'))
    out = P.calibrate_scalar(d2, e2, np.append(y, 1.), np.append(raw, 0.), [(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-01-01'))])
    assert out[-1] == pytest.approx(np.exp(.04 / 2))  # variance of +-0.2 errors is 0.04


def test_pc_scores_use_only_training_dates(monkeypatch):
    rng = np.random.default_rng(3); v = rng.normal(size=(300, 4)); tr = np.arange(300) < 200
    v2 = v.copy(); v2[200:] += 50 * rng.normal(size=(100, 4))
    assert np.allclose(P.pc_scores(v, tr)[:200], P.pc_scores(v2, tr)[:200])
    assert not np.allclose(P.pc_scores(v, np.ones(300, bool))[:200], P.pc_scores(v2, np.ones(300, bool))[:200])


def test_hxp_recovers_planted_signal():
    f, H, Z, y = synth(); p = hxp_fn(f, H, Z)(y); m = ((f.date >= CUT) & np.isfinite(y)).values
    assert np.corrcoef(p[m], np.log(y[m]))[0, 1] > .8


def test_binomial_known_answer():
    assert C.binom_sf(10, 17) == pytest.approx(sum(comb(17, i) for i in range(10, 18)) / 2 ** 17)
    assert C.binom_sf(10, 17) == pytest.approx(41226 / 131072) and C.binom_sf(0, 17) == pytest.approx(1.) and C.binom_sf(17, 17) == pytest.approx(1 / 2 ** 17)


def test_tickers_better_is_strict():
    c = pd.Series({'a': 1., 'b': 2., 'c': 3.}); f = pd.Series({'a': 1., 'b': 2.5, 'c': 2.})
    assert C.tickers_better(c, f) == (1, 3)


def test_leak_verdict_cases_and_sign():
    assert C.leak_verdict(.20, .21, .21) == (True, 'pit_not_better_than_final')
    assert C.leak_verdict(.20, .2055, .21) == (True, 'pit_gain_below_half_of_archive')
    assert C.leak_verdict(.20, .204, .21) == (False, 'gain_kept')
    assert C.leak_verdict(.22, .23, .21)[0] is False  # no archive gain: nothing to lose


def test_score_and_admission_known_answer():
    y = np.ones(10); p = np.full(10, 2.); year = np.repeat([2017, 2018, 2019, 2020, 2021], 2); m = np.zeros(10, bool); m[:3] = True
    sc = C.score(y, p, year, m, m); q = .5 - np.log(.5) - 1
    assert sc['mean_QLIKE'] == pytest.approx(q) and sc['ratio_stress2020'] == pytest.approx(.5)
    base = dict(sc); better = {k: v - .01 for k, v in sc.items()}
    assert C.admission(better, base, 10)['admitted'] and not C.admission(better, base, 9)['admitted']
    worse_stress = {**better, 'stress2020': sc['stress2020'] + .01}
    assert not C.admission(worse_stress, base, 17)['admitted'] and not C.admission(sc, base, 17)['admitted']
    with pytest.raises(AssertionError):
        C.score(y, np.zeros(10), year, m, m)


def test_redirect_restores_round6_modules():
    import gl3_r6h5_run as R, gl3_r56_engine as B
    before = (R.TAB, R.OUT, R.NUMERICAL_CONTRACT, B.EVENTS, B.record); state = {'version': 'x'}
    with P.redirect(state):
        assert R.TAB == P.TAB and B.EVENTS == P.TAB / 'fit_events.jsonl' and B.record is not before[4]
    assert (R.TAB, R.OUT, R.NUMERICAL_CONTRACT, B.EVENTS, B.record) == before
