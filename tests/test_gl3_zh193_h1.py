# -*- coding: utf-8 -*-
"""Zh-193 (h = 1): one test set per module. PIT / closed window / forward-shift mutation / bars >= 2022 / known answers. No network, no data >= 2022."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'analysis'))
import gl3_r6b_layers as L
import gl3_zh190_stage23_core as Q
import gl3_zh193_h1_bars as BR
import gl3_zh193_h1_layers as LY

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- bars: known answers
def synth_day(sigma, n_min=560, start=590, jump=None, seed=0):
    r = np.random.default_rng(seed).normal(0, sigma, n_min)
    if jump:
        r[300] += jump
    t = np.arange(start + 1, start + 1 + n_min)
    return t, 100 * np.exp(np.cumsum(r))


def test_rv_constant_variance_matches_one_grid_and_truth():
    sigma, days = 1e-3, 150
    shifted, one, truth = [], [], sigma ** 2 * (BR.S1 - BR.S0)
    for s in range(days):
        t, p = synth_day(sigma, seed=s)
        f = BR.day_features(t, p)
        shifted.append(f['rv5s'])
        one.append(BR.grid_stats(t, p, 0)['rv'])
    assert abs(np.mean(shifted) / truth - 1) < .03
    assert abs(np.mean(shifted) / np.mean(one) - 1) < .03
    assert np.std(shifted) < np.std(one)  # averaging 10 grids reduces the noise


def test_bv_not_above_rv_plus_jump_and_no_jump_means_small_jump():
    js, rv, bv = [], [], []
    for s in range(100):
        f = BR.day_features(*synth_day(1e-3, seed=s))
        assert abs(f['jump5s'] - max(f['rv5s'] - f['bv5s'], 0.)) < 1e-18  # jump = max(RV - BV, 0)
        js.append(f['jump5s'])
        rv.append(f['rv5s'])
        bv.append(f['bv5s'])
    assert abs(np.mean(bv) / np.mean(rv) - 1) < .05 and np.mean(js) / np.mean(rv) < .08  # no jumps: BV ~ RV, jump ~ 0
    fj = BR.day_features(*synth_day(1e-3, jump=.02, seed=1))
    assert fj['jump5s'] > 10 * np.mean(js)  # an injected jump is picked up


def test_hour_parts_and_semivariances_add_up():
    t, p = synth_day(1e-3, seed=3)
    f = BR.day_features(t, p)
    assert f['rv_h1st'] < f['rv5s'] and f['rv_hlast'] < f['rv5s']
    assert abs(f['rs_pos'] + f['rs_neg'] - f['rv5s']) < 1e-15


# ---------------------------------------------------------------- bars: closed years and no forward shift
def test_bars_after_2021_are_not_read(tmp_path):
    with pytest.raises(ValueError):
        BR.read_bars(tmp_path / 'SBER_2022.csv')  # refused by the name, the file is not even opened
    p = tmp_path / 'SBER_2021.csv'
    p.write_text('begin,open,close\n2022-01-03 10:00:00,1,1\n2021-12-30 10:00:00,1,1\n')
    with pytest.raises(AssertionError):
        BR.read_bars(p)  # a 2022 stamp inside a 2021 file is refused after the read


def raw_frame(n=12):
    d = pd.bdate_range('2020-01-01', periods=n)
    r = pd.DataFrame({'date': d, 'ticker': 'AAA', 'rv5s': np.arange(1, n + 1) * 1e-4, 'bv5s': 5e-5, 'jump5s': 0., 'rv_h1st': 1e-5, 'rv_hlast': 1e-5,
                      'rs_pos': 1e-5, 'rs_neg': 1e-5, 'rq5s': 1e-9, 'open_d': 100 + np.arange(n), 'close_d': 100.5 + np.arange(n),
                      'rv_eve': np.where(np.arange(n) > 4, 2e-4, np.nan)})
    return r, pd.DataFrame({'date': d, 'ticker': 'AAA', 'rv_w': 1e-4})


def feature_is_day_d(out, raw):
    return np.allclose(out.ln_rv5s.values, np.log(raw.rv5s.values), equal_nan=True)


def test_features_use_day_d_and_mutation_forward_shift_is_caught():
    raw, b = raw_frame()
    assert feature_is_day_d(BR.assemble(raw, b), raw)
    fwd = raw.copy()
    fwd['rv5s'] = fwd.rv5s.shift(-1)  # mutation: bars of day d+1
    assert not feature_is_day_d(BR.assemble(fwd, b), raw)


def test_gap_and_evening_use_previous_day_only():
    raw, b = raw_frame()
    out = BR.assemble(raw, b)
    assert np.isnan(out.gap_sig.iloc[0])
    assert np.isclose(out.gap_sig.iloc[3], np.log(raw.open_d.iloc[3] / raw.close_d.iloc[2]) / np.sqrt(1e-4), rtol=1e-5)
    # evening RV of d-1 is on row d: raw.rv_eve is defined from row 5, so the feature starts on row 6
    assert out.ln_rv_eve_prev.iloc[:6].isna().all() and np.isclose(out.ln_rv_eve_prev.iloc[6], np.log(2e-4), rtol=1e-5)


# ---------------------------------------------------------------- fit module: weights, monotone vector, ledger
def test_half_life_weights_known_answer_and_pit_mutation():
    import gl3_zh193_h1_fit as Z
    cut = pd.Timestamp('2020-01-01')
    d = pd.DatetimeIndex(['2019-01-01', '2018-01-01', '2019-12-31'])
    w1, w2 = Z.half_life_weights(d, cut, 1.), Z.half_life_weights(d, cut, 2.)
    assert abs(w1[0] - .5) < 2e-3 and abs(w2[1] - .5) < 2e-3 and w1[2] > .99 and (w1 <= 1).all()
    assert Z.half_life_weights(pd.DatetimeIndex(['2020-06-01']), cut, 1.)[0] > 1  # a future row would be overweight: the cutoff mask must exclude it


def test_monotone_vector_marks_only_ln_rv_d_w_m():
    import gl3_zh193_h1_fit as Z
    names = ['a', 'ln_rv_w', 'b', 'ln_rv_d', 'ln_rv_m']
    assert Z.mono_vec(names, [0, 1, 3, 4]) == [0, 1, 1, 1] and Z.mono_vec(names, [0, 2]) == [0, 0]


def test_ledger_refuses_target_end_not_before_cut(tmp_path):
    import gl3_zh193_h1_fit as Z
    ok = (1, 'x', '2019-01-01', 10, '2018-12-28')
    Z.EVENTS, old, oldq = tmp_path / 'e.jsonl', Z.EVENTS, Q.EVENTS
    try:
        Z.ledger(*ok)
        with pytest.raises(AssertionError):
            Z.ledger(1, 'x', '2019-01-01', 10, '2019-01-01')
        with pytest.raises(AssertionError):
            Z.ledger(1, 'x', '2022-01-01', 10, '2021-12-30')
    finally:
        Z.EVENTS, Q.EVENTS = old, oldq
    import json
    assert json.loads((tmp_path / 'e.jsonl').read_text().splitlines()[0])['h'] == 1


# ---------------------------------------------------------------- closed window h = 1
def frame(n_t=3, years=range(2016, 2022)):
    rows = []
    for t in range(n_t):
        d = pd.bdate_range(f'{min(years)}-01-01', f'{max(years)}-12-31')
        rows.append(pd.DataFrame({'date': d, 'ticker': f'T{t}'}))
    a = pd.concat(rows, ignore_index=True)
    a['year'] = a.date.dt.year
    a['end'] = a.groupby('ticker').date.shift(-1).fillna(a.date)       # target of day d = day d+1; the last row ends on itself
    a['end2'] = a.groupby('ticker').date.shift(-2).fillna(a.date)      # mutation: target two days ahead
    return a


def test_closed_window_lag_one_and_mutation_same_day_is_caught():
    a = frame(2, range(2020, 2021))
    x = np.arange(len(a), dtype=float)
    c1 = L.closed(a, x, 1)
    own = a.groupby('ticker').cumcount().values
    assert (c1[own > 0] == x[np.flatnonzero(own > 0) - 1]).all()   # value of day d-1 on day d
    c0 = L.closed(a, x, 0)
    assert (c0 == x).all()  # mutation lag 0: the forecast of day d would sit in the window of day d
    assert not np.array_equal(c1[own > 0], c0[own > 0])
    assert Q.closed_ok(a.date, a.end, a.ticker, 1)
    assert not Q.closed_ok(a.date, a.end2, a.ticker, 1)


# ---------------------------------------------------------------- G layers: PIT and known answers
def synth_layers(seed=0):
    a = frame(3)
    n = len(a)
    r = np.random.default_rng(seed)
    base = np.exp(r.normal(-8, .3, n))
    y = base * np.exp(r.normal(0, .2, n))
    lns = np.log(np.column_stack([base * np.exp(r.normal(0, .1, n)) for _ in range(3)] + [base]))
    z3 = r.normal(0, 1, (n, 3))
    return a, y, a.end.values, lns, base, z3


def forecast_changes(fn, a, y, end, Y):
    """True if the forecast of year Y changes when targets of rows with year >= Y are inflated (a leak of the future)."""
    f0 = fn(y)
    y2 = np.where(a.year.values >= Y, y * 4, y)
    f1 = fn(y2)
    m = a.year.values == Y
    return bool(np.abs(np.log(f0[m]) - np.log(f1[m])).max() > 1e-9)


def test_g6_and_g4_use_only_past_targets_and_late_end_mutation_is_caught():
    a, y, end, lns, base, z3 = synth_layers()
    stress = (np.arange(len(a)) % 5 == 0)
    g6 = lambda yy, e=end: LY.g6_series(a, base, yy, e, z3)
    g4 = lambda yy, e=end: LY.g4_series(a, lns, yy, e, stress, base)[0]
    assert not forecast_changes(g6, a, y, end, 2019) and not forecast_changes(g4, a, y, end, 2019)
    early = np.asarray(a.end.values, 'datetime64[ns]') - np.timedelta64(800, 'D')  # mutation: target end declared long before it happens
    assert forecast_changes(lambda yy: g6(yy, early), a, y, end, 2019)
    assert forecast_changes(lambda yy: g4(yy, early), a, y, end, 2019)


def test_g4_calm_and_stress_series_of_year_are_out_of_sample():
    """Every row of year Y in the calm / stress series (inputs of the logistic mixture) comes from the cutoff of its own year: targets of later years
    must not change it (a series overwritten by in-sample fits of later cutoffs would)."""
    a, y, end, lns, base, z3 = synth_layers(6)
    stress = (np.arange(len(a)) % 4 == 0)
    y2 = np.where(a.year.values >= 2020, y * 5, y)
    r0 = LY.g4_series(a, lns, y, end, stress, base)
    r1 = LY.g4_series(a, lns, y2, end, stress, base)
    m = a.year.values <= 2019
    for s0, s1 in zip(r0, r1):
        assert np.abs(np.log(s0[m]) - np.log(s1[m])).max() < 1e-9


def test_g4_known_answer_regime_weights():
    a, _, end, _, _, _ = synth_layers(1)
    n = len(a)
    r = np.random.default_rng(5)
    stress = (np.arange(n) % 3 == 0)
    A, Bx = np.exp(r.normal(-8, .4, n)), np.exp(r.normal(-8, .4, n))
    y = np.where(stress, Bx, A)  # calm: A is exact, stress: B is exact
    lns = np.log(np.column_stack([A, Bx, A * 1.001, Bx * 1.001]))
    hard, calm, strs = LY.g4_series(a, lns, y, end, stress, np.sqrt(A * Bx))
    m = a.year.values >= 2018
    assert np.abs(np.log(hard[m] / y[m])).mean() < .05
    one = Q.level(np.log(np.sqrt(A * Bx)))
    assert np.abs(np.log(hard[m] / y[m])).mean() < .5 * np.abs(np.log(one[m] / y[m])).mean()  # regimes beat one common weight set
    assert np.abs(np.log(calm[m & ~stress] / y[m & ~stress])).mean() < .05 and np.abs(np.log(strs[m & stress] / y[m & stress])).mean() < .05


def test_g6_known_answer_and_sign():
    a, y, end, lns, base, z3 = synth_layers(2)
    out = LY.g6_series(a, base, y, end, z3)
    m = a.year.values >= 2018
    assert np.isfinite(out).all() and (out > 0).all()
    # with log-normal noise independent of z the multiplier exp((v - s2)/2) stays close to 1 and never below exp(-s2/2) (v >= 0)
    ratio = np.log(out[m] / base[m])
    assert ratio.min() > -.2 and np.abs(ratio.mean()) < .2
    assert np.all(out[a.year.values == 2016] == base[a.year.values == 2016])  # no history in 2016: base


def test_g5_points_use_lag_one():
    a, y, end, lns, base, z3 = synth_layers(3)
    pts = LY.g5_points(a, base, y)
    assert set(pts) == {f'hl{h}_w{w}' for h in (10, 20, 60) for w in (50, 75)}
    y2 = y.copy()
    i = len(a) // 2
    y2[i] = y2[i] * 50
    p2 = LY.g5_points(a, base, y2)['hl10_w50']
    same_ticker_next = i + 1
    assert abs(np.log(p2[i] / pts['hl10_w50'][i])) < 1e-12   # target of day d is not in the window of day d (lag 1)
    assert abs(np.log(p2[same_ticker_next] / pts['hl10_w50'][same_ticker_next])) > 1e-6
    assert all(np.all((v / base >= .7 - 1e-9) & (v / base <= 1.5 + 1e-9)) for v in pts.values())


def test_ev_loyo_pick_excludes_own_year():
    a, y, end, lns, base, z3 = synth_layers(4)
    sc = lambda p: {**{str(Y): float(np.mean(np.abs(np.log(p[a.year.values == Y] / y[a.year.values == Y])))) for Y in Q.YEARS}, 'mean_QLIKE': 0.}
    ev = LY.Ev(a.year.values, sc)
    pts = {'good': y.copy(), 'bad': y * 3}
    out = ev.choose('T', pts)
    assert ev.sel['T_pick']['2019'] == 'good' and np.allclose(out, Q.f32(y))
    fixed = LY.Ev(a.year.values, sel=ev.sel)
    assert np.allclose(fixed.choose('T', pts), out)  # the seed replay uses the stored pick
