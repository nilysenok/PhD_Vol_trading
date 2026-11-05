# -*- coding: utf-8 -*-
"""Zh-194 (h = 5, 22): one test set. Synthetic data, no network, no real data.
Closed window (mutation 'forecast made at d-h+1 is used in the window of day d', h = 5 and 22), regime / z of day d+1 caught, PIT of the logistic weights and
of the G6 regression (target end < cut), known answers (stress weight 0 -> calm expert; (a) with equal experts = the expert; G6 on a constant squared error; v >= 0), picks, ledger."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import gl3_r6b_layers as L  # noqa: E402
import gl3_zh190_boost_jobs as J  # noqa: E402
import gl3_zh190_stage23_core as Q  # noqa: E402
import gl3_zh190_stage23_layers as Y3  # noqa: E402
import gl3_zh194_chain as CH  # noqa: E402
import gl3_zh194_core as Z  # noqa: E402
import gl3_zh194_fit as Fit  # noqa: E402


def make_a(h, n=900, seed=1, start='2016-01-04'):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rvi = pd.Series(np.exp(3 + .3 * np.cumsum(rng.standard_normal(n)) / 10), index=dates)
    parts = []
    for t in ('AAA', 'BBB', 'CCC'):
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = .9 * z[i - 1] + .3 * rng.standard_normal()
        parts.append(pd.DataFrame(dict(date=dates, ticker=t, rv=np.exp(z - 8), close_RVI=rvi.values)))
    a = pd.concat(parts, ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)
    g = a.groupby('ticker')
    a['rv_w'] = g.rv.transform(lambda s: s.rolling(5, min_periods=1).mean())
    a['rq'] = a.rv ** 2 * 3
    a['year'] = a.date.dt.year
    a[f'_end{h}'] = g.date.shift(-h)
    a[f'_y{h}'] = np.column_stack([g.rv.shift(-j) for j in range(1, h + 1)]).mean(axis=1)
    return a


def base_level(a, seed=2):
    rng = np.random.default_rng(seed)
    return a.groupby('ticker').rv.transform(lambda s: s.rolling(22, min_periods=1).mean()).values * np.exp(.2 * rng.standard_normal(len(a)))


def zfeat(a):
    return np.column_stack([np.log(a.close_RVI.values), np.log(a.rv_w.values)]) - 0.


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, 'EVENTS', tmp_path / 'fit_events.jsonl')
    return tmp_path


@pytest.mark.parametrize('h', [5, 22])
def test_closed_window_mutation_caught(h):
    """Value of row j is known at row j+h; lag h-1 (a forecast made at d-h+1 in the window of day d) must change the output, lag h must not."""
    a = make_a(h)
    y, p = a[f'_y{h}'].values, base_level(a)
    j = np.flatnonzero((a.ticker == 'AAA').values)[400]
    y2 = y.copy()
    y2[j] *= 3.
    at = np.flatnonzero((a.ticker == 'AAA').values)[400 + h - 1]   # the day d = j + h - 1: target of row j is not closed yet
    clean = L.layer14(a, p, y, 20, .75, lag=h)[at], L.layer14(a, p, y2, 20, .75, lag=h)[at]
    mut = L.layer14(a, p, y, 20, .75, lag=h - 1)[at], L.layer14(a, p, y2, 20, .75, lag=h - 1)[at]
    assert clean[0] == clean[1]
    assert mut[0] != mut[1]
    pts = Z.ewma_points(a, p, y2, h)
    assert set(pts) == {f'hl{hl}_w{w}' for hl in (10, 20, 60) for w in (50, 75)} and all(np.isfinite(v).all() for v in pts.values())
    assert all(np.allclose(v[at], L.layer14(a, p, y2, int(k[2:].split('_')[0]), int(k.split('_w')[1]) / 100, lag=h)[at]) for k, v in pts.items())


def test_regime_of_next_day_caught():
    a = make_a(5)
    base = Y3.z_features(a)
    a2 = a.copy()
    last = a2.date.max()
    a2.loc[a2.date == last, 'close_RVI'] *= 5.   # a change on the last date must not touch the regime of any earlier day
    z2 = Y3.z_features(a2)
    earlier = (a.date < last).values
    assert np.allclose(base[earlier], z2[earlier], equal_nan=True)
    x, _ = L.rvi_pair(a, lead=1)                  # mutation: regime of day d+1
    x2, _ = L.rvi_pair(a2, lead=1)
    assert not np.allclose(x[earlier], x2[earlier], equal_nan=True)


@pytest.mark.parametrize('h', [5, 22])
def test_pit_logistic_and_g6_end_before_cut(h, ledger):
    a = make_a(h)
    y, end, p = a[f'_y{h}'].values, a[f'_end{h}'].values, base_level(a)
    z = Y3.z_features(a)
    CH.a2('b_mix', a, z, y, end, h, p, p * 1.3, Z.record)
    Z.g6_forecast(a, p, y, end, h, Z.g6_features(a), rec=Z.record)
    ev = pd.read_json(ledger / 'fit_events.jsonl', lines=True)
    assert {'b_mix_A2_logit', 'G6_ols'} <= set(ev.component) and (ev.h == h).all()
    assert (pd.to_datetime(ev.max_target_end) < pd.to_datetime(ev.cutoff)).all()


def test_g6_ignores_rows_not_closed_at_cut_and_mutation_caught(monkeypatch):
    h = 5
    a = make_a(h)
    y, end, p, f = a[f'_y{h}'].values, a[f'_end{h}'].values, base_level(a), Z.g6_features(a)
    cut = pd.Timestamp('2018-01-01')
    y2 = y.copy()
    y2[((a.date >= cut - pd.Timedelta(days=3)) & (a.date < cut)).values] *= 10.   # target ends on or after the cut: unknown at the cut
    y2[(a.year >= 2018).values] *= 10.
    va = (a.year == 2018).values
    assert np.array_equal(Z.g6_forecast(a, p, y, end, h, f)[va], Z.g6_forecast(a, p, y2, end, h, f)[va])
    monkeypatch.setattr(Q, 'past_mask', lambda e, c: np.ones(len(e), bool))   # mutation: no closed-window condition
    assert not np.allclose(Z.g6_forecast(a, p, y, end, h, f)[va], Z.g6_forecast(a, p, y2, end, h, f)[va])


def test_g6_constant_squared_error_known_answer_and_v_nonnegative():
    h = 5
    a = make_a(h)
    p = base_level(a)
    sgn = np.where(np.arange(len(a)) % 2, 1., -1.)
    y = p * np.exp(.4 * sgn)                      # squared log error = 0.16 everywhere
    f = Z.g6_features(a)
    out = Z.g6_forecast(a, p, y, a[f'_end{h}'].values, h, f)
    m = (a.year == 2018).values
    assert np.allclose(out[m] / p[m], np.exp(.16 / 2), rtol=1e-6)
    assert np.array_equal(out[(a.year == 2016).values], p[(a.year == 2016).values])   # no history: the base
    # v >= 0: e2 = 0.3 (f - 1) on f in [1, 2] before 2018, f = 0 in 2018 -> the raw prediction is -0.3, clipped to 0
    q = a.year.values < 2018
    fx = np.where(q, 1 + (np.arange(len(a)) % 7) / 6, 0.)
    feats = np.column_stack([fx, np.zeros(len(a)), np.zeros(len(a))])
    y2 = p * np.exp(np.sqrt(.3 * np.maximum(fx - 1, 0)) * sgn)
    out2 = Z.g6_forecast(a, p, y2, a[f'_end{h}'].values, h, feats)
    assert np.array_equal(out2[m], p[m])


def test_stress_weight_zero_gives_calm_and_equal_experts_give_expert(monkeypatch):
    h = 5
    a = make_a(h)
    y, end, calm = a[f'_y{h}'].values, a[f'_end{h}'].values, base_level(a)
    stress = calm * 1.7
    z = Y3.z_features(a)

    class R:
        x = np.array([-60., 0., 0.])
    monkeypatch.setattr(Y3, 'fit_logit_ln', lambda *args: R)
    m = (a.year >= 2017).values
    assert np.allclose(CH.a2('t', a, z, y, end, h, calm, stress, None)[m], calm[m], rtol=1e-12)   # weight of the stress expert = 0
    monkeypatch.undo()
    same = CH.a2('a', a, z, y, end, h, calm, calm, None)                                          # (a) with equal experts
    assert np.allclose(same, calm, rtol=1e-12)
    mix = CH.a2('b', a, z, y, end, h, calm, CH.a2('b_mix', a, z, y, end, h, calm, calm, None), None)   # (b) with equal experts
    assert np.allclose(mix, calm, rtol=1e-12)


def test_pick_best_loyo_and_forced():
    year = np.repeat([2016, 2017, 2018, 2019, 2020, 2021], 2)
    sc = lambda p: {**{str(y): float(p[0]) for y in range(2017, 2022)}, 'mean_QLIKE': float(p[0])}
    series = {'x': np.array([.3] * len(year)), 'y': np.array([.2] * len(year))}
    pk = Z.Pick(score=sc)
    assert pk.best('t', series) == 'y' and (pk.loyo('l', series, year) == .2).all()
    fo = Z.Pick(forced=pk.log)
    assert fo.best('t', series) == 'y' and (fo.loyo('l', series, year) == .2).all()
    with pytest.raises(AssertionError):
        Z.Pick(score=sc, forced={})


def test_ledger_refuses_leak_and_foreign_horizon(ledger):
    Z.record(5, 'x', '2018-01-01', 10, '2017-12-20')
    with pytest.raises(AssertionError):
        Z.record(5, 'x', '2018-01-01', 10, '2018-01-02')
    with pytest.raises(AssertionError):
        Z.record(1, 'x', '2018-01-01', 10, '2017-12-20')
    with pytest.raises(AssertionError):
        Z.record(22, 'x', '2022-01-01', 10, '2021-12-20')


def test_seed_shift_of_bag_and_cells():
    assert Fit.bag_seeds(42) == J.SEEDS and len(Fit.bag_seeds(43)) == 10 and not set(Fit.bag_seeds(43)) & set(J.SEEDS)
    assert len({tuple(Fit.bag_seeds(s)) for s in Z.SEEDS}) == 5 and Z.SEEDS == (42, 43, 44, 45, 46)
    assert Fit.cell('g', 22, 2018, 43).name == 'h22_2018_g_s43.parquet'
