# -*- coding: utf-8 -*-
"""Zh-190 stages 2-3: one test set for the module. Synthetic data, no network, no real data.
PIT (target end < cut in fits and in A-1 / A-2 weights), closed windows of G-3 / A-3 (mutation s = d - h + 1), forward shift (regime of d+1),
c = 0 of G-1 = HAR_i with correction, w = 0 of G-3 = HAR_i, known answers (NNLS ridge, logistic mixture, LOYO, Itog h)."""
# xtask: файл длиннее 200 строк — один набор тестов на модуль (требование задания)
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import gl3_r56_engine as B  # noqa: E402
import gl3_r6b_layers as L  # noqa: E402
import gl3_zh190_boost_core as K  # noqa: E402
import gl3_zh190_boost_schemes as S  # noqa: E402
import gl3_zh190_stage23_core as Q  # noqa: E402
import gl3_zh190_stage23_fit as F  # noqa: E402
import gl3_zh190_stage23_layers as Y3  # noqa: E402

H = 5


def make_a(h=H, n=900, seed=1, start='2016-01-04'):
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
    a['year'] = a.date.dt.year
    a[f'_end{h}'] = g.date.shift(-h)
    a[f'_y{h}'] = np.column_stack([g.rv.shift(-j) for j in range(1, h + 1)]).mean(axis=1)
    return a


def levels(a, seed=2):
    rng = np.random.default_rng(seed)
    base = a.groupby('ticker').rv.transform(lambda s: s.rolling(22, min_periods=1).mean()).values
    return [base * np.exp(.2 * rng.standard_normal(len(a))) for _ in range(4)]


def tail_rows(a, cut, h=H):
    """Rows before the cut whose target ends on or after it (not yet known at the cut)."""
    return ((a.date < cut) & (a[f'_end{h}'] >= cut)).values


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, 'EVENTS', tmp_path / 'fit_events.jsonl')
    monkeypatch.setattr(Q, 'TAB', tmp_path)
    monkeypatch.setattr(B, 'EVENTS', tmp_path / 'fit_events.jsonl')
    return tmp_path / 'fit_events.jsonl'


# ------------------------------------------------------------------ PIT
def test_ledger_refuses_target_end_not_before_cut(ledger):
    Q.record(5, 'x', '2019-01-01', 10, '2018-12-31')
    with pytest.raises(AssertionError):
        Q.record(5, 'x', '2019-01-01', 10, '2019-01-01')
    with pytest.raises(AssertionError):
        Q.record(5, 'x', '2022-01-01', 10, '2021-12-31')
    assert len(ledger.read_text().splitlines()) == 1


def make_ctx(cut='2018-01-01'):
    b = make_a().rename(columns={'rv': 'rv'})
    b = pd.concat([make_a(n=1100, start='2015-01-05')])
    lr = np.log(b.rv.values)
    x = np.column_stack([lr] + [np.random.default_rng(i).standard_normal(len(b)) for i in range(5)]).astype(np.float32)
    logy = np.log(np.clip(b[f'_y{H}'].values, 1e-12, None))
    off = np.log(b.groupby('ticker').rv.transform(lambda s: s.rolling(22, min_periods=1).mean()).values)
    return K.Ctx(b, x, logy, off, H, cut)


def test_fit_pit_all_fits_end_before_cut_and_mutation_caught(ledger):
    c = make_ctx()
    S.scheme_parts(c, 'panel', 'qlike')
    ev = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert ev and all(pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01') for e in ev)
    naive = (c.b.date < c.cut).values  # mutation: purge by the date of the row only
    assert c.b.loc[naive, f'_end{H}'].max() >= c.cut and c.b.loc[c.tr, f'_end{H}'].max() < c.cut
    with pytest.raises(AssertionError):
        Q.record(H, 'mut', c.cut, naive.sum(), c.b.loc[naive, f'_end{H}'].max())


def test_build_off_uses_inwin_of_loyo_lambda_for_train_and_forecast_for_year():
    a = make_a(n=60, start='2019-11-01')[['date', 'ticker']]
    cut = pd.Timestamp('2020-01-01')
    inw = pd.concat([a[a.date < cut].assign(cutoff=2020, ln_lam50=1., ln_lam100=2.), a.assign(cutoff=2019, ln_lam50=7., ln_lam100=8.)])
    har = a[a.date >= cut].assign(ln_raw=-5.)
    off = F.build_off(a, har, inw, {2020: .5}, 2020)
    assert (off[(a.date < cut).values] == 1.).all() and (off[(a.date >= cut).values] == -5.).all()
    assert (F.build_off(a, har, inw, {2020: 1.}, 2020)[(a.date < cut).values] == 2.).all()  # another lambda -> another column


# ------------------------------------------------------------------ G-1 / G-2 / G-3
def test_g1_c0_is_har_i_with_correction_and_g2_c0_is_har_i():
    ln_har, f = np.array([-8., -7.5, -9.]), np.array([.4, -.2, .1])
    s2 = Q.shrink_s2(np.array([.2, .4, .1]), np.array([.3, .3, .3]))
    np.testing.assert_allclose(s2, [.25, .35, .2])
    np.testing.assert_allclose(Q.g1_level(ln_har, f, 0., s2), np.exp(ln_har + s2 / 2), rtol=1e-14)
    np.testing.assert_allclose(Q.g2_level(ln_har, f, 0.), np.exp(ln_har), rtol=1e-14)
    assert not np.allclose(Q.g1_level(ln_har, f, 1., s2), Q.g1_level(ln_har, -f, 1., s2))  # sign of f matters


def test_val_s2_known_answer_and_c_scaling():
    res, gv, tk = np.array([1., 1., -1., -1.]), np.zeros(4), np.array(['A', 'A', 'B', 'B'])
    s, pool = Q.val_s2(res, gv, tk, 1.)
    assert s.to_dict() == {'A': 0., 'B': 0.} and pool == 1.   # ddof 0 by ticker, pooled apart
    s2, pool2 = Q.val_s2(res, res, tk, 1.)                      # f explains the residual exactly
    assert pool2 == 0. and (s2 == 0).all()
    assert Q.val_s2(res, res, tk, .5)[1] == pytest.approx(.25)


def test_g3_w0_gives_har_i_exactly_and_w1_gives_b():
    b, h = np.array([3e-4, 5e-4, 2e-4]), np.array([4e-4, 4e-4, 1e-4])
    np.testing.assert_allclose(Q.blend_ln(b, h, np.zeros(3)), h, rtol=1e-13)
    np.testing.assert_allclose(Q.blend_ln(b, h, np.ones(3)), b, rtol=1e-13)
    np.testing.assert_allclose(Q.blend_ln(b, h, np.full(3, .5)), np.sqrt(b * h), rtol=1e-13)


def test_g3_weight_known_answer_better_expert_gets_weight():
    a = make_a()
    y = a[f'_y{H}'].values
    w = Q.g3_weight(a, np.where(np.isnan(y), 3e-4, y), np.full(len(a), 3e-4), y, H)  # B is exact, HAR is a constant
    pos = a.groupby('ticker').cumcount().values
    assert w[pos >= 100].min() > .99               # the exact expert gets the whole weight
    assert (w[pos < 10] == .5).all()               # nothing closed yet


def perturb_closed(fn, a, ycol, monkeypatch, mutate, p):
    y = a[ycol].values.copy()
    y2 = y.copy()
    y2[p - H + 1:p + 1] *= 4.   # rows whose target ends after the date of row p
    base, same = fn(y), fn(y2)
    monkeypatch.setattr(L, mutate[0], mutate[1])
    return base, same, fn(y), fn(y2)


def test_g3_closed_windows_and_mutation_s_equals_d_minus_h_plus_1(monkeypatch):
    a = make_a()
    lb, lh = levels(a)[:2]
    p = int((a.ticker == 'AAA').values.nonzero()[0][0]) + 400
    fn = lambda y: Q.g3_weight(a, lb, lh, y, H)[p]
    orig = L.closed
    base, same, mb, ms = perturb_closed(fn, a, f'_y{H}', monkeypatch, ('closed', lambda a_, x, lag=None: orig(a_, x, lag - 1)), p)
    assert base == same                    # row p ignores targets that end after its date
    assert mb != ms                        # the mutation (window up to s = d - h + 1) sees them: caught


def test_a3_closed_windows_and_mutation(monkeypatch):
    a = make_a()
    base_lvl = levels(a)[0]
    p = int((a.ticker == 'BBB').values.nonzero()[0][0]) + 500
    fn = lambda y: Y3.a3_points(a, base_lvl, y, H)['hl60_w75'][p]
    orig = L.layer14
    base, same, mb, ms = perturb_closed(fn, a, f'_y{H}', monkeypatch, ('layer14', lambda a_, lv, y, hl, w, lag=None: orig(a_, lv, y, hl, w, lag=lag - 1)), p)
    assert base == same and mb != ms
    assert set(Y3.a3_points(a, base_lvl, a[f'_y{H}'].values, H)) == {f'hl{hl}_w{w}' for hl in (20, 60, 120) for w in (50, 75)}


def test_closed_ok_detects_window_that_is_not_closed():
    a = make_a()
    assert Q.closed_ok(a.date, a[f'_end{H}'], a.ticker, H)
    assert not Q.closed_ok(a.date, a[f'_end{H}'] + pd.Timedelta(days=20), a.ticker, H)


# ------------------------------------------------------------------ A-1
def test_nnls_ridge_known_answer_and_nonnegativity():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 3))
    y = 1. + .3 * X[:, 0] + .7 * X[:, 1]
    w, b = nnls_small(X, y)
    np.testing.assert_allclose(w, [.3, .7, 0.], atol=1e-6)
    assert b == pytest.approx(1., abs=1e-6)
    w3, _ = nnls_small(X, y - .5 * X[:, 2])        # negative true coefficient: clipped to zero, all weights >= 0
    assert w3[2] == 0. and (w3 >= 0).all()
    w1, _ = Y3.nnls_ridge(X[:, :2], y, 1.)
    w100, _ = Y3.nnls_ridge(X[:, :2], y, 1e5)
    assert np.linalg.norm(w100) < np.linalg.norm(w1) < np.linalg.norm([.3, .7])  # ridge shrinks


def nnls_small(X, y):
    return Y3.nnls_ridge(X, y, 1e-9)


def a_inputs():
    a = make_a()
    y, end = a[f'_y{H}'].values, a[f'_end{H}'].values
    lv = levels(a)
    return a, y, end, np.log(np.column_stack(lv)), lv


def mutant_past_mask(end, cut):   # targets that end up to 20 days after the cut are taken as known
    return np.asarray(end, 'datetime64[ns]') < np.datetime64(pd.Timestamp(cut) + pd.Timedelta(days=20))


def test_a1_pit_weights_ignore_targets_ending_at_or_after_cut_and_mutation_caught(monkeypatch):
    a, y, end, lns, lv = a_inputs()
    cut = pd.Timestamp('2018-01-01')
    y2 = y.copy()
    y2[tail_rows(a, cut)] *= 5.
    yr = (a.year == 2018).values
    o1, o2 = (Y3.a1_forecast(a, lns, v, end, H, lv[3])[yr] for v in (y, y2))
    np.testing.assert_array_equal(o1, o2)
    monkeypatch.setattr(Q, 'past_mask', mutant_past_mask)
    m1, m2 = (Y3.a1_forecast(a, lns, v, end, H, lv[3])[yr] for v in (y, y2))
    assert not np.allclose(m1, m2, rtol=1e-12, atol=0)


def test_a1_2016_falls_back_to_gstar_and_records_pit(ledger):
    a, y, end, lns, lv = a_inputs()
    out = Y3.a1_forecast(a, lns, y, end, H, lv[3], rec=Q.record)
    np.testing.assert_array_equal(out[(a.year == 2016).values], lv[3][(a.year == 2016).values])
    ev = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert {e['component'] for e in ev} == {'A1_pool', 'A1_ticker'} and all(e['max_target_end'] < e['cutoff'] for e in ev)


# ------------------------------------------------------------------ A-2
def test_a2_logistic_mixture_known_answer_stress_expert_exact_in_stress():
    rng = np.random.default_rng(5)
    n = 4000
    z = rng.standard_normal((n, 2))
    calm, stress = np.full(n, 1e-3), np.full(n, 4e-3)
    st = z[:, 0] > 0
    y = np.where(st, stress, calm)
    zz = np.column_stack([np.ones(n), z])
    r = Y3.fit_logit_ln(zz, y, calm, stress)
    with np.errstate(all='ignore'):
        g = 1 / (1 + np.exp(-zz @ r.x))
    assert g[st].mean() > .9 and g[~st].mean() < .1 and r.x[1] > 0 and abs(r.x[2]) < abs(r.x[1])


def a2_inputs():
    a, y, end, lns, lv = a_inputs()
    return a, y, end, Y3.z_features(a), lv


def test_a2_pit_weights_ignore_targets_after_cut_and_mutation_caught(monkeypatch):
    a, y, end, z, lv = a2_inputs()
    y2 = y.copy()
    y2[tail_rows(a, pd.Timestamp('2018-01-01'))] *= 5.
    yr = (a.year == 2018).values
    o1, o2 = (Y3.a2_forecast(a, z, v, end, H, lv[0], lv[1])[yr] for v in (y, y2))
    np.testing.assert_array_equal(o1, o2)
    monkeypatch.setattr(Q, 'past_mask', mutant_past_mask)
    m1, m2 = (Y3.a2_forecast(a, z, v, end, H, lv[0], lv[1])[yr] for v in (y, y2))
    assert not np.allclose(m1, m2, rtol=1e-12, atol=0)


def test_a2_forward_shift_regime_of_next_day_caught(monkeypatch):
    a, y, end, z, lv = a2_inputs()
    d0 = pd.Timestamp(sorted(a.date.unique())[600])
    a2 = a.copy()
    a2.loc[a2.date > d0, 'close_RVI'] *= 10.
    early = (a.date <= d0).values & (a.year == 2018).values
    o1 = Y3.a2_forecast(a, z, y, end, H, lv[0], lv[1])[early]
    o2 = Y3.a2_forecast(a, Y3.z_features(a2), y, end, H, lv[0], lv[1])[early]
    np.testing.assert_array_equal(o1, o2)                      # the regime of d does not see d + 1
    monkeypatch.setattr(L, 'RVI_LEAD', 1)                       # mutation: regime of day d + 1
    m1 = Y3.a2_forecast(a, Y3.z_features(a), y, end, H, lv[0], lv[1])[early]
    m2 = Y3.a2_forecast(a, Y3.z_features(a2), y, end, H, lv[0], lv[1])[early]
    assert not np.allclose(m1, m2, rtol=1e-12, atol=0)
    assert not np.array_equal(Y3.z_features(a)[a.date.values == np.datetime64(d0)], z[a.date.values == np.datetime64(d0)])  # the feature itself moved


# ------------------------------------------------------------------ selection, Itog h
def test_loyo_pick_excludes_own_year_and_2016_takes_2017():
    sc = {'A': {str(y): (.1 if y == 2017 else .5) for y in Q.YEARS}, 'B': {str(y): .3 for y in Q.YEARS}}
    pk = Q.loyo_pick(sc, ['A', 'B'])
    assert pk[2017] == 'B' and pk[2016] == 'B'   # A wins 2017 itself, but 2017 is not in its own pick
    assert pk[2018] == 'B'
    out = Q.loyo_series(pk, {'A': np.zeros(4), 'B': np.ones(4)}, np.array([2016, 2017, 2018, 2021]))
    assert (out == 1.).all()


def itog_rows(better=3, mean_h=.2, stress=.3, tick=10):
    base = {n: {**{f'QLIKE_{y}': .3 for y in Q.YEARS}, 'mean_QLIKE': .25, 'stress2020': .3, 'stressRVI': .3} for n in ('FIN_PIT', 'HARi_LOYO')}
    h = {**{f'QLIKE_{y}': (.2 if i < better else .4) for i, y in enumerate(Q.YEARS)}, 'mean_QLIKE': mean_h, 'stress2020': stress, 'stressRVI': .3,
         'tickers_better_vs_FIN_PIT': tick, 'tickers_better_vs_HARi_LOYO': tick}
    return h, base


def test_itog_h_boundaries_each_condition_in_own_column():
    assert Q.itog_h(*itog_rows())['itog_h']
    assert not Q.itog_h(*itog_rows(better=2))['itog_years'] and not Q.itog_h(*itog_rows(better=2))['itog_h']
    assert not Q.itog_h(*itog_rows(mean_h=.25))['itog_mean']          # equal is not better
    assert Q.itog_h(*itog_rows(stress=.3))['itog_stress'] and not Q.itog_h(*itog_rows(stress=.31))['itog_stress']   # not worse: equal passes
    assert not Q.itog_h(*itog_rows(tick=9))['itog_tickers'] and Q.itog_h(*itog_rows(tick=10))['itog_tickers']


def test_parse_config_of_bstar():
    assert Q.parse_config('ticker-huber4') == ('ticker', 'huber4', False, False)
    assert Q.parse_config('cat-qlike') == ('cat', 'qlike', False, False)
    assert Q.parse_config('cat-huber1-bag10') == ('cat', 'huber1', True, False)
