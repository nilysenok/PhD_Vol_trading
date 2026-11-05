# -*- coding: utf-8 -*-
"""Zh-190 stage 1: one test set for the module (P8). Synthetic data, no network, no real data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import lightgbm as lgb  # noqa: E402
import gl3_r56_engine as B  # noqa: E402
import gl3_r6h5_engine as R6  # noqa: E402
import gl3_zh190_boost_core as K  # noqa: E402
import gl3_zh190_boost_schemes as S  # noqa: E402
import gl3_zh190_boost_eval as E  # noqa: E402


def make_frame(h, n=1100, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2015-01-05', periods=n)
    parts = []
    for t in ('AAA', 'BBB', 'CCC', 'DDD'):
        z = np.zeros(n)
        for i in range(1, n):
            z[i] = .9 * z[i - 1] + .3 * rng.standard_normal()
        parts.append(pd.DataFrame(dict(date=dates, ticker=t, rv=np.exp(z + .1 * rng.standard_normal(n) - 8))))
    b = pd.concat(parts, ignore_index=True).sort_values(['ticker', 'date']).reset_index(drop=True)
    b['close'] = b.ticker.map({'AAA': 10., 'BBB': 20., 'CCC': 50., 'DDD': 90.})
    g = b.groupby('ticker')
    b[f'_end{h}'] = g.date.shift(-h)
    b[f'_y{h}'] = np.column_stack([g.rv.shift(-j) for j in range(1, h + 1)]).mean(axis=1)
    return b


def make_ctx(h=5, cut='2018-01-01', n=1100):
    b = make_frame(h, n)
    lr = np.log(b.rv.values)
    g = b.groupby('ticker').rv
    x = np.column_stack([lr, np.log(g.transform(lambda s: s.rolling(5, min_periods=1).mean()).values),
                         np.log(g.transform(lambda s: s.rolling(22, min_periods=1).mean()).values)]
                        + [np.random.default_rng(i).standard_normal(len(b)) for i in range(5)]).astype(np.float32)
    tcode = b.ticker.map({t: i for i, t in enumerate(sorted(b.ticker.unique()))}).values
    xc = np.column_stack([x, tcode, np.zeros(len(b))]).astype(np.float32)
    logy = np.log(np.clip(b[f'_y{h}'].values, 1e-12, None))
    off = x[:, 2].astype(float)
    return K.Ctx(b, x, logy, off, h, cut, Xc=xc)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(B, 'EVENTS', tmp_path / 'fit_events.jsonl')
    return tmp_path / 'fit_events.jsonl'


def events(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_qlike_gradient_hessian_known_answer():
    """Loss L(f) = exp(y-f) - (y-f) - 1: grad = 1 - exp(y-f), hess = exp(y-f); central differences agree."""
    y = np.array([-1.3, 0.2, 2.5, -0.4], dtype=np.float32).astype(float)  # LightGBM keeps labels in float32
    f =np.array([-1.0, 0.5, 1.0, -0.4])
    ds = lgb.Dataset(np.zeros((4, 1)), label=y).construct()
    g, hs = K.qlike_obj(f, ds)
    loss = lambda z: np.exp(y - z) - (y - z) - 1
    e = 1e-5
    np.testing.assert_allclose(g, (loss(f + e) - loss(f - e)) / (2 * e), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(hs, (loss(f + e) - 2 * loss(f) + loss(f - e)) / e ** 2, rtol=1e-4)
    np.testing.assert_allclose(g, 1 - np.exp(y - f))
    np.testing.assert_allclose(hs, np.exp(y - f))


def test_huber_threshold_of_final_and_multiples():
    c = make_ctx()
    assert K.HUBER_ALPHA == 0.9
    for loss, mult in (('huber1', 1), ('huber2', 2), ('huber4', 4)):
        p = K.lgb_params(c, loss, {}, 1)
        assert p['objective'] == 'huber' and p['alpha'] == pytest.approx(.9 * mult)
    assert K.lgb_params(c, 'mse', {}, 1)['objective'] == 'regression'


@pytest.mark.parametrize('h', [1, 5, 22])
def test_purge_and_embargo_h_days(h):
    c = make_ctx(h)
    dates = np.sort(c.b.date.unique())
    pos = lambda d: int(np.searchsorted(dates, np.datetime64(d), 'left'))
    end = c.b[f'_end{h}']
    assert (end[c.tr] < pd.Timestamp(dates[pos(c.cut) - h])).all()      # outer: end of target h dates before the cut
    assert (end[c.ti] < pd.Timestamp(dates[pos(c.edge) - h])).all()      # inner: h dates before validation
    assert (c.b.date[c.ti] < c.edge).all() and (c.b.date[c.vi] >= c.edge).all()
    assert not (c.ti & c.vi).any() and not (c.tr & c.pw).any()
    if h == 5:
        a = c.b.assign(_y5=c.b['_y5'])
        np.testing.assert_array_equal(R6.purged(a, c.cut), c.tr)         # same rule as purged() of rounds 6/6b


def test_mutation_no_embargo_is_caught(monkeypatch):
    """Mutation: gap_boundary without the embargo (gap 0) must break the embargo property."""
    good = make_ctx(5)
    monkeypatch.setattr(R6, 'gap_boundary', lambda a, cut, gap=5: pd.Timestamp(cut))
    bad = make_ctx(5)
    dates = np.sort(good.b.date.unique())
    limit = pd.Timestamp(dates[int(np.searchsorted(dates, np.datetime64(good.cut), 'left')) - 5])
    assert (good.b['_end5'][good.tr] < limit).all()
    assert not (bad.b['_end5'][bad.tr] < limit).all()


def test_target_and_feature_alignment_checks_catch_forward_shift():
    """Mutation 'shift forward': the target of day d+1 (or a feature of d+1) is caught by the alignment check."""
    b = make_frame(5)
    assert E.alignment_ok(b, 5, b.rv.values)
    shifted = b.copy()
    shifted['_y5'] = shifted.groupby('ticker')['_y5'].shift(-1)          # target of d+1
    assert not E.alignment_ok(shifted, 5, b.rv.values)
    feat_next = b.groupby('ticker').rv.shift(-1).values                   # feature of d+1
    assert not E.alignment_ok(b, 5, feat_next)


@pytest.mark.parametrize('scheme', K.SCHEMES)
@pytest.mark.parametrize('loss', ['mse', 'huber2', 'qlike'])
def test_pit_all_fits_before_cut_and_forecasts_valid(scheme, loss, ledger):
    c = make_ctx(5)
    m, unit = S.scheme_parts(c, scheme, loss)
    f = S.finalize(c, loss, m, unit)
    ev = events(ledger)
    assert ev and all(pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01') for e in ev)
    assert all(e['h'] == 5 for e in ev)
    np.testing.assert_array_equal(f['pw'], np.flatnonzero(c.pw))         # covers every row of the cutoff year
    assert np.isfinite(f['pred']).all() and (f['pred'] > 0).all()
    assert (c.b.date.values[f['pw']] >= np.datetime64(c.cut)).all()


def test_o7_correction_ddof0_and_none_for_qlike(ledger):
    c = make_ctx(5)
    m, unit = S.scheme_parts(c, 'panel', 'mse')
    f = S.finalize(c, 'mse', m, unit)
    s2 = np.var(c.res[m['vi']] - m['gv'])                                 # ddof 0, one scalar per panel
    assert f['s2_unit'].iloc[0] == pytest.approx(s2, rel=1e-12)
    np.testing.assert_allclose(f['pred'], np.exp(f['raw'] + s2 / 2))
    mq, uq = S.scheme_parts(c, 'panel', 'qlike')
    fq = S.finalize(c, 'qlike', mq, uq)
    np.testing.assert_allclose(fq['pred'], np.exp(fq['raw']))             # no exp(s2/2) for QLIKE in logs


def test_ticker_scheme_uses_top_features_and_per_ticker_s2(ledger):
    c = make_ctx(5)
    m, unit = S.scheme_parts(c, 'ticker', 'mse')
    f = S.finalize(c, 'mse', m, unit)
    assert sorted(f['s2_unit'].index) == ['AAA', 'BBB', 'CCC', 'DDD']
    assert f['s2_unit'].nunique() == 4
    assert sum(1 for e in events(ledger) if e['phase'] == 'refit') == 4 + 1  # four tickers + the panel for the gains


@pytest.mark.parametrize('loss', ['mse', 'qlike'])
def test_cont_adds_50_to_200_trees_over_panel(loss, ledger):
    c = make_ctx(5)
    S.scheme_parts(c, 'cont', loss)
    refit = [e for e in events(ledger) if e['phase'] == 'refit' and 'extra' in e['detail']]
    assert len(refit) == 4
    for e in refit:
        d = e['detail']
        assert 50 <= d['extra'] <= 200
        assert d['total'] == d['panel_trees'] + d['extra']                # trees beyond the panel


def test_cont_differs_from_panel_and_dataset_built_once(ledger):
    c = make_ctx(5)
    mp, _ = S.scheme_parts(c, 'panel', 'mse')
    n_cache = len(c.cache)
    mc, _ = S.scheme_parts(c, 'cont', 'mse')
    assert len(c.cache) == n_cache                                        # panel learner and its Datasets reused by cont
    assert not np.allclose(mp['g'], mc['g'])
    assert np.isfinite(mc['g']).all()


def test_bagging_average_is_mean_of_log_forecasts(ledger):
    c = make_ctx(5)
    ms = [S.scheme_parts(c, 'panel', 'mse', seed=s)[0] for s in (42, 43)]
    a = S.average(ms)
    np.testing.assert_allclose(a['g'], (ms[0]['g'] + ms[1]['g']) / 2)
    assert not np.allclose(ms[0]['g'], ms[1]['g'])


def test_groups_and_optuna_are_deterministic(ledger):
    c = make_ctx(5)
    groups = {'AAA': 0, 'BBB': 0, 'CCC': 1, 'DDD': 1}
    p1, rows1 = S.tune(c, 'panel', 'mse', groups, 0, n_trials=3)
    p2, rows2 = S.tune(c, 'panel', 'mse', groups, 0, n_trials=3)
    assert p1 == p2 and len(rows1) == 3 and all(r['group'] == 0 for r in rows1)
    m, unit = S.scheme_parts(c, 'panel', 'mse', groups=groups, tuned={0: p1})
    assert sorted(set(unit)) == [0, 1] and len(m['pw']) == int(c.pw.sum())


def test_kmeans_groups_are_stable():
    b = make_frame(5)
    vol = np.ones(len(b))
    g1, g2 = S.clusters(b, vol, pd.Timestamp('2018-01-01')), S.clusters(b, vol, pd.Timestamp('2018-01-01'))
    assert g1 == g2 and set(g1) == {'AAA', 'BBB', 'CCC', 'DDD'}
