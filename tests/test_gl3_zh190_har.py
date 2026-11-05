# -*- coding: utf-8 -*-
"""Ж-190, HAR_i со сжатием: синтетика + подложенные ошибки. Без сети, без данных ≥ 2022."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import gl3_zh190_har as H

STEPS = (2015, 2016, 2017)
TICKS = ('AAA', 'BBB', 'CCC')


def panel(seed=0, last='2017-12-29'):
    """3 тикера; у каждого свои истинные β (поэтому потикерный OLS ≠ общий); цель — h=5 вперёд."""
    rng = np.random.default_rng(seed); dates = pd.bdate_range('2012-01-02', last); parts = []
    for i, tk in enumerate(TICKS):
        z = pd.DataFrame(dict(date=dates, ticker=tk)); z['end'] = z.date.shift(-5); parts.append(z)
    a = pd.concat(parts, ignore_index=True); a['year'] = a.date.dt.year
    X = rng.normal(size=(len(a), 8)); beta = {tk: rng.normal(size=9) * (i + 1) for i, tk in enumerate(TICKS)}
    ln = np.array([beta[t][0] + X[j] @ beta[t][1:] for j, t in enumerate(a.ticker)])
    y = np.where(a.end.notna(), np.exp(ln * .05 - 8 + rng.normal(0, .1, len(a))), np.nan)
    return a, X, y, a.end.values


def lstsq(X, ly, m):
    Z = np.column_stack([np.ones(m.sum()), X[m]]); return np.linalg.lstsq(Z, ly[m], rcond=None)[0]


def inwin_ln(res, vy, col):
    return res['inwin'][res['inwin'].cutoff == vy]


def test_lambda1_potikerny_ols_lambda0_obshchiy():
    a, X, y, end = panel(); res = H.run_h(a, X, y, end, steps=STEPS); ly = np.log(y + 1e-10)
    end_y = np.asarray(pd.DatetimeIndex(end).year, float); tr = H.train_mask(a.year.values, end_y, y, 2017)
    w = inwin_ln(res, 2017, None); assert len(w) == tr.sum()
    key = w.set_index(['date', 'ticker']); rows = a[tr].set_index(['date', 'ticker']).index
    pool = np.column_stack([np.ones(tr.sum()), X[tr]]) @ lstsq(X, ly, tr)
    np.testing.assert_allclose(key.loc[rows, 'ln_lam0'].values, pool, atol=1e-5)
    for tk in TICKS:
        sel = tr & (a.ticker == tk).values; own = np.column_stack([np.ones(sel.sum()), X[sel]]) @ lstsq(X, ly, sel)
        got = key.loc[a[sel].set_index(['date', 'ticker']).index, 'ln_lam100'].values
        np.testing.assert_allclose(got, own, atol=1e-5)
    mid = key.loc[rows, 'ln_lam50'].values; both = (key.loc[rows, 'ln_lam0'].values + key.loc[rows, 'ln_lam100'].values) / 2
    np.testing.assert_allclose(mid, both, atol=1e-5)


def test_koeffitsienty_ne_vidyat_stroki_s_kontsom_celi_posle_cut():
    a, X, y, end = panel(); y2 = y.copy(); leak = (a.year == 2016).values & (pd.DatetimeIndex(end).year == 2017); assert leak.sum() > 0
    y2[leak] = y2[leak] * 50
    b0 = H.run_h(a, X, y, end, steps=STEPS)['beta']; b1 = H.run_h(a, X, y2, end, steps=STEPS)['beta']
    cols = [c for c in b0 if c.startswith('b')]
    np.testing.assert_allclose(b0[b0.cutoff == 2017][cols].values, b1[b1.cutoff == 2017][cols].values, atol=1e-12)
    m0 = H.run_h(a, X, y, end, emb=False, steps=STEPS)['beta']; m1 = H.run_h(a, X, y2, end, emb=False, steps=STEPS)['beta']  # мутация: без эмбарго
    assert not np.allclose(m0[m0.cutoff == 2017][cols].values, m1[m1.cutoff == 2017][cols].values, atol=1e-6), 'мутация без эмбарго не поймана'


def test_s2_ne_vidit_stroki_s_kontsom_celi_posle_cut():
    a, X, y, end = panel(); y2 = y.copy(); leak = (a.year == 2016).values & (pd.DatetimeIndex(end).year == 2017); y2[leak] = y2[leak] * 50
    s0 = H.run_h(a, X, y, end, steps=STEPS)['s2']; s1 = H.run_h(a, X, y2, end, steps=STEPS)['s2']
    np.testing.assert_allclose(s0[s0.cutoff == 2017][['s2_i', 's2_pool']].values, s1[s1.cutoff == 2017][['s2_i', 's2_pool']].values, atol=1e-12)
    rng = np.random.default_rng(1); n = 400; te = pd.to_datetime(['2016-06-01'] * 300 + ['2017-01-03'] * 100).values
    e = pd.DataFrame(dict(ticker='AAA', te=te, ep=np.r_[rng.normal(0, .1, 300), np.full(100, 5.)], et=np.r_[rng.normal(0, .1, 300), np.full(100, 5.)]))
    ok, _ = H.past_s2({2016: e}, 2017, 2016); bad, _ = H.past_s2({2016: e}, 2017, 2016, emb=False)
    assert ok < .02 and bad > .5, 'мутация s² без эмбарго не поймана'


def test_peremeshannye_tikery_lovyatsya():
    a, X, y, end = panel(); ly = np.log(y + 1e-10); rng = np.random.default_rng(3)
    a2 = a.copy(); a2['ticker'] = rng.permutation(a.ticker.values); end_y = np.asarray(pd.DatetimeIndex(end).year, float); tr = H.train_mask(a.year.values, end_y, y, 2017)
    cols = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']

    def dist(frame):
        b = H.run_h(frame, X, y, end, steps=STEPS)['beta']; b = b[b.cutoff == 2017].set_index('ticker')
        return max(np.abs(b.loc[tk, cols].values.astype(float) - lstsq(X, ly, tr & (a.ticker == tk).values)).max() for tk in TICKS)
    assert dist(a) < 1e-9 and dist(a2) > .05, 'перемешанные тикеры не пойманы'


def test_forecast_smeshivanie_s2():
    tick = np.array(['AAA', 'BBB']); lp = np.array([-8., -8.]); lt = np.array([-7., -9.])
    ln, p = H.forecast(tick, lp, lt, .04, {'AAA': .2}, .5)
    np.testing.assert_allclose(ln, [-7.5, -8.5]); np.testing.assert_allclose(p, np.exp(ln + .5 * np.array([.5 * .2 + .5 * .04, .5 * .04 + .5 * .04])))
    np.testing.assert_allclose(H.forecast(tick, lp, lt, .04, {'AAA': .2}, 1.)[0], lt); np.testing.assert_allclose(H.forecast(tick, lp, lt, .04, {'AAA': .2}, 0.)[1], np.exp(lp + .02))


def test_pervyj_shag_bez_popravki_i_malo_tochek_dayut_pul():
    assert H.past_s2({}, 2016, 2016) == (0., {})
    e = pd.DataFrame(dict(ticker=['AAA'] * 40 + ['BBB'] * 5, te=pd.Timestamp('2016-03-01'), ep=np.r_[np.zeros(40), np.ones(5)], et=np.r_[np.zeros(40), np.ones(5)]))
    s2p, si = H.past_s2({2016: e}, 2017, 2016); assert si['BBB'] == s2p and si['AAA'] == 0.


def test_ledger_i_granitsa_2022(tmp_path, monkeypatch):
    monkeypatch.setattr(H, 'EVENTS', tmp_path / 'ev.jsonl'); a, X, y, end = panel()
    H.run_h(a, X, y, end, steps=STEPS, rec=lambda k, vy, n, mx, dt=None: H.record(5, k, f'{vy}-01-01', n, mx, 'fit', dt))
    ev = [json.loads(s) for s in (tmp_path / 'ev.jsonl').read_text().splitlines()]; assert len(ev) == len(STEPS) * (1 + len(TICKS))
    assert all(pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01') for e in ev)
    with pytest.raises(AssertionError):
        H.record(5, 'x', '2017-01-01', 10, '2017-01-03')  # мутация: конец цели >= cut
    with pytest.raises(AssertionError):
        H.record(5, 'x', '2022-01-01', 10, '2021-12-30')
    b, Xb, yb, eb = panel(last='2022-03-31')
    with pytest.raises(AssertionError):
        H.run_h(b, Xb, yb, eb, steps=STEPS)  # мутация: данные ≥ 2022 в окне


def test_inwin_ravno_oknu_obucheniya():
    a, X, y, end = panel(); res = H.run_h(a, X, y, end, steps=STEPS); end_y = np.asarray(pd.DatetimeIndex(end).year, float)
    for vy in STEPS:
        w = res['inwin'][res['inwin'].cutoff == vy]; assert len(w) == H.train_mask(a.year.values, end_y, y, vy).sum() and w.date.max() < pd.Timestamp(f'{vy}-01-01')
        assert np.isfinite(w.filter(like='ln_').values).all() and len(w.filter(like='ln_').columns) == 5


def test_loyo_isklyuchaet_svoj_god():
    sc = {'lam0': {str(y): .5 for y in H.YEARS}, 'lam25': {str(y): (.0 if y == 2019 else .6) for y in H.YEARS}}
    year = np.repeat(H.YEARS, 2); preds = {k: np.full(len(year), float(i)) for i, k in enumerate(sc)}
    out, pick = H.loyo(sc, preds, year); assert pick[2019] == 'lam0' and pick[2018] == 'lam25' and out[year == 2019].tolist() == [0., 0.]
    mut = min(preds, key=lambda k: np.mean([sc[k][str(y)] for y in H.YEARS]))  # мутация: свой год в выборе
    assert mut == 'lam25' != pick[2019]
