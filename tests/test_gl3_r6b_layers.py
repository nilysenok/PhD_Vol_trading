# -*- coding: utf-8 -*-
"""Слои раунда 6б h=5 (Г9–Г12, Г14, Г15): синтетика, известный ответ, подложенные ошибки (У13). Без сети, без данных >= 2022."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/analysis'))
import gl3_r6b_layers as L

I0 = 800  # позиция проверяемой строки внутри тикера; late(a) — строки с позицией >= I0 в своём тикере


def panel(n=1500, tks=('A', 'B', 'C'), seed=0):
    rng = np.random.default_rng(seed); dates = pd.bdate_range('2014-01-01', periods=n)
    a = pd.concat([pd.DataFrame(dict(date=dates, ticker=t)) for t in tks], ignore_index=True)
    a['_end5'] = a.groupby('ticker').date.shift(-5); a['year'] = a.date.dt.year
    a['close_RVI'] = a.date.map(pd.Series(np.exp(rng.normal(3, .3, n)), index=dates)); a['rv_w'] = np.exp(rng.normal(-8, .4, len(a)))
    return a


def late(a):
    return (a.groupby('ticker').cumcount() >= I0).values


def noisy(a, seed=1):
    rng = np.random.default_rng(seed); fin = np.exp(rng.normal(-8, .3, len(a)))
    return fin, fin * np.exp(rng.normal(0, .5, len(a))), fin * np.exp(rng.normal(0, .5, len(a)))  # финал, y, HAR-J


def run_layer(name, a, y, fin, harj, state):
    return {'10': lambda: L.layer10(a, fin, y, state, .5), '14': lambda: L.layer14(a, fin, y, 20, .5), '15': lambda: L.layer15(a, fin, harj, y)}[name]()


def shuffle_tickers(monkeypatch):
    """Мутация У13: множитель одного тикера применяется к строкам другого (результаты потикерных окон сдвинуты на блок)."""
    orig = L.by_ticker
    monkeypatch.setattr(L, 'by_ticker', lambda a, x, fn: np.roll(orig(a, x, fn), len(a) // a.ticker.nunique()))


@pytest.mark.parametrize('name', ['10', '14', '15'])
def test_closed_window_no_leak_and_mutation_s_equals_d_minus_4(name, monkeypatch):
    a = panel(); fin, y, harj = noisy(a); state = np.random.default_rng(2).random(len(a)) > .5
    i = a.index[a.ticker == 'A'][I0]; y2 = y.copy(); y2[i - 4] *= 8  # цель строки i-4 заканчивается в i+1 > d
    base = run_layer(name, a, y, fin, harj, state)[i]
    np.testing.assert_allclose(run_layer(name, a, y2, fin, harj, state)[i], base, rtol=1e-12)
    y3 = y.copy(); y3[i - 5] *= 8  # цель строки i-5 заканчивается в i: уже известна
    assert run_layer(name, a, y3, fin, harj, state)[i] != pytest.approx(base, rel=1e-9)
    monkeypatch.setattr(L, 'LAG', 4)
    assert run_layer(name, a, y2, fin, harj, state)[i] != pytest.approx(run_layer(name, a, y, fin, harj, state)[i], rel=1e-9)  # мутация ловится


def test_regime_rvi_uses_only_past_and_mutation_next_day(monkeypatch):
    a = panel(); d0 = a.date.unique()[900]; d1 = a.date.unique()[901]
    a.loc[a.date == d0, 'close_RVI'] = 1.; a.loc[a.date == d1, 'close_RVI'] = 1e6
    m = (a.date == d0).values
    assert not L.stress_rvi(a)[m].any()
    monkeypatch.setattr(L, 'RVI_LEAD', 1)
    assert L.stress_rvi(a)[m].all() and L.stress_hard(a)[m].all()  # режим по RVI дня d+1 — мутация ловится


def test_regime_rvw_quantile_excludes_day_d():
    a = panel(); d = a.date.unique()[900]; m = (a.date == d).values
    w, q = L.rvw_pair(a); a2 = a.copy(); a2.loc[m, 'rv_w'] = 1e3
    w2, q2 = L.rvw_pair(a2)
    np.testing.assert_allclose(q2[m], q[m])  # день d не входит в квантиль
    assert (w2[m] > q2[m]).all() and L.stress_hard(a2)[m].all()
    a3 = a.copy(); a3.loc[a.date == a.date.unique()[901], 'rv_w'] = 1e3
    np.testing.assert_allclose(L.rvw_pair(a3)[1][m], q[m])  # будущее не входит


def test_known_answer_rv_equals_c_times_sigma2_k_converges():
    a = panel(); fin, _, _ = noisy(a); c = 1.3; y = c * fin; st = np.arange(len(a)) % 7 < 3; m = late(a)
    for p in (L.layer10(a, fin, y, st, .5), L.layer14(a, fin, y, 20, .5), L.layer14(a, fin, y, 120, .75)):
        np.testing.assert_allclose(p[m], y[m], rtol=1e-9)  # k = c, QLIKE = 0


def test_known_answer_equal_qlike_gives_weight_half():
    a = panel(); fin, y, _ = noisy(a); r1 = 2.; q = r1 - np.log(r1) - 1
    r2 = brentq(lambda r: r - np.log(r) - 1 - q, 1e-6, 1 - 1e-9); fin = y / r1; harj = y / r2; m = late(a)
    p = L.layer15(a, fin, harj, y)
    np.testing.assert_allclose(p[m], .5 * fin[m] + .5 * harj[m], rtol=1e-9)  # Q^hyb = Q^HAR -> w = 0,5


def test_ticker_shuffle_caught_g10_g14(monkeypatch):
    a = panel(); fin, _, _ = noisy(a); y = fin * a.ticker.map({'A': .7, 'B': 1.3, 'C': 1.4}).values; st = np.zeros(len(a), bool); m = late(a)
    for f in (lambda: L.layer10(a, fin, y, st, 1.), lambda: L.layer14(a, fin, y, 20, 1.)):
        np.testing.assert_allclose(f()[m], y[m], rtol=1e-9)
    shuffle_tickers(monkeypatch)
    assert not np.allclose(L.layer10(a, fin, y, st, 1.)[m], y[m], rtol=1e-3)
    assert not np.allclose(L.layer14(a, fin, y, 20, 1.)[m], y[m], rtol=1e-3)


def test_g15_per_ticker_weights_and_shuffle_caught(monkeypatch):
    a = panel(tks=('A', 'B')); _, y, _ = noisy(a)  # A: финал точен, HAR-J вдвое мимо; B: наоборот
    fin = np.where(a.ticker == 'A', y, 2 * y); harj = np.where(a.ticker == 'A', 2 * y, y); isa = (a.ticker == 'A').values; m = late(a)
    wa = (L.layer15(a, fin, harj, y) - harj) / (fin - harj)
    np.testing.assert_allclose(wa[isa & m], .75, rtol=1e-9); np.testing.assert_allclose(wa[~isa & m], .25, rtol=1e-9)
    orig = L.roll_mean  # оба Q (гибрид и HAR-J) уходят к соседнему тикеру
    monkeypatch.setattr(L, 'roll_mean', lambda a_, x, n: np.roll(orig(a_, x, n), len(a_) // 2))
    wm = (L.layer15(a, fin, harj, y) - harj) / (fin - harj)
    assert not np.allclose(wm[isa & m], .75, rtol=1e-3)


def g11_case():
    a = panel(); fin, _, _ = noisy(a); s = a.ticker.map({'A': .4, 'B': 1.2, 'C': 0.}).values; sign = np.where(np.arange(len(a)) % 2 == 0, 1., -1.)
    return a, fin, fin * np.exp(s * sign), np.log(fin)


def test_g11_known_answer_clip_and_mutation(monkeypatch):
    a, fin, y, raw = g11_case(); m18 = (a.year == 2018).values; t = a.ticker.values[m18]
    p, ev = L.layer11(a, fin, y, raw); r = p[m18] / fin[m18]
    np.testing.assert_allclose(r[t == 'A'], np.exp(.16 / 2), rtol=2e-3)
    np.testing.assert_allclose(r[t == 'B'], 1.25, rtol=1e-12)  # exp(.72)=2,05 обрезано до 1,25
    np.testing.assert_allclose(r[t == 'C'], 1., rtol=1e-12)
    assert all(e['h'] == 5 and pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) for e in ev if e['phase'] == 'calibration')
    monkeypatch.setattr(L, 'K11_CLIP', (0., np.inf))
    assert not np.allclose((L.layer11(a, fin, y, raw)[0][m18] / fin[m18])[t == 'B'], 1.25, rtol=1e-3)  # обрезка снята — ловится


def test_g11_ticker_shuffle_caught(monkeypatch):
    a, fin, y, raw = g11_case(); m18 = (a.year == 2018).values; t = a.ticker.values[m18]
    base = L.layer11(a, fin, y, raw)[0][m18] / fin[m18]; orig = L._grp; calls = []
    def shifted(x):
        calls.append(1); v = orig(x); return v if len(calls) % 2 else np.roll(v, len(x) // 3)  # обучение — верные тикеры, применение — чужие
    monkeypatch.setattr(L, '_grp', shifted)
    bad = L.layer11(a, fin, y, raw)[0][m18] / fin[m18]
    assert not np.allclose(bad[t == 'C'], base[t == 'C'], rtol=1e-3)


def test_g11_no_future_target_in_calibration():
    a = panel(); fin, y, _ = noisy(a); raw = np.log(fin); cut = pd.Timestamp('2018-01-01')
    base = L.layer11(a, fin, y, raw)[0]; m18 = (a.year == 2018).values
    lat = (a._end5 >= cut).values & (a.date < cut).values; assert lat.any()
    y2 = y.copy(); y2[lat] *= 5
    np.testing.assert_allclose(L.layer11(a, fin, y2, raw)[0][m18], base[m18], rtol=1e-12)
    early = (a._end5 < cut).values & (a.date >= cut - pd.DateOffset(years=1)).values & (a.date < cut).values; y3 = y.copy(); y3[early] *= 5
    assert not np.allclose(L.layer11(a, fin, y3, raw)[0][m18], base[m18], rtol=1e-6)


def test_g11_without_raw_uses_level_and_falls_back_without_history():
    a = panel(); fin, y, _ = noisy(a); p, ev = L.layer11(a, fin, y, None)
    assert np.isfinite(p[(a.year == 2018).values]).all()
    b = a[a.year >= 2017].reset_index(drop=True); f2, y2, _ = noisy(b); q, ev2 = L.layer11(b, f2, y2, None)
    m17 = (b.year == 2017).values; np.testing.assert_allclose(q[m17], f2[m17]); assert ev2[0]['phase'] == 'fallback_no_history'


def test_g14_clip_and_mutation(monkeypatch):
    a = panel(); fin, _, _ = noisy(a); y = 3. * fin; m = late(a)
    np.testing.assert_allclose(L.layer14(a, fin, y, 20, 1.)[m], 1.5 * fin[m], rtol=1e-12)
    np.testing.assert_allclose(L.layer14(a, fin, .2 * fin, 20, 1.)[m], .7 * fin[m], rtol=1e-12)
    monkeypatch.setattr(L, 'K14_CLIP', (0., np.inf))
    assert not np.allclose(L.layer14(a, fin, y, 20, 1.)[m], 1.5 * fin[m], rtol=1e-3)


def wave_case(n=1500):
    a = panel(n); fin, y, _ = noisy(a); wave = np.where((np.arange(len(a)) // 50) % 2 == 0, 1., -1.)
    calm = np.where(wave > 0, y * 1.6, y); f = np.where(wave > 0, y, y * 1.6)  # в стрессе точен финал, в покое — спокойная модель
    z = np.column_stack([wave + np.random.default_rng(4).normal(0, .8, len(a)), np.random.default_rng(3).normal(size=len(a))])
    return a, y, f, calm, z, wave


def test_logistic_learns_regime_ledger_closed_and_hard_fallback():
    a, y, f, calm, z, wave = wave_case(); hard = wave > 0
    p, ev = L.gate_logistic(a, f, calm, y, hard, z, 'G9')
    assert len([e for e in ev if e['phase'] == 'fit']) >= 1
    for e in ev:
        assert e['h'] == 5 and (e['max_target_end'] is None or pd.Timestamp(e['max_target_end']) < pd.Timestamp(e['cutoff']) < pd.Timestamp('2022-01-01'))
    m = (a.year == 2018).values; g = (p - calm) / (f - calm)
    assert g[m & hard].mean() > g[m & ~hard].mean() + .3  # вес финала выше в стрессе
    k = (a.year >= 2017).values; short = a[k].reset_index(drop=True)
    p2, ev2 = L.gate_logistic(short, f[k], calm[k], y[k], wave[k] > 0, z[k], 'G9'); m17 = (short.year == 2017).values
    assert ev2[0]['phase'] == 'fallback_hard'; np.testing.assert_allclose(p2[m17], L.hard_gate(f[k], calm[k], wave[k] > 0)[m17])


def test_hard_gate_and_loyo_excludes_held_year():
    np.testing.assert_array_equal(L.hard_gate(np.array([1., 2.]), np.array([5., 6.]), np.array([True, False])), [1., 6.])
    a = pd.DataFrame(dict(year=[2017, 2018, 2019, 2020, 2021]))
    sc = {'a': {'2017': -5., '2018': 1., '2019': 1., '2020': 1., '2021': 1.}, 'b': {str(y): .6 for y in L.YEARS}}
    preds = {'a': np.arange(5.), 'b': np.arange(5.) + 10}
    out, pick = L.loyo(['a', 'b'], sc, preds, a)
    assert pick[2017] == 'b' and pick[2018] == 'a'  # мутация «с учётом held-года» выбрала бы a для 2017
    assert out[0] == 10. and out[1] == 1.


def test_scorer_row_matches_manual():
    a = panel(2000); fin, y, _ = noisy(a); idx = np.where((a.year.values >= 2017) & (a.year.values <= 2021))[0]
    assert set(a.year.values[idx]) == set(L.YEARS)
    sc = L.Scorer(y[idx], a.year.values[idx], a.ticker.values[idx], idx, a.date.values[idx] < np.datetime64('2017-03-01'),
                  a.date.values[idx] < np.datetime64('2017-02-01'), fin)
    r = sc.row('x', True, fin); assert r['better_years'] == 0 and not r['admitted'] and r['n_better_tickers'] == 0
    r2 = sc.row('y', True, y); assert r2['mean_QLIKE'] < 1e-12 and r2['n_better_tickers'] == 3
