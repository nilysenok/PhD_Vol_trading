"""G10, G11: признаки этапа 2 (ручной пример) и HAR-признак PIT."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily, har  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402

A = 0.001
A2 = (A * 100) ** 2


def chain(n_days, rets_of, gaps=None, start="2020-01-06"):
    """n_days по 53 бара 10:00..18:40; r_i заданы rets_of(k); открытие дня k = close_{k-1}·exp(gap_k)."""
    gaps = gaps or {}
    days = pd.bdate_range(start, periods=n_days)
    frames, c0 = [], 100.0
    for k, day in enumerate(days):
        r = np.array(rets_of(k), float)
        o1 = c0 * np.exp(gaps.get(k, 0.0))
        close = o1 * np.exp(np.cumsum(r))
        opn = np.concatenate([[o1], close[:-1]])
        t0 = pd.Timestamp(str(day.date()) + " 10:00")
        frames.append(pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * j) for j in range(53)], "open": opn,
                                    "high": np.maximum(opn, close), "low": np.minimum(opn, close), "close": close,
                                    "volume": 1, "value": 1.0}))
        c0 = close[-1]
    return pd.concat(frames, ignore_index=True), days


def rets_k(k):
    r = np.array([A if j % 2 == 0 else -A for j in range(53)])
    if k == 29:
        r[:6] = 2 * A      # первый час: 6 баров по 2A
        r[-6:] = -3 * A    # последний час: 6 баров по -3A
    return r


def test_hand_example_hours_semivariances_gap_dow():
    gap = 0.01
    bars, days = chain(30, rets_k, gaps={29: gap})
    F = ft.build(bars, min_pairs=10)
    row = F.X[29]
    r = rets_k(29)
    r2 = (100 * r) ** 2
    assert np.isclose(row[5], np.log(6 * 4 * A2)) and np.isclose(row[6], np.log(6 * 9 * A2))
    assert np.isclose(row[7], r2[r > 0].sum()) and np.isclose(row[8], r2[r < 0].sum())
    assert np.isclose(row[7] + row[8], r2.sum())
    rv_m = F.rv[8:30].mean()
    assert np.isclose(row[9], (100 * gap) ** 2 / rv_m)
    assert row[10] == days[29].dayofweek and F.X[22, 10] == days[22].dayofweek
    assert np.isclose(row[0], np.log(r2.sum()))


def test_gap_feature_is_nan_before_rv_m_and_without_previous_day():
    bars, _ = chain(30, rets_k)
    F = ft.build(bars, min_pairs=10)
    assert np.isnan(F.X[0, 9]) and np.isnan(F.X[10, 9]) and np.isfinite(F.X[25, 9])


def test_no_future_in_features_row_t():
    bars, days = chain(40, rets_k)
    full = ft.build(bars, min_pairs=10)
    cut = bars[bars["begin"].dt.normalize() <= days[30]]
    part = ft.build(cut, min_pairs=10)
    assert np.array_equal(full.X[:31], part.X, equal_nan=True)


def test_har_feature_equals_stage1_forecast_and_is_pit():
    rng = np.random.default_rng(5)
    bars, days = chain(330, lambda k: rng.normal(0, 0.001 * (1 + 0.5 * np.sin(k / 7)), 53))
    F = ft.build(bars)
    p = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    fin = np.flatnonzero(np.isfinite(F.X[:, ft.I_HAR]))
    # признаки с индекса 21 (22-й день); ровно 252 пары к индексу 273; раньше — NaN
    assert fin[0] == 21 + 252 and len(fin) == 330 - fin[0]
    for i in (fin[0], fin[10], fin[-1]):
        want = np.log(har.forecast_at(p, int(i))["harj_b"])
        assert abs(F.X[i, ft.I_HAR] - want) < 1e-12
    cut = bars[bars["begin"].dt.normalize() <= days[fin[10]]]
    G = ft.build(cut)
    assert np.array_equal(G.X[:, ft.I_HAR], F.X[:fin[10] + 1, ft.I_HAR], equal_nan=True)
