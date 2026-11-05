"""G8, G2, G3: OLS на известном ответе, пары без утечки и без склейки, сетка дат."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily, har, metrics  # noqa: E402


def test_ols_hand_example():
    beta, s2, n = har.ols(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([1.0, 3.0, 4.0, 8.0]))
    assert np.allclose(beta, [0.7, 2.2], atol=1e-12) and abs(s2 - 0.9) < 1e-12 and n == 4


def test_ols_recovers_known_beta():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50000, 4))
    b = np.array([0.3, 0.4, 0.35, 0.1, -0.2])
    y = b[0] + X @ b[1:] + 0.5 * rng.normal(size=50000)
    beta, s2, _ = har.ols(X, y)
    assert np.allclose(beta, b, atol=0.01) and abs(s2 - 0.25) < 0.01
    beta0, s20, _ = har.ols(X[:100], b[0] + X[:100] @ b[1:])
    assert np.allclose(beta0, b, atol=1e-9) and s20 < 1e-18


def test_ols_refuses_short_window():
    with pytest.raises(ValueError):
        har.ols(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        har.ols(np.zeros((0, 1)), np.zeros(0))


def test_predict_bias_correction():
    a, b = har.predict(np.array([0.1, 0.5]), np.array([2.0]), 0.4)
    assert np.isclose(a, np.exp(1.1)) and np.isclose(b / a, np.exp(0.2))


def _panel(n_pre=60, n_post=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = [x for x in pd.bdate_range("2021-12-01", periods=400) if not (daily.GAP[0] <= x <= daily.GAP[1])]
    dates = pd.DatetimeIndex(dates[:n_pre] + [x for x in dates if x > daily.GAP[1]][:n_post])
    df = pd.DataFrame({"rv": np.exp(rng.normal(-8.5, 0.5, len(dates))), "j": np.exp(rng.normal(-12, 1, len(dates)))},
                      index=dates)
    return har.make_panel(daily.add_windows(df))


def test_pairs_never_use_future_and_never_cross_gap():
    p = _panel()
    brk = int(np.flatnonzero(np.diff(p.seg))[0])  # последняя строка до разрыва
    for i in (30, brk, brk + 1, brk + 30, len(p.rv) - 1):
        Xt, yt = har.pairs(p, i, har.COLS_HARJ)
        t = np.array([k for k in range(i) if np.isfinite(p.X[k]).all() and p.seg[k] == p.seg[k + 1]])
        assert len(yt) == len(t) and np.array_equal(yt, p.lnrv[t + 1]) and t.max() + 1 <= i
        assert brk not in t  # пара (последний день до разрыва -> первый после) не строится


def test_grid_has_no_pair_across_gap_and_waits_for_windows():
    p = _panel()
    g = har.eligible(p, first_target="2000-01-01")
    brk = int(np.flatnonzero(np.diff(p.seg))[0])
    assert brk not in g
    post = [i for i in g if p.seg[i] == 1]
    assert min(post) == brk + 1 + 21  # RV_m набран на 22-й день сегмента
    assert all(p.seg[i] == p.seg[i + 1] for i in g)


def test_forecast_at_baselines_and_models():
    p = _panel()
    i = 50
    f = har.forecast_at(p, i)
    assert set(f) == {"rw", "rv_w", "rv_m", "har_a", "har_b", "harj_a", "harj_b"}
    assert f["rw"] == p.rv[i] and np.isclose(f["rv_w"], p.rv[i - 4:i + 1].mean())
    assert np.isclose(f["rv_m"], p.rv[i - 21:i + 1].mean())
    assert f["har_b"] > f["har_a"] and f["harj_b"] > f["harj_a"]


def test_forecast_table_one_grid_for_all_models():
    p = _panel()
    t = har.forecast_table(p, first_target=str(p.dates[45])[:10])
    assert t.groupby("model")["d"].apply(lambda s: tuple(s)).nunique() == 1 and t["model"].nunique() == 7
    assert (t["target_date"] > t["d"]).all()


def test_qlike_hand_values():
    assert metrics.qlike([1.0], [1.0])[0] == 0.0
    assert np.isclose(metrics.qlike([2.0], [1.0])[0], 2 - np.log(2) - 1)
    s = metrics.summary([2.0, 1.0], [1.0, 1.0])
    assert np.isclose(s["mean_r"], 1.5) and np.isclose(s["rmse_ln"], np.log(2) / np.sqrt(2))
    with pytest.raises(ValueError):
        metrics.summary([], [])
