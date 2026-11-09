"""X1-пилот P0, G2: масштаб RV. B(w ≡ 1) = RV_d побитно (путь `daily.build_daily`); мутация «выброшенный бар» ломает тождество; положительные w со средним 1 держат B/RV между min w и max w;
нормировка V7 масштаб RV не сохраняет (известный ответ: Σz = (RV − n·μ)/σ)."""
import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401
from models_v2 import daily as D
from models_v2.s16 import x1_p0_scale as S


def synth_bars(days=6, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(days):
        t0 = pd.Timestamp("2019-03-04") + pd.Timedelta(days=k)
        c = 100 * np.exp(np.cumsum(0.002 * rng.standard_normal(n)))
        for i in range(n):
            o = c[i - 1] if i else c[0]
            rows.append((t0 + pd.Timedelta(hours=10, minutes=10 * i), o, max(o, c[i]) * 1.001, min(o, c[i]) * 0.999, c[i], 1000.0 + i, 1e5))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume", "value"])


def rets(bars):
    r = D.intraday_returns(D.main_session(bars))
    return r["r"].to_numpy(), r["date"].to_numpy()


def test_b_w1_equals_daily_rv_bitwise():
    bars = synth_bars()
    d, _ = D.build_daily(bars, index_days=None)
    r, day = rets(bars)
    B = S.agg_B(r, day)
    assert np.array_equal(B.reindex(d.index).to_numpy(), d["rv"].to_numpy())


def test_mutation_dropped_bar_is_caught():
    bars = synth_bars()
    d, _ = D.build_daily(bars, index_days=None)
    r, day = rets(bars)
    w = np.ones(len(r))
    w[7] = 0.0                                     # выброшенный бар
    assert not np.array_equal(S.agg_B(r, day, w).reindex(d.index).to_numpy(), d["rv"].to_numpy())


def test_positive_mean_one_weights_bounded_and_not_exact():
    bars = synth_bars()
    r, day = rets(bars)
    rng = np.random.default_rng(1)
    w = np.exp(0.5 * rng.standard_normal(len(r)))
    w = w / pd.Series(w).groupby(day).transform("mean").to_numpy()
    ratio = (S.agg_B(r, day, w) / S.agg_B(r, day)).to_numpy()
    wg = pd.Series(w).groupby(day)
    assert (ratio >= wg.min().to_numpy() - 1e-12).all() and (ratio <= wg.max().to_numpy() + 1e-12).all()
    assert not np.allclose(ratio, 1.0)             # масштаб точно сохраняется только при w ≡ 1


def test_zero_returns_day_gives_zero_not_nan():
    assert S.agg_B(np.zeros(5), np.array([1] * 5)).iloc[0] == 0.0


def test_normalised_sum_does_not_keep_rv_scale_known_answer():
    bars = synth_bars(days=30)
    days = pd.DatetimeIndex(sorted(set(D.main_session(bars)["begin"].dt.normalize())))
    d, _ = D.build_daily(bars, index_days=None)
    store = S.SQ.build_store(bars, days)
    t = S.m10_norm_sums(store, d["rv"])
    assert len(t) > 0
    mu_n = t["mu_r2"] * 40
    expect = (t["rv"] - mu_n) * t["inv_sd_r2"]
    assert np.allclose(t["sum_z_r2"], expect, rtol=1e-9)
    assert not np.allclose(t["sum_z_r2"] / t["rv"], 1.0, atol=0.1)
