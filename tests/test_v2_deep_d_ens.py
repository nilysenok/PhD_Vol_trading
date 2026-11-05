"""G3 D: правила ансамблей — PIT (закрытые прогнозы), симплекс, E2 ≡ C (MD15), режимы (MD4), inverse (MD5), средние (MD6), MD1. Синтетика, без сети."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s5.b1 import rules
from models_v2.s5.d import ens

N = 700


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(5)
    d = pd.bdate_range("2019-01-02", periods=N)
    h = np.exp(rng.normal(0, 0.4, N))
    x = h * np.exp(rng.normal(0, 0.3, N))
    l_ = h * np.exp(rng.normal(0.1, 0.3, N))
    rv = h * np.exp(rng.normal(0, 0.6, N))
    td = d[1:].append(pd.DatetimeIndex([d[-1] + pd.Timedelta(days=1)]))
    ctx = {"rv": rv, "h": h, "d": d.to_numpy("datetime64[ns]"), "td": td.to_numpy("datetime64[ns]"),
           "st_g4": rng.random(N) > 0.7, "st_rv": rng.random(N) > 0.6}
    return ctx, {"X": x, "L": l_, "Xr": x * 1.02, "Xh": x * 0.98}


def test_simplex_sums_to_one_and_lexicographic():
    for K in (2, 3):
        G = ens.simplex(K)
        assert np.allclose(G.sum(axis=1), 1) and (G >= 0).all() and len(G) == {2: 11, 3: 66}[K]
        assert (G == G[np.lexsort(G.T[::-1])]).all()


def test_e2_equals_c_h3_bitwise(world):
    ctx, M = world
    for kind in ("roll", "exp"):
        wc = rules.h3_weights(kind, ctx["rv"], ctx["h"], M["X"], ctx["d"], ctx["td"])
        c = rules.mix(wc, ctx["h"], M["X"])
        S = np.column_stack([ctx["h"], M["X"]])
        w = ens.w_window(ctx["rv"], S, ens.closed(ctx), 252 if kind == "roll" else 0)
        assert np.allclose(w[:, 0], wc, rtol=0, atol=1e-15) and len(np.unique(wc)) > 2   # веса те же (MD15); не вырожденный случай
        assert np.allclose(np.einsum("nk,nk->n", w, S), c, rtol=1e-12, atol=0)


def test_equal_weights_before_252_closed_and_weights_valid(world):
    ctx, M = world
    for row in ens.TRIALS:
        s = ens.build(row, ctx, M)
        assert np.isfinite(s).all() and (s > 0).all()
    w = ens.w_window(ctx["rv"], ens.stack(("H", "X", "L"), ctx, M), ens.closed(ctx), 252)
    assert np.allclose(w[:252], 1 / 3) and np.allclose(w.sum(axis=1), 1)


def test_md1_unclosed_forecast_changes_weights(world):
    ctx, M = world
    a = ens.build("e4_roll", ctx, M)
    b = ens.build("e4_roll", ctx, M, extra_closed=1)
    assert not np.array_equal(a, b)   # мутант (пара d в окне) даёт другие прогнозы — тест его ловит


def test_weights_at_d_ignore_forecast_origin_d_and_later(world):
    ctx, M = world
    k = 600
    S = ens.stack(("H", "X", "L"), ctx, M)
    base = ens.w_window(ctx["rv"], S, ens.closed(ctx), 252)[k]
    rv2, S2 = ctx["rv"].copy(), S.copy()
    rv2[k:], S2[k:] = rv2[k:] * 9, S2[k:] * 0.2     # порча прогнозов/RV, закрывающихся после d
    assert np.array_equal(ens.w_window(rv2, S2, ens.closed(ctx), 252)[k], base)


def test_inverse_weights_follow_past_qlike(world):
    ctx, M = world
    S = ens.stack(("H", "X"), ctx, M)
    m = ens.closed(ctx)
    w = ens.w_inverse(ctx["rv"], S, m)
    k = 650
    q = metrics.qlike(ctx["rv"][:, None], S)[m[k] - 252:m[k]].mean(axis=0)
    assert np.allclose(w[k], (1 / q) / (1 / q).sum()) and w[k].argmax() == q.argmin()
    w_bad = ens.w_inverse(ctx["rv"], S, np.minimum(m + 1, N))   # MD5: окно с парой d
    assert not np.allclose(w_bad[k], w[k])


def test_regime_weights_use_state_at_d_not_d_plus_1(world):
    ctx, M = world
    S = ens.stack(("H", "X", "L"), ctx, M)
    m = ens.closed(ctx)
    ok = ens.w_regime(ctx["rv"], S, m, ctx["st_g4"])
    bad = ens.w_regime(ctx["rv"], S, m, np.append(ctx["st_g4"][1:], False))   # MD4: режим d+1
    assert not np.array_equal(ok, bad) and np.allclose(ok.sum(axis=1), 1)


def test_regime_group_has_only_same_group_closed(world):
    ctx, M = world
    S = ens.stack(("H", "X"), ctx, M)
    m = ens.closed(ctx)
    grp = ctx["st_g4"]
    w = ens.w_regime(ctx["rv"], S, m, grp)
    k = 650
    G = ens.simplex(2)
    j = np.arange(m[k])
    j = j[grp[j] == grp[k]]
    tot = metrics.qlike(ctx["rv"][j][:, None], np.einsum("nk,pk->np", S[j], G)).sum(axis=0)
    assert np.array_equal(w[k], ens.pick(tot, G))


def test_md6_log_mean_not_above_arithmetic(world):
    ctx, M = world
    for s in ("xl", "hx", "hl", "hxl", "xrh"):
        g, a = ens.build("mean_log_" + s, ctx, M), ens.build("mean_lvl_" + s, ctx, M)
        assert (g <= a + 1e-12).all() and (g < a).any()   # мутант (логарифмическое = арифметическое) не проходит


def test_spec_counts_16_trials_10_controls():
    assert len(ens.TRIALS) == 13 and sum(r.startswith("mean_") for r in ens.SPEC) == 10
    assert len(set(ens.TRIALS) | {"d_lgbm_f4", "d_xgb_f4_r1008", "d_xgb_f4_hl504"}) == 16
