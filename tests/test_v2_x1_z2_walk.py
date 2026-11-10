"""Z2: помесячная переоценка H1 — PIT (закрыто + 1, окно ступени 1, ξ с барами t⁺, согласованные координаты № 146), a = 0 ⇒ F = X1 побитно, включение, будущие бары не меняют прогнозы месяца."""
import math

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_z2_hawkes as HK, x1_z2_synth as SY, x1_z2_walk as WK, x1_z_walk as W


@pytest.fixture(scope="module")
def base():
    sim = SY.simulate(0.0015, 0.5, math.log(2) / 80, 0.5, 0.6, 700, seed=21)
    ctx = sim["ctx"]
    x1, e = SY.x1_frame(ctx, seed=22)
    d = pd.DatetimeIndex(x1["d"])
    starts = [p for p in W.month_starts(d) if d[p] >= W.ACTIVATION]
    return dict(ctx=ctx, x1=x1, e=e, d=d, starts=starts, fits=WK.stage1(ctx, d), out=None)


def run(b, ctx=None, x1=None, fits=None):
    ctx = ctx or b["ctx"]
    return WK.walk(x1 if x1 is not None else b["x1"], ctx, fits or WK.stage1(ctx, b["d"]), b["e"])


def first_month(b):
    return slice(b["starts"][0], b["starts"][1])


def test_walk_shape_activation_and_g_formula(base):
    out = run(base, fits=base["fits"])
    a, b = base["starts"][0], base["starts"][1]
    assert (out["active"].iloc[:a] == 0).all() and (out["active"].iloc[a:b] == 1).all()
    assert np.array_equal(out["F"].iloc[:a], base["x1"]["F"].iloc[:a])                # до включения F = X1 побитно
    assert np.allclose(out["g"].iloc[a:b], out["a"].iloc[a:b] * np.log1p(out["xi"].iloc[a:b]))
    assert (out["xi"].iloc[a:b] >= 0).all()


def test_not_enough_extreme_pairs_means_no_inclusion(base):
    out = WK.walk(base["x1"], base["ctx"], base["fits"], np.zeros(len(base["x1"]), int))
    assert (out["active"] == 0).all() and np.array_equal(out["F"], base["x1"]["F"])


def f_equals_x1(out) -> bool:
    return bool(np.array_equal(out["F"].to_numpy(), out["X1"].to_numpy()) and (out["g"] == 0).all())


def test_a_zero_gives_x1_bitwise_and_stray_constant_mutation_caught(base, monkeypatch):
    monkeypatch.setattr(WK, "refit", lambda pool, u, x: (0.0, 0.0))
    assert f_equals_x1(run(base, fits=base["fits"]))
    monkeypatch.setattr(WK, "forecasts", lambda X1, g: X1 * np.exp(g + 1e-9))
    assert not f_equals_x1(run(base, fits=base["fits"]))


def test_future_bars_do_not_change_forecasts_of_month(base):
    s = first_month(base)
    ctx = dict(base["ctx"], N=base["ctx"]["N"].copy(), m=base["ctx"]["m"].copy())
    ctx["N"][s.stop + 1:], ctx["m"][s.stop + 1:] = 1.0, 2.0
    a = run(base, fits=base["fits"])
    b = run(base, ctx=ctx, fits=WK.stage1(ctx, base["d"]))
    assert np.array_equal(a["F"].iloc[: s.stop], b["F"].iloc[: s.stop])


def test_closed_plus_one_mutation_caught(base, monkeypatch):
    s = first_month(base)
    x1 = base["x1"].copy()
    x1.loc[x1.index[s.start], "rv"] *= 50.0                # RV пары дня D = d[a] известен только в t⁺ > D
    honest = run(base, fits=base["fits"])
    assert np.array_equal(honest["F"].iloc[: s.stop], run(base, x1=x1, fits=base["fits"])["F"].iloc[: s.stop])
    monkeypatch.setattr(W, "closed_date", lambda tp, d: d)   # «закрыто + 1»: пара видна в день прогноза
    a, b = run(base, fits=base["fits"]), run(base, x1=x1, fits=base["fits"])
    assert not np.array_equal(a["F"].iloc[s], b["F"].iloc[s])


def test_stage1_window_excludes_session_D_and_mutation_caught(base, monkeypatch):
    a = base["starts"][0]
    ctx = dict(base["ctx"], N=base["ctx"]["N"].copy(), m=base["ctx"]["m"].copy())
    ctx["N"][a], ctx["m"][a] = 1.0, 3.0                       # бары сессии D
    pos = a                                                    # склад и x1 на одних датах
    t0, t1 = base["fits"][a][0], WK.stage1(ctx, base["d"])[a][0]
    assert t0 == t1                                            # θ_D по барам < D: побитно
    monkeypatch.setattr(HK, "win_end", lambda p: p + 1)
    assert WK.stage1(ctx, base["d"])[a][0] != WK.stage1(base["ctx"], base["d"])[a][0] and pos == a


def test_xi_of_day_d_ignores_bars_of_t_plus_and_mutation_caught(base, monkeypatch):
    th, f = base["fits"][base["starts"][0]]
    p = base["starts"][0] + 3
    ctx = dict(base["ctx"], N=base["ctx"]["N"].copy(), m=base["ctx"]["m"].copy())
    ctx["N"][p + 1], ctx["m"][p + 1] = 1.0, 2.0
    xi = lambda c: HK.xi_all(th, c, f, c["dnext"])[p]
    assert xi(base["ctx"]) == xi(ctx)
    monkeypatch.setattr(HK, "xi_row", lambda q: q + 1)
    assert xi(base["ctx"]) != xi(ctx)


def test_pool_xi_uses_theta_D_for_all_rows_and_own_month_mutation_caught(base, monkeypatch):
    a = base["starts"][1]                                      # второй месяц: в пуле строки первого месяца
    seen = {}
    real = WK.pool_xi

    def spy(par, ctx, f, pos):
        out = real(par, ctx, f, pos)
        seen[len(seen)] = (par, f, out)
        return out
    monkeypatch.setattr(WK, "pool_xi", spy)
    honest = run(base, fits=base["fits"])
    par, f, out = list(seen.values())[1]
    assert par == base["fits"][a][0] and np.array_equal(out, HK.xi_all(par, base["ctx"], f, base["ctx"]["dnext"])[base["ctx"]["days"].get_indexer(base["d"])])
    blocks = list(zip(base["starts"][:-1], base["starts"][1:]))

    def own_month(par, ctx, f, pos):                            # мутант № 146: ξ_j от θ СВОЕГО месяца
        res = real(par, ctx, f, pos)
        for s0, s1 in blocks:
            th, ff = base["fits"][s0]
            res[s0:s1] = HK.xi_all(th, ctx, ff, ctx["dnext"])[pos[s0:s1]]
        return res
    monkeypatch.setattr(WK, "pool_xi", own_month)
    assert not np.allclose(run(base, fits=base["fits"])["g"].to_numpy(), honest["g"].to_numpy(), equal_nan=True) or not np.allclose(run(base, fits=base["fits"])["a"].to_numpy(), honest["a"].to_numpy(), equal_nan=True)
