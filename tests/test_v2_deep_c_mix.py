"""G5 C: смеси H3 и переключатели над готовыми прогнозами — только закрытые прогнозы и прошлые годы; режим по прошлому (MC8–MC12, № 22)."""
import numpy as np
import pandas as pd

from models_v2.s5.c import mix

N, K = 420, 330


def ctx(seed=3):
    r = np.random.default_rng(seed)
    d = np.arange("2019-01-01", N, dtype="datetime64[D]").astype("datetime64[ns]")
    c = {"rv": np.exp(r.normal(0, 1, N)), "h": np.exp(r.normal(0, 1, N)), "d": d, "td": d + np.timedelta64(1, "D"),
         "qw": r.integers(0, 5, N).astype(float), "ty": np.where(np.arange(N) < N // 2, 2020, 2021)}
    c["grp"] = (c["qw"] >= 4).astype(float)
    return c, np.exp(r.normal(0, 1, N))


def corrupt(c, b, lo):
    c2, b2 = dict(c), b.copy()
    c2["rv"] = c["rv"].copy()
    c2["rv"][lo:] *= 50
    c2["h"] = c["h"].copy()
    c2["h"][lo + 1:] *= 30
    b2[lo + 1:] *= 30
    return c2, b2


def test_connection_h3_uses_only_closed_forecasts():
    """Подключение H3 (b1.h3_rows не покрыт тестами): σ̂² на K не меняется при порче всего, что известно позже d_K; мутант td = d ловится."""
    c, b = ctx()
    c2, b2 = corrupt(c, b, K)
    for kind in ("exp", "roll", "reg"):
        a, z = mix.h3(kind, c, b), mix.h3(kind, c2, b2)
        assert a[K] == z[K], kind
        mut = dict(c, td=c["d"])                         # мутация td = d: незакрытые прогнозы в окне
        mut2 = dict(c2, td=c2["d"])
        assert mix.h3(kind, mut, b)[K] != mix.h3(kind, mut2, b2)[K], kind
    assert (mix.times(pd.DataFrame({"d": pd.to_datetime(c["d"]), "target_date": pd.to_datetime(c["td"])}))[1] != c["d"]).all()


def test_extra_closed_mutant_caught():
    c, b = ctx()
    c2, b2 = corrupt(c, b, K)
    assert mix.h3("roll", c, b, 1)[K] != mix.h3("roll", c2, b2, 1)[K]


def test_r4_theta_uses_only_past_years_and_mutant_include_year_caught():
    c, _ = ctx()
    alt = np.full(N, 10.0)
    c["h"] = np.ones(N)
    c["rv"] = np.where(c["ty"] == 2020, 1.0, 10.0)
    ok, mut = mix.r4(c, alt), mix.r4(c, alt, include_year=True)
    y21 = c["ty"] == 2021
    assert (ok[y21] == 1.0).all() and (mut[y21] != ok[y21]).any()


def test_regime_rv_past_only_and_mutant_current_day_caught():
    diff = 0
    for s in range(20):
        x = np.exp(np.random.default_rng(s).normal(0, 1, 600))
        a, m = mix.regime_rv(x), mix.regime_rv(x, include_current=True)
        assert not a[:252].any()
        diff += int((a != m).sum())
    assert diff > 0


def test_f3_only_in_calm_and_stress_state_not_shifted():
    st = np.array([True, False, True, False])
    f4, top2, h = np.array([1., 2, 3, 4]), np.array([10., 20, 30, 40]), np.array([5., 6, 7, 8])
    assert (mix.switch(st, f4, top2) == [1, 20, 3, 40]).all()          # в стрессе — без F3
    assert (mix.switch(st, h, top2) == [5, 20, 7, 40]).all()
    assert (mix.switch(np.roll(st, 1), f4, top2) != mix.switch(st, f4, top2)).any()   # мутант d+1 отличим


def test_build_covers_all_rows_and_positive():
    c, b = ctx()
    c["st_g4"] = c["qw"] >= 4
    c["st_rv"] = c["qw"] >= 3
    bb = {"f4": b, "opt": b * 1.1, "top2": b * 0.9}
    for r in mix.C2_H3 + mix.C3:
        s = mix.build(r, c, bb)
        assert s.shape == (N,) and (s > 0).all() and np.isfinite(s).all(), r
    assert len(mix.C2_H3) == 8 and len(mix.C3) == 6
