"""G4 D: главный тест обрезки на новых модельных строках D — LGBM, rolling-окно, веса 2 года, ансамбли; мутант «признаки +1 день» (MD7, № 1, 15, 22)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5.b2 import engine, feat, synth_b2
from models_v2.s5.c import fitx
from models_v2.s5.d import engine_d, ens

D = 560
HP = fitx.HPX(2, 20.0, 5.0, 0.1, 0.7, 0.6, 150, 0.0, 0.0)
ADD = feat.GROUPS["F4"]
CASES = {"lgbm": dict(lib="lgbm"), "xgb_roll": dict(lib="xgb", win=200), "xgb_hl": dict(lib="xgb", hl=engine_d.HL504), "lgbm_roll": dict(lib="lgbm", win=200),
         "lgbm_s43": dict(lib="lgbm", seed=43)}


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def f_at(w, case):
    kw = dict(CASES[case])
    dz = engine.make_dz(w.dz0, w.feats, engine.Cfg("d_f4", add=ADD))
    m = engine_d.day_fit(w.F, dz, D, HP, kw.pop("seed", 42), **kw)
    return np.float64(m["predict"](dz.X[[D]])[0]).tobytes(), m["s2_es"]


@pytest.mark.parametrize("case", list(CASES))
def test_main_truncation_test_every_d_component(worlds, case):
    w, dmg = worlds
    assert f_at(w, case) == f_at(dmg, case)


def test_mutation_features_plus_one_day_caught(worlds):
    w, dmg = worlds
    for x in (w, dmg):
        x.feats = x.feats.shift(-1)
        x.dz0 = type(x.dz0)(np.vstack([x.dz0.X[1:], x.dz0.X[-1:]]), x.dz0.y, x.dz0.off)
    try:
        assert f_at(w, "lgbm") != f_at(dmg, "lgbm")
    finally:
        for x in (w, dmg):
            x.feats = x.feats.shift(1)


def test_ensemble_forecast_at_d_ignores_data_after_d():
    n, k = 700, 600
    rng = np.random.default_rng(9)
    dts = pd.bdate_range("2019-01-02", periods=n)
    td = dts[1:].append(pd.DatetimeIndex([dts[-1] + pd.Timedelta(days=1)]))
    h = np.exp(rng.normal(0, 0.4, n))
    base = dict(rv=h * np.exp(rng.normal(0, 0.6, n)), h=h, d=dts.to_numpy("datetime64[ns]"), td=td.to_numpy("datetime64[ns]"),
                st_g4=rng.random(n) > 0.7, st_rv=rng.random(n) > 0.6)
    M = {"X": h * np.exp(rng.normal(0, 0.3, n)), "L": h * np.exp(rng.normal(0, 0.3, n)), "Xr": h * 1.01, "Xh": h * 0.99}
    dam = {kk: v.copy() for kk, v in base.items()}
    M2 = {kk: v.copy() for kk, v in M.items()}
    dam["rv"][k:] *= 7
    for v in M2.values():
        v[k + 1:] *= 3          # прогнозы и RV, которые закрываются после d = k, портятся
    for row in ens.TRIALS:
        a, b = ens.build(row, base, M)[k], ens.build(row, dam, M2)[k]
        assert np.float64(a).tobytes() == np.float64(b).tobytes(), row
    mutant = ens.build("e4_roll", dam, M2, extra_closed=1)[k]   # MD1: пара d в окне — прогноз уже зависит от порчи
    assert np.float64(mutant).tobytes() != np.float64(ens.build("e4_roll", base, M, extra_closed=1)[k]).tobytes()
