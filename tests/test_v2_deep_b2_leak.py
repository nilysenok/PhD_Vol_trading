"""G6 B2: главный тест обрезки на каждой новой модельной строке (прогноз на d побитно не меняется при порче данных после 18:50 d), LOYO без утечки."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import runner, select
from models_v2.s5.b2 import engine, feat, loyo, synth_b2

D = 560
GRP = {g.lower(): engine.Cfg("b2_" + g.lower(), add=c) for g, c in feat.GROUPS.items()}
CFGS = (list(GRP.values()) + [engine.Cfg("b2_" + k, drop=k) for k in engine.DROP] +
        [engine.Cfg("b2_top2", add=feat.GROUPS["F1"] + feat.GROUPS["F2"])] + engine.window_cfgs())


@pytest.fixture(scope="module")
def worlds():
    a, b, c = synth_b2.pair(700, 41)
    w = synth_b2.World(a, b, c)
    d = pd.Timestamp(w.F.dates[D])
    return w, synth_b2.World(synth_b2.damage(a, d), synth_b2.damage(b, d), synth_b2.damage(c, d))


def block_T(w, cfg) -> int:
    for T, ds in engine.blocks(w.F, np.arange(300, 699), cfg.freq).items():
        if D in ds:
            return T
    raise AssertionError


@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: c.name)
def test_main_truncation_test_every_row(worlds, cfg):
    w, dmg = worlds
    T = block_T(w, cfg)
    assert T <= D and (T < D or cfg.freq == "d")
    f1, f2 = w.forecast(cfg, T, D), dmg.forecast(cfg, T, D)
    assert f1.tobytes() == f2.tobytes()


def test_truncation_test_is_not_vacuous_mutation_features_plus_one_day(worlds):
    w, dmg = worlds
    cfg = GRP["f1"]
    for x in (w, dmg):
        x.feats = x.feats.shift(-1)          # мутация № 1: признаки строки d видят d+1
        x.dz0 = type(x.dz0)(np.vstack([x.dz0.X[1:], x.dz0.X[-1:]]), x.dz0.y, x.dz0.off)
    try:
        assert w.forecast(cfg, D, D).tobytes() != dmg.forecast(cfg, D, D).tobytes()
    finally:
        for x in (w, dmg):
            x.feats = x.feats.shift(1)


def test_loyo_fold_has_no_val_year_and_no_pairs_after_cut(worlds):
    w, _ = worlds
    Tc = select.cut_row(w.F, 2021)
    tr, val = runner.fold_split(w.F, w.dz0, Tc, 2020)
    yr = lambda ix: w.F.dates[ix + 1].astype("datetime64[Y]").astype(int) + 1970   # noqa: E731
    assert (yr(tr) != 2020).all() and (yr(val) == 2020).all() and tr.max() + 1 <= Tc and val.max() + 1 <= Tc
    mut_tr, mut_val = runner.fold_split(w.F, w.dz0, Tc + 1, 2020)                  # мутант № 15: окно заходит за 31.12 — ловится
    assert max(mut_tr.max(), mut_val.max()) + 1 > Tc


def test_loyo_fold_equals_s4_for_plain_config(worlds):
    w, _ = worlds
    Tc = select.cut_row(w.F, 2021)
    a = loyo.fold(w.F, w.dz0, engine.Cfg("x"), synth_b2.HP, Tc, 2020)
    b = runner.fold_score(w.F, w.dz0, synth_b2.HP, Tc, 2020, False)
    assert a == b and np.isfinite(a)
