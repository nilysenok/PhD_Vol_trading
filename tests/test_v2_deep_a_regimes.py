"""A11: режимы только по прошлому."""
import numpy as np
import pandas as pd

from models_v2.s5 import common, regimes


def test_perekhody_i_napravlenie():
    q = pd.Series([0.0, 0.0, 2.0, np.nan, 1.0])
    assert list(regimes.transitions(q).dropna()) == ["q0>q0", "q0>q2"]
    assert list(regimes.direction(q).dropna()) == ["same", "up"]


def test_real_summa_n_po_kvintilyam():
    df = pd.read_parquet(common.REP / "PREDICTIONS.parquet")
    t = common.read_comment_csv(common.REP / "REGIME_DIAGNOSTICS.csv")
    q = t[(t.kind == "quintile_w") & (t.model == "b2")]["n"].sum()
    assert q == df["quintile_w"].notna().sum() == common.N_GRID
    st = t[(t.kind == "stress_g4") & (t.model == "b2")]["n"].sum()
    assert st == common.N_GRID
    assert {"mar_jun_2020", "y2022"} <= set(t[t.kind == "window"].label)
    tr = t[(t.kind == "transition_dir") & (t.model == "b2")]["n"].sum()
    assert tr == df["quintile_w"].notna().sum() - 1


def test_mut_kvintil_po_tekushchemu_dnyu_lovitsya():
    from models_v2.s5 import pit
    x = np.exp(np.random.default_rng(0).normal(size=600))
    ok = pit.past_bucket(x, [.2, .4, .6, .8])
    bad = pit.past_bucket(x, [.2, .4, .6, .8], include_current=True)
    assert not np.array_equal(ok, bad, equal_nan=True)
    assert (ok[252:] <= 4).all() and np.isnan(ok[:252]).all()
