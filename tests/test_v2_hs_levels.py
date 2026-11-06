"""Ж-212 (HS11, G13): уровни 1–3 на синтетике с известным ответом; уровень 4 — «не проверялся»; мутации: N Холма = числу тестов, перепутанный знак Δ, лучшая модель = опора."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5.d import stats_d
from models_v2.s6.e import crit_e
from models_v2.s7 import hf_levels as V
from models_v2.s8 import hs_gstage as G, hs_levels as LV


def synth_df(n=2271, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2017-01-03", periods=n)
    truth = np.exp(rng.normal(0, 0.5, n))
    mk = lambda s: truth * np.exp(rng.normal(0, s, n))
    df = pd.DataFrame({"d": d, "target_date": d + pd.Timedelta(days=1), "rv": truth * rng.gamma(30.0, 1 / 30.0, n), "HAR": mk(0.6), "XJ": mk(0.4), "LF": mk(0.4)})
    return df, mk


def run(df, M, N=521):
    S = {"HAR-J": df["HAR"].to_numpy(float), "XGBoost": df["XJ"].to_numpy(float), "LightGBM": df["LF"].to_numpy(float), "M:best": M}
    t, lv = G.tests_and_levels(df, S, N)
    return t, lv.iloc[0]


def test_strictly_better_model_reaches_levels_1_2_3_and_level4_not_checked():
    df, mk = synth_df()
    t, r = run(df, mk(0.05))
    assert r["L1"] and r["L2"] and r["L3"] and r["verdict"] == "L3" and r["L4"] == "не проверялся"
    assert "потолок" in r["ceiling_note"]


def test_model_equal_to_reference_gets_no_level():
    df, _ = synth_df()
    t, r = run(df, df["XJ"].to_numpy(float) * np.exp(1e-9))         # практически XJ: лучше HAR, но не лучше S*
    assert r["L1"] and not r["L2"] and r["verdict"] == "L1"


def test_worse_model_gets_none_and_sign_mutation_is_caught():
    df, mk = synth_df()
    _, bad = run(df, mk(1.2))
    assert bad["verdict"] == "нет"
    _, good = run(df, mk(0.05))
    assert good["verdict"] == "L3" and bad["verdict"] != good["verdict"]


def test_holm_uses_full_N_not_number_of_tests():
    df, mk = synth_df()
    M = mk(0.3)
    t521, _ = run(df, M, 521)
    t8, _ = run(df, M, 8)
    a = t521[(t521["test"] == "A") & (t521["period"] == "all")]["p_holm"].iloc[0]
    b = t8[(t8["test"] == "A") & (t8["period"] == "all")]["p_holm"].iloc[0]
    assert a > b and a == min(1.0, a)                                   # N = 521 строже, чем N = числу тестов (мутация «N = число тестов» даёт b)


def test_top5_wo_removes_largest_contributors():
    dl = np.r_[np.full(95, 0.01), np.full(5, -10.0)]
    assert LV.top5_wo(dl) == 0.01 and dl.mean() < 0                      # преимущество целиком в 5 днях — без них Δ > 0
