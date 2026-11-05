"""A8: метрики на синтетике с известным ответом; согласованность с PREDICTIONS."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import common, stats


def test_idealnyj_prognoz():
    rv = np.exp(np.random.default_rng(0).normal(size=200))
    m = stats.full_metrics(rv, rv)
    assert abs(m["mz_alpha_ln"]) < 1e-12 and abs(m["mz_beta_ln"] - 1) < 1e-12 and abs(m["r2_mz_ln"] - 1) < 1e-12
    assert abs(m["cal_beta"] - 1) < 1e-12 and abs(m["bias_ln"]) < 1e-12 and m["qlike"] < 1e-12 and m["big_ln_share"] == 0


def test_konstanta_v_k_raz():
    rv = np.exp(np.random.default_rng(1).normal(size=200))
    m = stats.full_metrics(rv, 2.0 * rv)
    assert abs(m["bias_ln"] + np.log(2)) < 1e-12 and abs(m["mean_r"] - 0.5) < 1e-12 and abs(m["mz_alpha_ln"] - np.log(0.5)) < 1e-12
    assert abs(m["qlike"] - (0.5 + np.log(2) - 1)) < 1e-12 and m["big_ln_share"] == 0


def test_krupnye_oshibki_porogi():
    rv = np.ones(4)
    m = stats.full_metrics(rv, np.array([1.0, np.e ** 1.5, np.e ** -1.5, 1.0]))
    assert m["big_ln_share"] == 0.5


def test_win_share_i_mut_znak():
    a, b = np.array([1.0, 1, 1, 1]), np.array([2.0, 2, 2, 2])
    assert stats.win_share(a, b) == 1.0 and stats.win_share(b, a) == 0.0 and stats.win_share(a, a) == 0.5


def test_mut_win_share_bez_nichyikh_lovitsya():
    a = np.array([1.0, 1.0])
    strict = float((a < a).mean())
    assert strict != stats.win_share(a, a)


def test_pustoj_srez_oshibka():
    with pytest.raises(ValueError):
        stats.full_metrics([], [])
    with pytest.raises(ValueError):
        stats.win_share([], [])


def test_srez_ravno_srednemu_iz_predictions():
    df = pd.read_parquet(common.REP / "PREDICTIONS.parquet")
    d = stats.diagnostics(df)
    for lab, m in common.period_masks(df["target_date"]).items():
        r = d[(d.scope == "period") & (d.label == lab) & (d.model == "b2")].iloc[0]
        assert abs(r.qlike - df.loc[m, "qlike_b2"].mean()) <= 1e-12 and r.n == m.sum()
    assert d[(d.scope == "year")].groupby("model")["n"].sum().eq(len(df)).all()
