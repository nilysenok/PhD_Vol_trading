"""A9: накопленная разность потерь Б-2 − HAR-J(б)."""
import numpy as np
import pandas as pd

from models_v2.s5 import common, cum


def _df(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"target_date": pd.date_range("2020-01-01", periods=n), "qlike_b2": rng.random(n), "qlike_harj": rng.random(n)})


def test_posledneye_znachenie_ravno_n_na_raznost_srednikh():
    d = _df()
    t = cum.table(d)
    assert abs(t["cum_b2_minus_harj"].iloc[-1] - len(d) * (d.qlike_b2.mean() - d.qlike_harj.mean())) <= 1e-9


def test_mut_znak_pereputan_lovitsya():
    d = _df()
    assert abs(cum.table(d, sign=-1)["cum_b2_minus_harj"].iloc[-1] - cum.table(d)["cum_b2_minus_harj"].iloc[-1]) > 1e-3


def test_real_tablica_i_png():
    p = pd.read_parquet(common.REP / "PREDICTIONS.parquet")
    t = common.read_comment_csv(common.REP / "CUMULATIVE_LOSS_DIFF.csv")
    assert abs(t["cum_b2_minus_harj"].iloc[-1] - len(p) * (p.qlike_b2.mean() - p.qlike_harj.mean())) <= 1e-9
    assert (common.REP / "cum_loss_diff.png").stat().st_size > 5000
