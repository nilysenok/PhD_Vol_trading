"""A6: PREDICTIONS.parquet — единая таблица."""
import numpy as np
import pandas as pd

from models_v2 import har, metrics
from models_v2.s2 import features as ft
from models_v2.s3 import scale
from models_v2.s5 import common, frame
from models_v2 import prov

P = common.REP / "PREDICTIONS.parquet"
COLS = ["d", "target_date", "rv", "s2_b2", "s2_harj", "err_ln_b2", "err_ln_harj", "err_lvl_b2", "err_lvl_harj", "qlike_b2", "qlike_harj", "dloss",
        "sign_b2", "sign_harj", "rv_d", "rv_w", "rv_m", "tercile_w", "quintile_w", "stress_g4", "mar_jun_2020", "y2022"]


def test_shema_setka_i_poteri():
    df = pd.read_parquet(P)
    c = common.controls()
    assert set(COLS) <= set(df.columns) and len(df) == common.N_GRID and df["d"].is_monotonic_increasing and df["d"].is_unique
    assert (df["d"].to_numpy() == c["d"].to_numpy()).all() and (df["target_date"].to_numpy() == c["target_date"].to_numpy()).all()
    for m in ("b2", "harj"):
        assert np.abs(df["qlike_" + m] - metrics.qlike(df.rv, df["s2_" + m])).max() <= 1e-12
    assert abs(df["dloss"].mean() - (df.qlike_b2.mean() - df.qlike_harj.mean())) <= 1e-12
    assert len([x for x in df.columns if x.startswith("f_")]) == len(frame.FEAT) == 28


def test_rezhimy_pit_i_signal_znaka():
    df = pd.read_parquet(P)
    assert df["quintile_w"].between(0, 4).all() and df["tercile_w"].between(0, 2).all()
    assert (np.sign(df["err_ln_b2"]) == df["sign_b2"]).all()
    assert (df["err_lvl_b2"] == df["rv"] - df["s2_b2"]).all()
    assert (df["rv_d"] > 0).all() and (df["rv_m"] > 0).all()


def test_mut_priznaki_svoej_stroki_ravny_normirovannym_bazovym():
    df = pd.read_parquet(P)
    F = ft.build(pd.read_parquet(prov.BARS))
    pan = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    grid = har.eligible(pan)
    Z = scale.normalize(F.X, scale.scale(F, "m"))[grid]
    for j, n in enumerate(ft.NAMES[:ft.N_BASE]):
        assert np.array_equal(df["f_" + n].to_numpy(), Z[:, j], equal_nan=True), n
    shifted = np.roll(Z, -1, axis=0)   # мутация: признаки +1 день
    assert not np.array_equal(df["f_ln_rv"].to_numpy(), shifted[:, 0], equal_nan=True)
