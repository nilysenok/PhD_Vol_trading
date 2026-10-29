"""R-ж: n_k2_konechnyh — число конечных повторов k2 (то n, что в p_k2); n_k2 — всего."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import r1_zh  # noqa: E402

nan = float("nan")
HYB = r1_zh.NAMES["hyb"][1]


def _df():
    return pd.DataFrame(dict(tip=["baza", HYB, HYB, "kontrol"], msr=[1.0, 2.0, 3.0, 4.0]))


def test_kolonki_est_i_n_sovpadaet_s_konechnymi():
    v = [1.0, nan, 2.0, nan, 20.0]
    d = r1_zh.dopolnit_n(_df(), v)
    assert (d.loc[d.tip == HYB, "n_k2_konechnyh"] == 3).all()
    assert (d.loc[d.tip == HYB, "n_k2"] == 5).all()
    assert d.loc[d.tip != HYB, ["n_k2", "n_k2_konechnyh"]].isna().all().all()
    # n в p_k2 — то же число: p = (1 + #≥)/(n+1) при n = 3
    assert r1_zh.p_k2_znachenie(10.0, v) == 2 / (3 + 1)


def test_mutaciya_n_ravno_vsego_lovitsya():
    v = [1.0, nan, 2.0, nan, 20.0]
    d = r1_zh.dopolnit_n(_df(), v)
    mut = len(v)                                  # мутация: n = всего
    assert d.loc[d.tip == HYB, "n_k2_konechnyh"].iloc[0] != mut


def test_vse_konechnye_n_ravno_vsego_i_nanmean():
    d = r1_zh.dopolnit_n(_df(), [1.0, 2.0])
    assert (d.loc[d.tip == HYB, "n_k2_konechnyh"] == 2).all()
    assert np.nanmean([1.0, nan, 3.0]) == 2.0
