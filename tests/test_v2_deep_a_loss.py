"""A13: QLIKE-оптимальный сдвиг масштаба (PIT) и описание функции потерь Б-2."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models_v2 import metrics
from models_v2.s5 import common, lossck, pit


def _syn(n=700, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2018-01-01", periods=n).to_numpy()
    td = np.roll(d, -1)
    td[-1] = d[-1] + np.timedelta64(1, "D")
    return np.exp(rng.normal(size=n) + 0.3), np.exp(rng.normal(size=n)), d, td


def test_zakrytaya_formula_minimiziruet_qlike():
    rv, s2, d, td = _syn()
    c = pit.scale_shift(rv, s2, d, td)
    k = 600
    m = int(np.searchsorted(td, d[k], side="right"))   # закрытые прогнозы на момент d_k
    num = minimize_scalar(lambda x: metrics.qlike(rv[:m], s2[:m] * np.exp(x)).mean(), bounds=(-3, 3), method="bounded", options={"xatol": 1e-10})
    assert abs(num.x - c[k]) < 1e-6


def test_do_252_zakrytykh_sdvig_nul():
    rv, s2, d, td = _syn()
    c = pit.scale_shift(rv, s2, d, td)
    m = np.searchsorted(td, d, side="right")
    assert (c[m < pit.MIN_CLOSED] == 0).all() and (c[m >= pit.MIN_CLOSED] != 0).any()


def test_mut_para_d_v_okne_lovitsya():
    rv, s2, d, td = _syn()
    assert np.abs(pit.scale_shift(rv, s2, d, td) - pit.scale_shift(rv, s2, d, td, extra_closed=1)).max() > 1e-6


def test_real_tablica_i_pometka():
    t = common.read_comment_csv(common.REP / "LOSS_CHECK.csv")
    assert set(t.variant) == {"raw", "scale_shift_pit"} and (t.note == lossck.NOTE).all()
    raw = t[(t.variant == "raw") & (t.slice == "all") & (t.model == "b2")].qlike.iloc[0]
    assert abs(raw - 0.22048650276427248) <= 1e-12
    assert "MSE" in lossck.TRAIN_LOSS["objective"] and "RV_m" in lossck.TRAIN_LOSS["target"]
