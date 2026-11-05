"""GB2/GB3 (закрытие «частично»): форма учится при смене уровня (Б-2), монотонность по нормированным ln RV (Б-4),
набор признаков Б-5 — ровно 5 столбцов."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v2_hyb_leak import synth  # noqa: E402

from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import pb  # noqa: E402


def test_b5_design_has_exactly_five_columns_from_core_and_har():
    bars, _ = synth(n_days=400)
    F = ft.build(bars)
    D, off = pb.build_designs(F)
    d = D["b5"]
    assert d.X.shape[1] == 5 and d.cols == (0, 1, 2, 3, 4) and d.need == (0, 1, 2, 3, 4)
    assert np.array_equal(d.X, F.X[:, [0, 1, 2, 3, ft.I_HAR]], equal_nan=True)
    assert (off["b5"] == 0).all() and D["b1"].X.shape[1] == ft.N_BASE and D["s252"].X.shape[1] == ft.N_BASE


def level_shift_data(n=900, shift_at=700, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 11))
    ln_s = np.where(np.arange(n) < shift_at, 0.0, np.log(10.0))
    form = 0.6 * x[:, 0] + 0.3 * x[:, 1]
    return x, ln_s, form + rng.normal(0, 0.2, n)


@pytest.mark.parametrize("lib", ["xgb", "lgbm"])
def test_scaled_design_follows_level_shift_but_level_design_does_not(lib):
    x, ln_s, y_norm = level_shift_data()
    spec, tr, es, te = boost.Spec(lib, "mse", False, False), slice(0, 450), slice(450, 650), slice(750, 900)
    pred_n = boost.fit(spec, 3, 0.5, x[tr], y_norm[tr], x[es], y_norm[es])[0]
    err_norm = np.abs(pred_n(x[te]) + ln_s[te] - (y_norm[te] + ln_s[te])).mean()  # ln σ̂² = ln S + f(x_norm)
    xl = x.copy()
    xl[:, 0] += ln_s                                  # уровень входит в признак; форма та же
    y_lvl = y_norm + ln_s
    pred_l = boost.fit(spec, 3, 0.5, xl[tr], y_lvl[tr], xl[es], y_lvl[es])[0]
    err_lvl = np.abs(pred_l(xl[te]) - y_lvl[te]).mean()
    assert err_norm < 0.35 and err_lvl > 1.0 and err_lvl > 3 * err_norm


@pytest.mark.parametrize("lib", ["xgb", "lgbm"])
def test_monotone_plus_on_normalized_ln_rv_columns_and_free_fit_is_not_monotone(lib):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (700, 11))
    y = -0.9 * x[:, 0] + 0.5 * x[:, 3] + rng.normal(0, 0.2, 700)  # по колонке 0 истинная связь убывающая
    tr, es = slice(0, 450), slice(450, 700)
    grid = np.tile(np.median(x, axis=0), (60, 1))
    grid[:, 0] = np.linspace(-2.5, 2.5, 60)
    fm = boost.fit(boost.Spec(lib, "mse", False, True), 3, 0.5, x[tr], y[tr], x[es], y[es])[0](grid)
    ff = boost.fit(boost.Spec(lib, "mse", False, False), 3, 0.5, x[tr], y[tr], x[es], y[es])[0](grid)
    assert (np.diff(fm) >= -1e-12).all()                 # монотонность (+) по колонке 0 (нормированный ln RV_d)
    assert (np.diff(ff) < -1e-6).any()                   # без ограничения модель идёт против знака — мутация «монотонность снята» видна
