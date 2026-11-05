"""GB2: RV_252 и нормировка масштаба Б-2/Б-4."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s3 import scale  # noqa: E402


def test_rv252_window_min_days_and_no_stitching_across_gap():
    rv, seg = np.arange(1.0, 701.0), np.r_[np.zeros(400, int), np.ones(300, int)]
    s = scale.rv252(rv, seg)
    assert np.isnan(s[:21]).all() and s[21] == pytest.approx(rv[:22].mean()) and s[100] == pytest.approx(rv[:101].mean())
    assert s[399] == pytest.approx(rv[148:400].mean())            # окно 252 строки
    assert np.isnan(s[400:421]).all() and s[421] == pytest.approx(rv[400:422].mean())  # новый сегмент, без склейки


def test_rv252_uses_only_rows_up_to_d():
    rv, seg = np.random.default_rng(0).uniform(1, 2, 500), np.zeros(500, int)
    a = scale.rv252(rv, seg)[300]
    rv2 = rv.copy()
    rv2[301:] *= 100
    assert scale.rv252(rv2, seg)[300].tobytes() == a.tobytes()
    assert scale.rv252(np.roll(rv, -1), seg)[300] != a  # мутация: окно сдвинуто на +1 день


def test_normalize_is_level_invariant_and_keeps_gap_dow():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 12))
    X[:, 7:9] = rng.uniform(0.1, 2, (50, 2))
    X[:, 3] = rng.uniform(0.1, 2, 50)
    S = rng.uniform(1, 3, 50)
    k = np.log(10.0)
    Y = X.copy()
    for c in scale.LN_COLS:
        Y[:, c] += k
    Y[:, 7:9] *= 10
    Y[:, 3] = np.log1p(np.expm1(X[:, 3]) * 10)
    Z, Z2 = scale.normalize(X, S), scale.normalize(Y, S * 10)
    assert np.allclose(Z, Z2, atol=1e-12) and (Z[:, 9:11] == X[:, 9:11]).all() and Z.shape[1] == 11
    assert not np.allclose(scale.normalize(X, S), scale.normalize(X, S * 10))  # мутация: масштаб не тот
