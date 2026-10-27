"""ОВ (владелец 27.10): окно отбора платит выход позиции, открытой на его последнем баре (как Н5).
Синтетика; эталон — прямой подсчёт на Python. Мутация «нет выхода на конце» краснит тесты."""
import math
import numpy as np

from fastbt.approach_cd import masked_eval, scaled_eval
from fastbt.fast_grid import grid_val_sharpe
from fastbt.stats import sharpe_window

C = 0.0004
VS, VE = 3, 12


def _data():
    rng = np.random.default_rng(7)
    n = 16
    lr = rng.normal(0, 0.01, n)
    pos = np.zeros(n)
    pos[4:9] = 1.0
    pos[10:] = -1.0        # открыта на последнем баре окна (ve-1 = 11)
    return lr, pos


def _ref(pos, lr, vs, ve, comm, exit_end=True):
    prev, rs = 0.0, []
    for t in range(vs, ve):
        x = pos[t] * lr[t] - abs(pos[t] - prev) * comm
        if exit_end and t == ve - 1:
            x -= abs(pos[t]) * comm
        rs.append(x)
        prev = pos[t]
    r = np.array(rs)
    return r.mean() / r.std(ddof=1) * math.sqrt(252.0)


def test_sharpe_window_pays_exit():
    lr, pos = _data()
    assert abs(sharpe_window(pos, lr, VS, VE, C) - _ref(pos, lr, VS, VE, C)) < 1e-12
    assert sharpe_window(pos, lr, VS, VE, C) < _ref(pos, lr, VS, VE, C, exit_end=False)


def test_flat_at_end_pays_nothing_extra():
    lr, pos = _data()
    pos[10:] = 0.0
    assert abs(sharpe_window(pos, lr, VS, VE, C) - _ref(pos, lr, VS, VE, C, False)) < 1e-12


def test_masked_scaled_grid_equal_reference():
    lr, pos = _data()
    n = len(lr)
    masks = np.ones((2, n), np.bool_)
    masks[1, 6:9] = False
    sh, ex, tp = np.zeros(2), np.zeros(2), np.zeros(2)
    masked_eval(pos, masks, lr, VS, VE, C, sh, ex, tp)
    sc = np.ascontiguousarray(masks.astype(np.float64))
    sh2 = np.zeros(2)
    scaled_eval(pos, sc, lr, VS, VE, C, sh2)
    fam = np.ascontiguousarray(masks.astype(np.float64)[:, None, None, :])
    out = np.zeros((1, 2))
    grid_val_sharpe(pos[None, :].copy(), lr[None, :].copy(), np.array([0]), np.array([0]),
                    np.array([VS]), np.array([VE]), fam, 0, C, out)
    for m in range(2):
        want = _ref(pos * masks[m], lr, VS, VE, C)
        assert abs(sh[m] - want) < 1e-12
        assert abs(sh2[m] - want) < 1e-12
        assert abs(out[0, m] - want) < 1e-12
