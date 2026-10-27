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
    pos[10] = 0.5          # бары 10 и 11 различаются: сдвиг бара выхода ловится
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


def _segs(n=20):
    segb = np.array([0, 6, 12, n], np.int64)
    seg_of = np.zeros(n, np.int64)
    for k in range(3):
        seg_of[segb[k]:segb[k + 1]] = k
    return segb, seg_of, np.diff(segb).astype(np.float64)


def test_a_window_stats_pay_exit_pos_and_trades():
    from fastbt.stats import NF, seg_stats_pos, seg_stats_trades, window
    lr, pos = _data()
    pos = np.zeros(20)
    pos[2:5] = 1.0
    pos[8:14] = -1.0       # открыта на конце окна (бар 11), держится дальше
    segb, seg_of, nb = _segs()
    want = _ref(pos, lr, 0, 12, C)
    a = np.zeros((3, NF))
    seg_stats_pos(pos, lr, segb, seg_of, 0, 20, C, a)
    assert abs(window(a, nb, 1, C)[0] - want) < 1e-9
    te, tx, td = np.array([2, 8]), np.array([5, 14]), np.array([1, -1], np.int8)
    b = np.zeros((3, NF))
    seg_stats_trades(te, tx, td, 2, lr, segb, seg_of, 20, C, b)
    assert abs(window(b, nb, 1, C)[0] - want) < 1e-9
    assert window(b, nb, 1, C)[0] < _ref(pos, lr, 0, 12, C, exit_end=False)


def test_a_window_flat_at_end_unchanged():
    from fastbt.stats import NF, seg_stats_pos, window
    lr, _ = _data()
    pos = np.zeros(20)
    pos[2:5] = 1.0
    segb, seg_of, nb = _segs()
    a = np.zeros((3, NF))
    seg_stats_pos(pos, lr, segb, seg_of, 0, 20, C, a)
    assert abs(window(a, nb, 1, C)[0] - _ref(pos, lr, 0, 12, C, False)) < 1e-9


def _stats(kind, pos, te, tx, td):
    from fastbt.stats import NF, seg_stats_pos, seg_stats_trades
    lr, _ = _data()
    segb, seg_of, nb = _segs()
    out = np.zeros((3, NF))
    if kind == "pos":
        seg_stats_pos(pos, lr, segb, seg_of, 0, 20, C, out)
    else:
        seg_stats_trades(te, tx, td, len(te), lr, segb, seg_of, 20, C, out)
    return out, nb


def test_trades_entry_and_exit_on_segment_last_bar():
    """Вход на последнем баре сегмента 1 (бар 11) платит выход при j=1; выход на последнем баре
    сегмента 0 (бар 5, позиция там 0) не платит ничего лишнего при j=0."""
    from fastbt.stats import window
    lr, _ = _data()
    pos = np.zeros(20)
    pos[2:5] = 1.0                 # сделка [2,5): позиция на баре 5 уже 0
    pos[11:15] = -1.0              # вход на баре 11 — последний бар сегмента 1
    te, tx, td = np.array([2, 11]), np.array([5, 15]), np.array([1, -1], np.int8)
    ot, nb = _stats("trades", pos, te, tx, td)
    op, _ = _stats("pos", pos, te, tx, td)
    assert np.allclose(ot, op, atol=1e-15)
    for j, end in ((0, 6), (1, 12)):
        assert abs(window(ot, nb, j, C)[0] - _ref(pos, lr, 0, end, C)) < 1e-9
    # выход ровно на последнем баре сегмента 0 (x = 5): сделка [2,5), сегмент 0 = [0,6) -> бар 5
    pos2 = np.zeros(20)
    pos2[2:6] = 1.0                # x = 6 — первый бар сегмента 1, позиция на баре 5 held
    ot2, _ = _stats("trades", pos2, np.array([2]), np.array([6]), np.array([1], np.int8))
    assert abs(window(ot2, nb, 0, C)[0] - _ref(pos2, lr, 0, 6, C)) < 1e-9
    pos3 = np.zeros(20)
    pos3[2:5] = 1.0                # выход x = 5 = последний бар сегмента 0
    ot3, _ = _stats("trades", pos3, np.array([2]), np.array([5]), np.array([1], np.int8))
    assert abs(window(ot3, nb, 0, C)[0] - _ref(pos3, lr, 0, 6, C)) < 1e-9


def test_a_window_drawdown_includes_exit_cost():
    """Растущий ряд, позиция держится до конца окна: просадка = только плата выхода."""
    from fastbt.stats import NF, seg_stats_pos, window
    n = 20
    lr = np.full(n, 0.01)
    pos = np.ones(n)
    segb, seg_of, nb = _segs(n)
    out = np.zeros((3, NF))
    seg_stats_pos(pos, lr, segb, seg_of, 0, n, C, out)
    mdd = window(out, nb, 1, C)[1]
    assert abs(mdd + C) < 1e-12
