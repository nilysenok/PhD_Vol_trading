"""S-A-IMPL-1 (тестировщик): синтетика с известным ответом, PIT (бары сессий > d и бары d после 18:40), mean_present b = 1, a на симплексе, слот по времени."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import daily as D
from models_v2.s17.fwd_src import Src
from models_v2.s18 import a_feat as A
import s18_a_feat_checks as C
from s18_a_guard import guard_real_files  # noqa: F401  (autouse)


@pytest.fixture(scope="module")
def F():
    return C.synth2()


def test_synthetic_s_har_equals_rv_windows_two_segments(F):
    assert F.seg.max() == 1 and (F.pos == 0).sum() == 2
    C.check_eq(F)


def test_pit_future_sessions_do_not_change_s(F):
    C.check_pit(F)


def test_mean_present_b_is_one_every_session(F):
    C.check_bmean(F)
    assert (F.n < 53).any()


def test_a_nonnegative_sum_one_and_har_at_omega_zero(F):
    C.check_simplex(F)


def test_slot_is_time_not_bar_number():
    G = A.build(C.bars_missing_first(), index_days=None)
    C.check_slots(G)


def test_spline_basis_centered_on_grid_and_known_shape():
    B = A.spline_basis(A.TAU)
    assert B.shape == (53, 4) and np.abs(B.mean(0)).max() < 1e-13
    assert np.abs(A.spline_basis(np.array([0.0, 1.0]))[:, 0] - (np.array([0, 1.0]) - A.TAU.mean())).max() < 1e-13   # N1 = τ
    assert A.PERM.tolist() == np.random.default_rng(0).permutation(53).tolist()


def test_real_bars_after_1840_of_day_d_and_sessions_after_d_do_not_change_s():
    """Реальные бары 2023: добавлены бары d после 18:40 и мусорные сессии > d — S дня d прежний."""
    src = Src()
    bars = src.bars("SBER", "2023-12-29")
    bars = bars[bars["begin"] >= "2022-09-01"].reset_index(drop=True)
    d = pd.Timestamp("2023-06-15")
    idx = src.index_days("2023-12-29")
    clean = bars[bars["begin"] <= d + pd.Timedelta(hours=18, minutes=40)]
    ex = bars[bars["begin"].dt.normalize() == d].tail(3).copy()
    ex["begin"] = ex["begin"] + pd.Timedelta(minutes=30)       # 19:00–19:20: вечерние, не основная сессия
    ex["close"] = ex["close"] * 1.2
    fut = bars[bars["begin"].dt.normalize() > d].copy()
    fut["close"] = fut["close"] * np.exp(np.random.default_rng(1).normal(0, 0.05, len(fut)))
    dirty = pd.concat([clean, ex, fut], ignore_index=True)
    F1, F2 = A.build(clean, idx), A.build(dirty, idx)
    i = F1.dates.get_loc(d)
    assert F2.dates[i] == d and len(F1.dates) == i + 1
    th = C.theta_rand(3)
    assert np.array_equal(C.ln_s(F1, th, [i]), C.ln_s(F2, th, [i]))
