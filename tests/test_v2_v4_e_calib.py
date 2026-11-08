"""V4-E, G23: экспоненциальная калибровка (E1–E4): известный ответ, PIT (закрыто + 1), панель без исключённой бумаги, иерархия."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from models_v2.s11 import v4_e_calib as CAL

T, NT = 400, 3


def synth(c0=1.3, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    date = np.datetime64("2018-01-01") + np.repeat(np.arange(T), NT).astype("timedelta64[D]")
    tix = np.tile(np.arange(NT), T)
    F0 = np.exp(rng.normal(0, 0.3, T * NT))
    rv = c0 * F0 * np.exp(rng.normal(0, noise, T * NT)) if noise else c0 * F0
    return rv, F0, tix, date, date + np.timedelta64(1, "D")


def run(mode, rv, F0, tix, date, tp, hl=(20, 20), mask=None):
    return CAL.ewma_b(rv, F0, tix, date, tp, NT, mode, hl[0], hl[1], np.ones(NT, bool) if mask is None else mask)


@pytest.mark.parametrize("mode", ["own", "panel", "two"])
def test_known_answer_constant_bias_converges(mode):
    rv, F0, tix, date, tp = synth(1.3)
    b, _ = run(mode, rv, F0, tix, date, tp)
    assert np.allclose(b[-NT * 50:], np.log(1.3), atol=1e-12)
    assert (b[:NT * 60] == 0).all()            # до 60 закрытых пар b = 0


def test_clip_and_below_min_pairs():
    rv, F0, tix, date, tp = synth(50.0)
    b, _ = run("own", rv, F0, tix, date, tp)
    assert b.max() == CAL.B_CLIP and (b[:NT * 60] == 0).all()


@pytest.mark.parametrize("mode", ["own", "panel", "two"])
def test_pit_future_rv_does_not_change_past_and_mutation_closed_plus_one_caught(mode):
    rv, F0, tix, date, tp = synth(1.2, noise=0.3)
    r = 300 * NT                                     # строка дня 300: её пара закрывается на день 301
    rv2 = rv.copy()
    rv2[r:] *= 9.0
    b1, b2 = run(mode, rv, F0, tix, date, tp)[0], run(mode, rv2, F0, tix, date, tp)[0]
    assert np.array_equal(b1[:r + NT], b2[:r + NT])      # F дня 300 не зависит от RV пар t⁺ ≥ 301
    with mock.patch.object(CAL, "closed_date", lambda t, d: d):
        m1, m2 = run(mode, rv, F0, tix, date, tp)[0], run(mode, rv2, F0, tix, date, tp)[0]
    assert not np.array_equal(m1[:r + NT], m2[:r + NT])


def test_excluded_ticker_not_in_panel_and_mutation_caught():
    rv, F0, tix, date, tp = synth(1.2, noise=0.3)
    mask = np.array([True, True, False])
    rv2 = rv.copy()
    rv2[tix == 2] *= 4.0
    c1, c2 = run("panel", rv, F0, tix, date, tp, mask=mask)[1], run("panel", rv2, F0, tix, date, tp, mask=mask)[1]
    assert np.array_equal(c1, c2)
    d1, d2 = run("panel", rv, F0, tix, date, tp)[1], run("panel", rv2, F0, tix, date, tp)[1]
    assert not np.array_equal(d1, d2)                # мутация (исключённая осталась в панели) меняет ĉ


def test_hierarchy_shrinks_towards_panel():
    rv, F0, tix, date, tp = synth(1.0, noise=0.2)
    rv = rv.copy()
    rv[tix == 0] *= 1.5                              # бумага 0 смещена, панель в целом — нет
    b, _ = run("own", rv, F0, tix, date, tp, hl=(60, 60))
    raw, _ = run("own", rv, F0, tix, date, tp, hl=(60, 60))
    own0 = b[tix == 0][-1]
    assert 0 < own0 < np.log(1.5)                    # стянуто: между панельным (≈ среднее по 3) и собственным
    assert raw[tix == 1][-1] < own0
