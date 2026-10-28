"""Ж-068: децили по сигналам голой идеи — синтетика с известным ответом + мутации (testy.md п. 1)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fastbt import diag_zh068 as dz

DATES = np.arange(np.datetime64("2019-12-20"), np.datetime64("2020-01-10"))      # 21 день; 2019-12-31 — индекс 11
CLOSE = 100.0 * np.exp(np.cumsum(np.r_[0.0, np.linspace(0.01, 0.21, 20)]))      # растущий ряд, разные шаги


def _setup():
    sig = np.zeros(len(DATES), int)
    sig[[3, 4, 10, 11]] = [1, -1, 1, -1]
    st = np.full(len(DATES), 0.5)
    return sig, st


def _patch_start(monkeypatch):
    monkeypatch.setattr(dz, "START", np.datetime64("2019-12-01"))


def _check_obs():
    sig, st = _setup()
    obs = dz.observations(sig, st, CLOSE, DATES, tau=2)
    assert [str(o[0]) for o in obs] == ["2019-12-23", "2019-12-24"]                 # d=10, 11: d+τ > 31.12 — отброшены
    assert obs[0][3] == pytest.approx(math.log(CLOSE[5] / CLOSE[3]))                 # лонг: (d, d+2]
    assert obs[1][3] == pytest.approx(-math.log(CLOSE[6] / CLOSE[4]))                # шорт: знак перевёрнут


def test_observations_known_answer(monkeypatch):
    _patch_start(monkeypatch)
    _check_obs()


def test_mutation_window_shifted_caught(monkeypatch):
    _patch_start(monkeypatch)
    monkeypatch.setattr(dz, "fwd_ret", lambda c, d, tau: float(np.log(c[d + tau - 1] / c[d - 1])))   # (d−1, d+τ−1]
    with pytest.raises(AssertionError):
        _check_obs()


def test_mutation_short_sign_not_flipped_caught(monkeypatch):
    _patch_start(monkeypatch)
    monkeypatch.setattr(dz, "signed", lambda r, side: r)
    with pytest.raises(AssertionError):
        _check_obs()


def test_mutation_tail_beyond_train_caught(monkeypatch):
    _patch_start(monkeypatch)
    monkeypatch.setattr(dz, "in_window", lambda d, tau, last: d + tau < len(DATES))     # хвост 2020 не отрезан
    with pytest.raises(AssertionError):
        _check_obs()


def test_nw_t_lag0_equals_plain_t_and_overlap_lowers_t():
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])
    assert dz.nw_t(x, 0) == pytest.approx(x.mean() / (x.std() / math.sqrt(len(x))))
    y = np.repeat([1.0, 2.0, 0.5, 1.5, 0.0, 2.5], 4) + 0.01 * np.arange(24)             # перекрытие: соседние значения почти равны
    assert abs(dz.nw_t(y, 3)) < abs(dz.nw_t(y, 0))
    assert math.isnan(dz.nw_t([1.0], 0)) and math.isnan(dz.nw_t([2.0, 2.0, 2.0], 1))


def _obs(sila, ret, day=0):
    return (np.datetime64("2018-01-02") + day, 1, sila, ret)


def test_deciles_saturated_row_and_counts():
    obs = [_obs(0.5 + 0.005 * i, 0.01 * (i % 3 - 1), i) for i in range(100)] + [_obs(1.0, 0.02, 100 + i) for i in range(7)]
    rows = dz.deciles(obs, tau=1)
    assert rows[-1][0] == "насыщ." and rows[-1][1] == 7 and rows[-1][4] == pytest.approx(2.0) and rows[-1][5] == 1.0 and rows[-1][7] == "шум"
    assert [r[1] for r in rows[:10]] == [10] * 10 and sum(r[1] for r in rows) == 107
    assert rows[0][2] == pytest.approx(0.5) and rows[9][3] == pytest.approx(0.995)


def test_deciles_empty_tape_no_crash():
    rows = dz.deciles([], tau=5)
    assert all(r[1] == 0 for r in rows) and len(rows) == 11


def test_neutral_ind_passes_everything_and_keeps_input():
    ind = dict(vf_pass=np.zeros(5, bool), adx14=np.zeros(5), vol_regime=np.full(5, 9.0), red_count5=np.zeros(5),
               green_count5=np.zeros(5), vol_sma5=np.ones(5), vol_sma20=np.ones(5))
    for sid in dz.NAMES:
        x = dz.neutral_ind(sid, ind)
        assert x["vf_pass"].all()
    assert (dz.neutral_ind("S3", ind)["adx14"] > 20).all() and (dz.neutral_ind("S6", ind)["adx14"] < 35).all()
    assert not ind["vf_pass"].any() and (ind["vol_regime"] == 9.0).all()                  # исходный ind не изменён
