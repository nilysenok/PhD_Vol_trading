"""V4-NTP (G29): состояние доверия — известные ответы на синтетике, тождества π ≡ 1 / π ≡ 0, fixed share (α = 0 застревает), шаг NTP4, мутация «закрыто + 1» в состоянии."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s11 import v4_ntp_state as SS

T = 400


def mk(nT=3, seed=0, c_better=True, n=T):
    """Дни подряд, t⁺ = следующий день. c_better: гибрид точен (C8 = RV), ML — с шумом; иначе наоборот."""
    rng = np.random.default_rng(seed)
    date = np.datetime64("2018-01-01") + np.repeat(np.arange(n), nT).astype("timedelta64[D]")
    tix = np.tile(np.arange(nT), n)
    rv = np.exp(rng.normal(0, 0.3, n * nT))
    bad = rv * np.exp(rng.normal(0, 0.5, n * nT))
    C, M = (rv, bad) if c_better else (bad, rv)
    return rv, M, C, tix, date, date + np.timedelta64(1, "D"), nT


def test_combine_endpoints_bitwise_and_middle():
    M, C = np.array([1.3, 2.1, 0.4]), np.array([1.1, 2.9, 0.5])
    assert np.array_equal(SS.combine(M, C, np.ones(3)), C) and np.array_equal(SS.combine(M, C, np.zeros(3)), M)
    assert np.allclose(SS.combine(M, C, np.full(3, 0.5)), np.sqrt(M * C))


def test_ewma_state_hand_computed_known_answer():
    """Одна бумага, hl = 1 (затухание 0,5), κ = 0, min_pairs = 1: m на строке r — EWMA D′ закрытых пар 0…r−1, s — с. о. вокруг этого среднего."""
    rng = np.random.default_rng(1)
    n = 6
    rv, M, C = np.exp(rng.normal(0, .3, n)), np.exp(rng.normal(0, .3, n)), np.exp(rng.normal(0, .3, n))
    date = np.datetime64("2018-01-01") + np.arange(n).astype("timedelta64[D]")
    m, s, _, cnt = SS.ewma_state(rv, M, C, np.zeros(n, int), date, date + np.timedelta64(1, "D"), 1, np.array([True]), hl=1.0, kappa=0.0, min_pairs=1)
    x = SS.qlike(rv, C) - SS.qlike(rv, M)
    assert np.isnan(m[0]) and cnt[0] == 0
    for r in range(1, n):
        w = 0.5 ** np.arange(r - 1, -1, -1)
        mean = (w * x[:r]).sum() / w.sum()
        assert m[r] == pytest.approx(mean, abs=1e-12)
        assert s[r] == pytest.approx(np.sqrt(max((w * x[:r] ** 2).sum() / w.sum() - mean ** 2, SS.EPS)), abs=1e-12)


@pytest.mark.parametrize("c_better", [True, False])
def test_pi1_known_direction_and_prior_before_60_pairs(c_better):
    rv, M, C, tix, date, tp, nT = mk(c_better=c_better)
    pm = np.ones(nT, bool)
    m, s, _, cnt = SS.ewma_state(rv, M, C, tix, date, tp, nT, pm)
    pi, z = SS.pi_discount(m, s)
    assert (pi[: 60 * nT] == 0.5).all() and np.isnan(m[: 60 * nT]).all()           # до 60 закрытых пар — приор
    late = pi[-nT * 100:]
    assert (late > 0.5).all() if c_better else (late < 0.5).all()                     # D′ < 0 → доверие гибриду; D′ > 0 → ML (σ(−m/s): масштаб ограничен с. о. D′)
    assert abs(late.mean() - 0.5) > 0.1
    assert (cnt[-nT:] == T - 1).all()


def test_fixed_share_returns_after_leader_change_and_alpha_zero_sticks():
    """Лидер меняется на шаге 300: с α = 0,02 доверие быстро возвращается к ML, без подмешивания (α = 0) — застревает у гибрида."""
    a = mk(c_better=True, seed=2, n=300)
    b = mk(c_better=False, seed=3, n=150)
    cat = lambda i: np.concatenate([a[i], b[i]]) if i in (0, 1, 2) else None
    rv, M, C = cat(0), cat(1), cat(2)
    nT = 3
    tix, date = np.tile(np.arange(nT), 450), np.datetime64("2018-01-01") + np.repeat(np.arange(450), nT).astype("timedelta64[D]")
    tp, pm = date + np.timedelta64(1, "D"), np.ones(nT, bool)
    pi1, _ = SS.fixed_share(rv, M, C, tix, date, tp, nT, pm)
    pi0, _ = SS.fixed_share(rv, M, C, tix, date, tp, nT, pm, alpha=0.0)
    mid, end = slice(299 * nT, 300 * nT), slice(-nT, None)
    assert (pi1[mid] > 0.9).all() and (pi0[mid] > 0.99).all()           # до смены лидер — гибрид
    assert (pi1[end] < 0.1).all()                                       # α = 0,02: вернулось
    assert (pi0[end] > 0.5).all()                                       # α = 0: накопленное преимущество не отыграно
    assert (pi1 >= 0.01 - 1e-12).all() and (pi1 <= 0.99 + 1e-12).all()  # подмешивание держит вес внутри [α/2, 1 − α/2]
    assert (pi1[: 60 * nT] == 0.5).all()


def test_fixed_share_eta_is_panel_median_of_closed_pairs():
    rv, M, C, tix, date, tp, nT = mk(n=80)
    _, eta = SS.fixed_share(rv, M, C, tix, date, tp, nT, np.ones(nT, bool))
    r = 70 * nT
    closed = tp <= date[r]
    pool = np.concatenate([SS.qlike(rv, C)[closed], SS.qlike(rv, M)[closed]])
    assert eta[r] == pytest.approx(1.0 / np.median(pool), rel=1e-12)


def test_smooth_step_bounded_and_mutation_larger_step_is_caught(monkeypatch):
    rng = np.random.default_rng(4)
    star, tix = rng.uniform(0, 1, 500), np.tile(np.arange(2), 250)
    out = SS.smooth(star, tix, 2)
    assert SS.max_step(out, tix, 2) <= SS.STEP + 1e-12 and np.isclose(out[0], 0.5 + np.clip(star[0] - 0.5, -0.02, 0.02))
    monkeypatch.setattr(SS, "STEP", 0.05)
    assert SS.max_step(SS.smooth(star, tix, 2), tix, 2) > 0.02 + 1e-9           # шаг больше 0,02 ловится проверкой


def test_ewma_state_closed_plus_one_mutation_changes_state(monkeypatch):
    """Подмена RV будущих пар (t⁺ > d) не меняет состояние на строке d; мутация closed_date = дата прогноза — меняет."""
    rv, M, C, tix, date, tp, nT = mk(n=150)
    day = date[100 * nT]
    rv2 = np.where(tp > day, rv * 7.3, rv)
    ok = lambda: np.array_equal(np.nan_to_num(SS.ewma_state(rv, M, C, tix, date, tp, nT, np.ones(nT, bool))[0][date <= day]),
                                np.nan_to_num(SS.ewma_state(rv2, M, C, tix, date, tp, nT, np.ones(nT, bool))[0][date <= day]))
    assert ok()
    monkeypatch.setattr(SS, "closed_date", lambda t, d: d)
    assert not ok()


def test_panel_excludes_masked_ticker():
    rv, M, C, tix, date, tp, nT = mk(n=120)
    pm = np.array([True, True, False])
    a = SS.ewma_state(rv, M, C, tix, date, tp, nT, pm)[2]
    rv2 = np.where(tix == 2, rv * 5.0, rv)
    b = SS.ewma_state(rv2, M, C, tix, date, tp, nT, pm)[2]
    assert np.array_equal(a, b)
    c = SS.ewma_state(rv2, M, C, tix, date, tp, nT, np.ones(nT, bool))[2]
    assert not np.array_equal(a, c)
