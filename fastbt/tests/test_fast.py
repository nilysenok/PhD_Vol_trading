"""FS: быстрый слой (fast_boot, fast_sigma, fast_grid) — на синтетике, данные не читаются.

Каждая проверка ловит намеренно внесённую ошибку (вариант «с ошибкой» должен проваливаться).
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import fast_boot, fast_sigma, report, s9_stats
from fastbt.approach_cd import _eval, scaled_eval
from fastbt.fast_grid import grid_val_sharpe


def _manual_rank(s, W, minp):
    out = np.full(len(s), np.nan)
    for t in range(len(s)):
        if np.isnan(s[t]):
            continue
        w = s[max(0, t - W):t]                 # окно [t−W, t−1], без текущего дня
        w = w[~np.isnan(w)]
        if len(w) >= minp:
            out[t] = (w < s[t]).sum() / len(w)
    return out


def _rank_with_today(s, W, minp):              # подложенная ошибка: окно с текущим днём
    return pd.Series(s).rolling(W, min_periods=minp).rank(pct=True).values


def _check_rank(fn, s, W, minp):
    return np.array_equal(np.nan_to_num(fn(s, W, minp), nan=-1),
                          np.nan_to_num(_manual_rank(s, W, minp), nan=-1))


def test_rank_excludes_today_and_catches_planted():
    rng = np.random.default_rng(3)
    s = rng.lognormal(-4, 0.3, 700)
    s[[5, 100, 101, 400]] = np.nan
    for W in (126, 252, 504):
        mp = s9_stats.minp(W)
        assert _check_rank(s9_stats.past_rank, s.copy(), W, mp)
        R = np.empty((1, len(s)))
        fast_sigma.past_rank_rows(s[None, :].copy(), np.array([len(s)]), W, mp, R)
        assert np.array_equal(np.nan_to_num(R[0], nan=-1),
                              np.nan_to_num(s9_stats.past_rank(s.copy(), W, mp), nan=-1))
        assert not _check_rank(_rank_with_today, s.copy(), W, mp)      # ошибка поймана


def test_rank_ignores_today_value():
    """Значение дня t не входит в окна рангов дней t (только в ранги t+1…)."""
    rng = np.random.default_rng(4)
    s = rng.lognormal(-4, 0.3, 400)
    r0 = s9_stats.past_rank(s.copy(), 126, 50)
    s2 = s.copy()
    s2[300] *= 10                               # всплеск в день 300
    r1 = s9_stats.past_rank(s2, 126, 50)
    assert np.array_equal(r0[:300], r1[:300], equal_nan=True)
    assert r1[300] == 1.0                        # сравнивается с прошлым окном
    assert not np.array_equal(r0[301:427], r1[301:427])


def test_memo_rank_same_as_old():
    rng = np.random.default_rng(5)
    s = rng.lognormal(-4, 0.3, 600)
    fast_sigma.clear()
    for W in (126, 252):
        a = fast_sigma.past_rank(s, W, s9_stats.minp(W))
        b = fast_sigma.past_rank(s, W, s9_stats.minp(W))   # из памяти
        c = s9_stats.past_rank(s.copy(), W, s9_stats.minp(W))
        assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(c, nan=-1))
        assert np.array_equal(np.nan_to_num(b, nan=-1), np.nan_to_num(c, nan=-1))
    assert fast_sigma.STATS["hit"] >= 2


@pytest.mark.parametrize("n", [1, 7, 8, 9, 127, 128, 129, 255, 1000, 1007, 4096, 8192])
def test_pairwise_sum_bitwise(n):
    x = np.random.default_rng(n).standard_normal(n) * 1e-2 + 3e-4
    assert fast_boot.pw_sum(x) == np.sum(x)


def test_boot_iid_bitwise_and_planted():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(1003) * 0.01 + 2e-4
    b = rng.standard_normal(1003) * 0.01
    fast_boot.clear()
    for reps in (200, 1000):
        assert fast_boot.boot_delta(a, b, reps) == report.boot_delta(a, b, reps)
    # подложенная ошибка: другой seed повторов даёт другой p/ДИ
    assert fast_boot.boot_delta(a, b, 1000, seed=43) != report.boot_delta(a, b, 1000)
    idx = fast_boot.iid_index(len(a), 1000)
    assert np.array_equal(fast_boot.boot_sharpes(a, 1000), report._sh(a[idx]))


def test_lw_same_p():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(1000) * 0.01 + 3e-4
    b = 0.6 * a + rng.standard_normal(1000) * 0.008
    o = s9_stats.lw_sharpe_diff(a, b, 500, 5)
    n = fast_boot.lw_sharpe_diff(a, b, 500, 5)
    assert o[0] == n[0] and o[1] == n[1] and o[2] == n[2]
    wrong = fast_boot.lw_sharpe_diff(a, b, 500, 3)             # подложенная ошибка: блок 3
    assert wrong[2] != o[2] or wrong[1] != o[1]


def test_fast_engine_forbids_back_rate():
    """Ъ ~23:20: быстрый слой не принимает ставку «back» (дни (d_prev, d], печати v1)."""
    import inspect
    from fastbt import fast_grid
    src = inspect.getsource(fast_grid._engine_cls)
    assert 'attrs.get("cal") != "back"' in src and "forward=True" in src


def _kernel(apos, row, kind, lr, vs, ve, comm=0.0005):
    out = np.zeros((1, 1))
    grid_val_sharpe(apos[None, :].copy(), lr[None, :].copy(), np.zeros(1, np.int64),
                    np.zeros(1, np.int64), np.array([vs]), np.array([ve]),
                    row[None, None, None, :].astype(np.float64), 0 if kind == "mask" else 1,
                    comm, out)
    return out[0, 0]


def test_grid_kernel_bitwise_masked_and_scaled():
    rng = np.random.default_rng(8)
    n = 900
    lr = rng.standard_normal(n) * 0.015
    apos = np.sign(rng.standard_normal(n)) * (rng.random(n) < 0.6)
    for _ in range(20):
        vs, ve = int(rng.integers(50, 300)), int(rng.integers(500, n))
        m = (rng.random(n) < 0.7).astype(float)
        sh, _, _ = _eval(apos, np.ascontiguousarray(m[None, :] > 0.5), lr, vs, ve, 0.0005)
        assert _kernel(apos, m, "mask", lr, vs, ve) == sh[0]
        w = np.where(rng.random(n) < 0.5, 1.0, np.where(rng.random(n) < 0.5, 0.5, 0.0))
        ref = np.zeros(1)
        scaled_eval(apos, np.ascontiguousarray(w[None, :]), lr, vs, ve, 0.0005, ref)
        assert _kernel(apos, w, "scale", lr, vs, ve) == ref[0]
    # подложенная ошибка: комиссия без умножения на |Δпозиции| даёт другой Шарп
    assert _kernel(apos, m, "mask", lr, vs, ve, comm=0.001) != sh[0]
