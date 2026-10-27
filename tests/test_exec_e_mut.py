"""Семья E, §8 постановки: мутации 1, 6, 7, 8 (остальные — test_exec_e.py, test_metriki_e.py, test_likvidnost_e.py).
Таблица «мутация → тест» — в PR; ловит ли тест мутант, проверено подменой исходника (docs/kanon/v24/gates/E2_isp.md)."""
import numpy as np
import pytest

from fastbt.exec_e import Day, Params, q_mask, run, shift_krug


def D(rows, pusto=False):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], pusto)


def F(x, n=1):
    return [(x, x, x, x)] * n


def trades(days, sig, sg, **kw):
    r = run("T", days, sig, np.asarray(sg, float), Params(**kw))
    return [(t.entry_day, t.entry_px, t.exit_day, t.reason) for t in r.trades], r


# --- 1. σ̂ сдвинута на +1 день: заявка d+1 ставится по σ̂ дня d+1 (известной лишь завтра) ---
def _days_sigma():
    # d=0 сигнал; d+1: low 98.5 → лимит 99 (σ̂₀=0,02, k=0,5) исполняется, лимит 97 (σ̂₁=0,06) — нет
    return [D(F(100.0)), D(F(100.0) + [(100, 100, 98.5, 99.5)]), D(F(99.5))]


def test_m1_sigma_plus_one_day_changes_fills():
    good = [0.02, 0.06, 0.02]
    t_good, _ = trades(_days_sigma(), [1, 1, 0], good, k=0.5)
    assert t_good and t_good[0][:2] == (1, 99.0)
    leak = shift_krug(good, -1)                      # σ̂[d] := σ̂[d+1]: использован прогноз, известный завтра
    t_leak, _ = trades(_days_sigma(), [1, 1, 0], leak, k=0.5)
    assert t_leak != t_good and t_leak == []


def test_m1_future_sigma_does_not_change_past_trades():
    rng = np.random.default_rng(1)
    n = 40
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = [D([(c, c * 1.01, c * 0.99, c)] + F(c)) for c in cl]
    sig = (rng.random(n) < 0.5).astype(int)
    base = np.full(n, 0.02)
    pert = base.copy()
    pert[25:] = 0.07                                 # σ̂ с дня 25 другая: сделки до дня 25 обязаны совпасть
    a, _ = trades(days, sig, base, k=0.5, a=1.0, b=1.0)
    b, _ = trades(days, sig, pert, k=0.5, a=1.0, b=1.0)
    assert [t for t in a if t[2] is not None and t[2] <= 25] == [t for t in b if t[2] is not None and t[2] <= 25]


# --- 6. E0 при q = 0 ≠ A побайтно ---
def test_m6_e0_q0_is_a_bytewise():
    rng = np.random.default_rng(2)
    n = 60
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = [D([(c, c * 1.01, c * 0.99, c)] + F(c)) for c in cl]
    sig = (rng.random(n) < 0.5).astype(int)
    st = rng.random(n)
    st[::7] = np.nan                                 # q = 0 не зависит от силы, даже от NaN
    m = q_mask(st, sig, 0.0)
    assert m.tobytes() == sig.astype(int).tobytes() and m is not sig
    sg = np.full(n, 0.02)
    rb = run("T", days, m, sg, Params(), rf_cal=np.full(n, 1e-4))
    # эталон A — независимый путь msp.portfolio (не сам sig): позиции побайтно, pnl и ставка — до 1e-12
    from fastbt import msp
    import pandas as pd
    dates = pd.bdate_range("2021-03-01", periods=n)
    lr = np.r_[np.diff(np.log([d.c[-1] for d in days])), 0.0]
    rate = pd.Series([0.0365], pd.to_datetime(["2021-01-01"]))
    port = msp.portfolio(pd.DataFrame(dict(date=dates, ticker="T", strategy="S", position=sig * 1.0,
                                           daily_gross_return=sig * lr)), rate, Params().comm)
    assert rb.pos.tobytes() == port["exposure"].values.tobytes()
    assert np.allclose(rb.pnl, port["r"].values, atol=1e-12, rtol=0)


def test_m6_q_positive_filters_and_is_monotone():
    rng = np.random.default_rng(3)
    sig, st = np.ones(400, int), rng.random(400)
    k = [q_mask(st, sig, q).sum() for q in (0.0, 0.5, 0.67, 0.75)]
    assert k[0] == 400 and k[1] < k[0] and k[2] < k[1] and k[3] < k[2]
    assert 0.40 < k[1] / 400 < 0.60                  # q = 0,5: около половины дней сигнала (после прогрева)
    assert (q_mask(st, sig, 0.5)[:20] == 0).all()    # меньше min_obs — нет сделки


def test_m6_q_bad_inputs():
    with pytest.raises(ValueError):
        q_mask([1, 2], [1], 0.5)
    with pytest.raises(ValueError):
        q_mask([1.0], [1], 1.0)


# --- 7. сила сигнала использует данные дня d+1 ---
def test_m7_strength_future_and_current_day_do_not_leak():
    rng = np.random.default_rng(4)
    sig, st = np.ones(300, int), rng.random(300)
    m0 = q_mask(st, sig, 0.67)
    for d in (50, 120, 250):
        st2 = st.copy()
        st2[d + 1:] = rng.random(300 - d - 1) * 100  # будущее после d изменено
        assert (q_mask(st2, sig, 0.67)[:d + 1] == m0[:d + 1]).all()
    d = 100                                          # окно строго прошлое: сила дня d не входит в свою историю
    st3, sg3 = np.arange(300, dtype=float), np.ones(300, int)
    assert q_mask(st3, sg3, 0.99)[d] == 1            # d > квантиль 0,99 прошлого
    st3[d + 1] = -1.0
    assert q_mask(st3, sg3, 0.99)[d] == 1            # сила d+1 на решение d не влияет


# --- 8. к2 со сдвигом 0 ≠ исходный ряд ---
def test_m8_k2_lag0_is_original_and_lag_changes_result():
    rng = np.random.default_rng(5)
    s = rng.random(50) * 0.05 + 0.01
    assert shift_krug(s, 0).tobytes() == s.tobytes()
    assert np.array_equal(shift_krug(s, 7), np.roll(s, 7)) and not np.array_equal(shift_krug(s, 7), s)
    assert np.array_equal(np.sort(shift_krug(s, 7)), np.sort(s))     # распределение сохранено
    n = 50
    cl = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    days = [D([(c, c * 1.04, c * 0.93, c)] + F(c)) for c in cl]    # лимиты и уровни достигаются
    sig = np.ones(n, int)
    _, r0 = trades(days, sig, s, k=0.5, a=1.0, b=1.0)
    _, r1 = trades(days, sig, shift_krug(s, 0), k=0.5, a=1.0, b=1.0)
    _, r2 = trades(days, sig, shift_krug(s, 11), k=0.5, a=1.0, b=1.0)
    assert r0.pnl.tobytes() == r1.pnl.tobytes() and r0.pnl.tobytes() != r2.pnl.tobytes()
