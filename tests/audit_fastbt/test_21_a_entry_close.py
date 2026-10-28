"""21. Вход A: решение на закрытии d, вход по close[d] (спецификация п. 2), позиция d→d+1 (FA2b).

Ответ вручную: сигнал +1 на баре 5, сигнальный выход на баре 9, стопы недостижимы → сделка
[5, 9), pos[5..8] = +1, доход Σ pos·lr = log(close[9]/close[5]) = log(110/105); комиссия 2c.
Причинность движка: замена цен, объёма и сигналов после бара t не меняет pos[0..t] (позиция бара
t решена на его закрытии) — контртренд/диапазон (lvl_run, ATR и σ) и тренд (trend_run).
"""
import numpy as np
import pytest

from synth_fa2b import bars, perturb_after
from fastbt import engine, engine_trend, stats
from fastbt.indicators import compute_base

C = np.array([100, 101, 102, 103, 104, 105, 106, 104, 108, 110, 111, 112], float)


def _arrays(n):
    return (np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int8))


def _lvl(sig, exl, exs, close, high, low, atr0, sigma, kind=0, a1=50.0, a2=50.0, mh=100.0,
         gamma=-1.0, med=0.0, adx=None):
    n = len(close)
    te, tx, td = _arrays(n)
    adx = np.zeros(n) if adx is None else adx
    dummy = np.full(n, np.nan)
    nt = engine.lvl_run(sig, exl, exs, engine.next_signal(sig), close, high, low, adx, atr0,
                        sigma, dummy, dummy, dummy, dummy, kind, a1, a2, mh, -1.0, 0, 0, 0,
                        gamma, med, 0, n, te, tx, td)
    return te[:nt], tx[:nt], td[:nt]


def _trend(sig, exl, exs, close, high, low, atr, rh, rl, isl=50.0, tm=99.0):
    n = len(close)
    te, tx, td = _arrays(n)
    atr0 = np.where(np.isnan(atr), 0.0, atr)
    nt = engine_trend.trend_run(sig, exl, exs, engine.next_signal(sig), close, high, low, atr,
                                atr0, np.full(n, 0.02), rh, rl, 0, isl, tm, 1e12, 0, 0.0, 0.0, 0,
                                30.0, -1.0, 0.0, 0, n, te, tx, td)
    return te[:nt], tx[:nt], td[:nt]


def _hand_inputs():
    n = len(C)
    sig = np.zeros(n, np.int8)
    sig[5] = 1
    ex = np.zeros(n, np.bool_)
    ex[9] = True
    return sig, ex, n


def test_entry_by_hand_level():
    sig, ex, n = _hand_inputs()
    te, tx, td = _lvl(sig, ex, ex, C, C.copy(), C.copy(), np.ones(n), np.full(n, 0.02))
    assert (te.tolist(), tx.tolist(), td.tolist()) == ([5], [9], [1])
    pos = stats.pos_from_trades(te, tx, td, len(te), n)
    assert pos.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    lr = np.r_[np.log(C[1:] / C[:-1]), 0.0]
    assert np.isclose((pos * lr).sum(), np.log(110 / 105))
    c = 0.0005
    out = np.zeros((2, stats.NF))
    segb = np.array([0, n], np.int64)
    stats.seg_stats_trades(te, tx, td, len(te), lr, segb, np.zeros(n, np.int64), n, c, out)
    assert np.isclose(out[0, stats.F_S], np.log(110 / 105) - 2 * c)


def test_entry_by_hand_trend():
    sig, ex, n = _hand_inputs()
    te, tx, td = _trend(sig, ex, ex, C, C.copy(), C.copy(), np.ones(n), C.copy(), C.copy())
    pos = stats.pos_from_trades(te, tx, td, len(te), n)
    assert pos.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]


def _random_signals(n, seed):
    rng = np.random.RandomState(seed)
    sig = np.where(rng.rand(n) < 0.06, rng.choice([-1, 1], n), 0).astype(np.int8)
    sig[:20] = 0
    el = rng.rand(n) < 0.04
    es = rng.rand(n) < 0.04
    return sig, el, es


def _perturb_sig(sig, el, es, t, seed):
    s2, l2, e2 = _random_signals(len(sig), seed)
    return (np.r_[sig[:t + 1], s2[t + 1:]].astype(np.int8), np.r_[el[:t + 1], l2[t + 1:]],
            np.r_[es[:t + 1], e2[t + 1:]])


def _run_all(td, sig, el, es):
    ind = compute_base(td.close, td.high, td.low, td.open, td.volume)
    atr0 = np.where(np.isnan(ind["atr14"]), 0.0, ind["atr14"])
    n = td.n
    out = []
    for kind, a1, a2 in ((0, 1.5, 2.5), (2, 1.0, 2.0)):
        r = _lvl(sig, el, es, td.close, td.high, td.low, atr0, td.sigma["h1"], kind, a1, a2,
                 15.0, gamma=1.0 if kind == 2 else -1.0, med=0.017, adx=ind["adx14"])
        out.append(stats.pos_from_trades(*r, len(r[0]), n))
    r = _trend(sig, el, es, td.close, td.high, td.low, ind["atr14"], ind["rh_10"], ind["rl_10"],
               isl=2.0, tm=3.0)
    out.append(stats.pos_from_trades(*r, len(r[0]), n))
    return out


def test_positions_ignore_future():
    td = bars(n=420, seed=5)
    sig, el, es = _random_signals(td.n, 1)
    base = _run_all(td, sig, el, es)
    assert all((p != 0).sum() > 30 for p in base)
    for t in range(40, 415, 3):                      # шок 30% после t: стоп d+1 сработал бы
        alt = _run_all(perturb_after(td, t, seed=t, scale=0.3),
                       *_perturb_sig(sig, el, es, t, seed=t))
        for p0, p1 in zip(base, alt):
            assert np.array_equal(p0[:t + 1], p1[:t + 1]), t


# ---------- мутации ----------

_pft = stats.pos_from_trades
_lvl_run, _trend_run = engine.lvl_run, engine_trend.trend_run


def _enter_prev_close(te, tx, td, nt, n):
    """Позиция с бара e−1: вход по close[e−1], когда сигнал бара e ещё не известен."""
    return _pft(np.maximum(te - 1, 0), np.maximum(tx - 1, 0), td, nt, n)


def _nx(a):
    return np.r_[a[1:], a[-1]]


def _lvl_next(*a):
    a = list(a)
    a[5], a[6] = _nx(a[5]), _nx(a[6])          # high/low дня d+1 в проверке стопа и тейка
    return _lvl_run(*a)


def _trend_next(*a):
    a = list(a)
    a[5], a[6] = _nx(a[5]), _nx(a[6])
    return _trend_run(*a)


MUTATIONS = {
    "entry_prev_close": lambda mp: mp.setattr(stats, "pos_from_trades", _enter_prev_close),
    "stops_next_bar": lambda mp: (mp.setattr(engine, "lvl_run", _lvl_next),
                                  mp.setattr(engine_trend, "trend_run", _trend_next)),
}
