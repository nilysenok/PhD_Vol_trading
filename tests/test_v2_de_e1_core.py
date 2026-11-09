"""E1 (X1): доверие по состоянию — инварианты и мутации (синтетика, известный ответ; в сеть и в данные проекта не ходит)."""
import torch  # noqa: F401  — первым (OpenMP)
import numpy as np
import pandas as pd

from models_v2.s11 import v4_ntp_state as V4
from models_v2.s15 import de0_groups as G, de_e1_state as S


def _days(n, start="2018-01-03"):
    return pd.bdate_range(start, periods=n)


def _series(n=420, seed=0, c_good_state=0):
    """Состояние 0: C точен (rv = C), состояние 1: M точен (rv = M). M и C разделены."""
    d = _days(n)
    s = (np.arange(n) % 2 == 1).astype(int)
    M, C = np.ones(n), np.full(n, 1.6)
    rv = np.where(s == c_good_state, C, M)
    tp = np.asarray(d[1:].append(pd.DatetimeIndex([d[-1] + pd.offsets.BDay(1)])))
    return d.to_numpy(), tp, rv, M, C, s


def test_pi_zero_gives_M_bitwise_and_F_positive():
    d, tp, rv, M, C, s = _series()
    assert np.array_equal(V4.combine(M, C, np.zeros(len(M))), M)
    assert np.array_equal(V4.combine(M, C, np.ones(len(M))), C)
    r = S.run(rv, M, C, s, d, tp)
    assert (r["F"] > 0).all() and np.isfinite(r["F"]).all() and ((r["pi"] > 0) & (r["pi"] < 1)).all()


def test_common_state_reduces_to_ntp2_known_answer():
    """Одно состояние на все дни и фиксированный α ⇒ π = V4 NTP2 (fixed_share, одна бумага, min_pairs = 0) — по формуле."""
    n = 300
    rng = np.random.default_rng(1)
    rv, M, C = np.exp(0.4 * rng.standard_normal(n)), np.exp(0.3 * rng.standard_normal(n)), np.exp(0.3 * rng.standard_normal(n))
    day, tp = np.arange(n), np.arange(n) + 1
    pis, _ = S._pass(V4.qlike(rv, C), V4.qlike(rv, M), np.zeros(n, int), day, tp, lambda i: (0.02, 0.02))
    ref, _ = V4.fixed_share(rv, M, C, np.zeros(n, int), day, tp, 1, np.array([True]), alpha=0.02, min_pairs=0)
    assert np.allclose(pis[:, 0], ref, rtol=0, atol=1e-12)
    assert (pis[:, 1] == 0.5).all()                           # процесс другого состояния без своих пар не движется


def test_states_have_separate_trust_and_beat_shared_trust():
    d, tp, rv, M, C, s = _series(600)
    sep, shared = S.run(rv, M, C, s, d, tp), S.run(rv, M, C, np.zeros(len(s), int), d, tp)
    late = slice(300, None)
    assert sep["pi"][late][s[late] == 0].mean() > 0.9 and sep["pi"][late][s[late] == 1].mean() < 0.1   # C верен в состоянии 0, M — в 1
    ql = lambda f: float(V4.qlike(rv[late], f[late]).mean())
    assert ql(sep["F"]) < 0.5 * ql(shared["F"])                  # общее π (≡ NTP) на тех же данных заметно хуже


def test_trust_uses_only_closed_pairs_real_path_and_mutation(monkeypatch):
    d, tp, rv, M, C, s = _series(300)
    base = S.run(rv, M, C, s, d, tp)
    short = S.run(rv[:200], M[:200], C[:200], s[:200], d[:200], tp[:200])
    assert np.array_equal(base["pi"][:200], short["pi"])          # прогноз дня i не зависит от данных после i
    rv2 = rv.copy()
    rv2[150] *= 7.0                                             # пара 150 (состояние 0) закрывается к дню 151; ближайший день состояния 0 — 152
    alt = S.run(rv2, M, C, s, d, tp)
    assert np.array_equal(alt["pi"][:152], base["pi"][:152]) and alt["pi"][152] != base["pi"][152]
    monkeypatch.setattr(S, "closed_date", lambda t_plus, date: date)      # мутация «закрыто + 1»: пара доступна в день своего прогноза
    mut = S.run(rv2, M, C, s, d, tp)
    assert mut["pi"][150] != alt["pi"][150]


def test_alpha_selected_by_past_only_and_2022_mutation_caught(monkeypatch):
    n = 300
    d = _days(n).to_numpy()
    avail = d + np.timedelta64(1, "D")
    s = np.zeros(n, int)
    early = np.arange(n) < 150
    loss = {0.05: np.where(early, 0.0, 1.0), 0.01: np.where(early, 1.0, 0.0), 0.02: np.full(n, 0.5)}
    assert S.select_alpha(loss, s, avail, d, 140, 0) == 0.05       # последние 126 закрытых пар к дню 140 — все «ранние»
    assert S.select_alpha(loss, s, avail, d, 299, 0) == 0.01
    assert S.select_alpha(loss, s, avail, d, 0, 0) == S.A_DEF      # закрытых пар нет — значение по умолчанию
    monkeypatch.setattr(S, "cutoff", lambda dates, i: dates[-1])  # мутация: выбор по всей выборке (в т.ч. по будущему)
    assert S.select_alpha(loss, s, avail, d, 140, 0) == 0.01


def test_alpha_grid_and_validation_window_are_declared_constants():
    assert S.GRID == (0.01, 0.02, 0.05) and S.VAL == 126 and S.Q_EXTREME == 0.99


def _store(n=700, bars=12, seed=3):
    rng = np.random.default_rng(seed)
    return dict(days=_days(n), r=1e-3 * rng.standard_normal((n, bars)))


def test_states_equal_de0_definition_and_ignore_future_bars(monkeypatch):
    st = _store()
    dates = st["days"][300:]
    s = S.states(st, dates)
    ref = G.extreme_flag(st, dates, G.day_stats(st, dates))
    assert np.array_equal(s[250:], ref[250:]) and (ref[250:] >= 0).all()
    k = 100
    st2 = dict(days=st["days"], r=st["r"].copy())
    pos = st["days"].get_loc(dates[k])
    st2["r"][pos + 1:] *= 1000.0                                   # будущие бары после дня k огромны
    assert np.array_equal(S.states(st2, dates)[: k + 1], s[: k + 1])
    st3 = dict(days=st["days"], r=st["r"].copy())
    st3["r"][-100:] *= 1000.0                                      # 14 % баров в конце — порог «по будущему» вырастает
    i = 5
    st3["r"][st["days"].get_loc(dates[i])] = 6e-3                  # день i: бар 6σ, по прошлому — экстрем
    assert S.states(st3, dates)[i] == 1
    monkeypatch.setattr(S, "past_end", lambda p: 10 ** 9)          # мутация: порог по барам ≥ d (будущие бары)
    assert S.states(st3, dates)[i] == 0
