"""Ж-051 итерация 3: распределитель топ-k (fastbt/e1_portfel.py). Синтетика; сеть и данные не нужны.

Каждый инвариант оформлен предикатом check_*(fn) -> bool; тест требует True на коде и False на мутации.
"""
import numpy as np
import pytest

from fastbt import e1_portfel as P


def _P(cols):
    """cols: список колонок-тикеров, каждая — строка из '+-.' по дням."""
    m = {"+": 1.0, "-": -1.0, ".": 0.0}
    return np.array([[m[c] for c in col] for col in cols]).T


def _rand(seed=3, T=120, n=8):
    rng = np.random.default_rng(seed)
    sig = np.zeros((T, n))
    for t in range(n):
        d = 0
        while d < T:
            d += int(rng.integers(1, 6))
            ln = int(rng.integers(1, 9))
            sig[d : d + ln, t] = rng.choice([-1.0, 1.0])
            d += ln
    return sig, rng.uniform(0.1, 3, (T, n)), rng.uniform(0.5, 2, (T, n)), rng.uniform(0.01, 0.05, (T, n))


# ---- предикаты ----
def check_golden(fn):
    sig = _rand()[0]
    return fn(sig, np.ones_like(sig), "net").tobytes() == sig.tobytes()


def check_slot_free_next_day(fn):
    sig = _P(["+++...", "..+++."])  # A занимает 0-2, B стартует в 2 — слот ещё занят
    sig2 = _P(["+++....", "...++.."])  # B стартует в 3 — слот свободен
    a, b = fn(sig, np.ones_like(sig), 1), fn(sig2, np.ones_like(sig2), 1)
    return a[:, 1].sum() == 0 and a[:, 0].sum() == 3 and b[:, 1].sum() == 2


def check_ranking_and_ties(fn):
    sig = _P(["+++", "+++", "+++"])
    ball = np.array([[1.0, 5.0, 5.0]] * 3)
    out = fn(sig, ball, 1)  # лучший — тикер 1 (ничья с 2 -> меньший индекс)
    return out[:, 1].sum() == 3 and out[:, 0].sum() == 0 and out[:, 2].sum() == 0


def check_no_eviction(fn):
    sig = _P(["+++...", ".++++."])
    ball = np.array([[1.0, 9.0]] * 6)
    out = fn(sig, ball, 1)
    return out[:, 0].sum() == 3 and out[:, 1].sum() == 0


def check_k_each_day(fn):
    sig, ball, _, _ = _rand()
    for k in (1, 2, 3, 5):
        out = fn(sig, ball, k)
        if ((out != 0).sum(axis=1) > k).any():
            return False
        # сделка берётся целиком или не берётся
        if ((out != 0) & (out != sig)).any():
            return False
    return True


def _pipe_ok(fn_ball):
    """Решения на дни <= d не зависят от strength/σ на днях > d."""
    sig, _, st, sg = _rand(5)
    base = P.raspredelit(sig, fn_ball(st, sg), 2)
    rng = np.random.default_rng(1)
    for d in range(5, 110, 7):
        st2, sg2 = st.copy(), sg.copy()
        st2[d + 1 :] = rng.uniform(0.1, 3, st2[d + 1 :].shape)
        sg2[d + 1 :] = rng.uniform(0.01, 0.05, sg2[d + 1 :].shape)
        alt = P.raspredelit(sig, fn_ball(st2, sg2), 2)
        # старты дня <= d: в позициях днях <= d различий быть не должно
        if not np.array_equal(base[: d + 1], alt[: d + 1]):
            return False
    return True


def check_sektor(fn):
    sig = _P(["+++", "+++", "+++"])
    out = fn(sig, np.array([[3.0, 2.0, 1.0]] * 3), 3, np.array([0, 0, 1]), 1)
    return out[:, 0].sum() == 3 and out[:, 1].sum() == 0 and out[:, 2].sum() == 3


# ---- тесты на коде ----
def test_golden_net_pobaytno():
    assert check_golden(P.raspredelit)
    sig = _rand()[0]
    assert P.raspredelit(sig, np.zeros_like(sig), None).tobytes() == sig.tobytes()
    assert P.raspredelit(sig, sig, "net") is not sig


def test_slot_svoboden_so_sleduyushchego_dnya():
    assert check_slot_free_next_day(P.raspredelit)


def test_rang_i_nichya():
    assert check_ranking_and_ties(P.raspredelit)


def test_bez_vytesneniya():
    assert check_no_eviction(P.raspredelit)


def test_k_sobljudaetsya_kazhdyj_den():
    assert check_k_each_day(P.raspredelit)


def test_vyrozhdennaya_tochka_ravna_roditeyu():
    sig = _rand()[0]
    mx = int((sig != 0).sum(axis=1).max())
    assert np.array_equal(P.raspredelit(sig, _rand()[1], mx), sig)


def test_bally_ne_smotryat_vpered():
    assert _pipe_ok(P.ball_sily)


def test_ball_formula_i_nulevaya_sigma():
    b = P.ball_sily(np.array([[-2.0, 3.0, 1.0]]), np.array([[4.0, 0.0, np.nan]]))
    assert b[0, 0] == 0.5 and np.isnan(b[0, 1]) and np.isnan(b[0, 2])


def test_nan_ball_poslednij_no_dopuskaetsya():
    sig = _P(["++", "++"])
    ball = np.array([[np.nan, 1.0]] * 2)
    assert P.raspredelit(sig, ball, 1)[:, 1].sum() == 2
    assert P.raspredelit(sig, ball, 2).tobytes() == sig.tobytes()


def test_sektor():
    assert check_sektor(P.raspredelit)


def test_pustaya_lenta_i_oshibki():
    z = np.zeros((10, 3))
    assert not P.raspredelit(z, z, 2).any()
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            P.raspredelit(z, z, bad)
    with pytest.raises(ValueError):
        P.raspredelit(z, np.zeros((9, 3)), 2)
    with pytest.raises(ValueError):
        P.raspredelit(z, z, 2, sektor=np.zeros(3))


def test_perevorot_znaka_novaya_sdelka():
    sig = _P(["++--", "...."])
    assert P.raspredelit(sig, np.ones_like(sig), 1).tobytes() == sig.tobytes()


# ---- мутации ----
def test_mut_slot_osvobozhdaetsya_v_tot_zhe_den(monkeypatch):
    orig = P._run_end
    monkeypatch.setattr(P, "_run_end", lambda s: orig(s) - 1)
    assert not check_slot_free_next_day(P.raspredelit)


def test_mut_ball_po_sigma_d_plus_1():
    mut = lambda st, sg: P.ball_sily(st, np.roll(sg, -1, axis=0))  # noqa: E731
    assert not _pipe_ok(mut)


def test_mut_ball_po_sile_d_plus_1():
    mut = lambda st, sg: P.ball_sily(np.roll(st, -1, axis=0), sg)  # noqa: E731
    assert not _pipe_ok(mut)


def test_mut_k_plus_odin():
    assert not check_k_each_day(lambda s, b, k, *a: P.raspredelit(s, b, k + 1, *a))


def test_mut_ranzhirovanie_naoborot():
    assert not check_ranking_and_ties(lambda s, b, k: P.raspredelit(s, -b, k))


def test_mut_vytesnenie():
    def evict(sig, ball, k):  # более сильный сигнал забирает слот у открытой позиции
        out = P.raspredelit(sig, ball, k).copy()
        out[:, 0] = 0
        out[1:, 1] = sig[1:, 1]
        return out

    assert not check_no_eviction(evict)


def test_mut_golden_net_ne_pobaytno():
    assert not check_golden(lambda s, b, k: s + 1e-12 * (s != 0))


def test_mut_sektor_ignoriruetsya():
    assert not check_sektor(lambda s, b, k, se, n: P.raspredelit(s, b, k))
