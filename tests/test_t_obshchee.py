"""К-T0: общие части T16–T20 (z, Z, сила, вес, правило e ≥ конец + 2, TEvaluator). Синтетика; сеть и данные ≥ 2022 не нужны.

Мутации (каждая — функция-мутант, на которой инвариант обязан падать): σ̂_d вместо σ̂_{d−1} в z; вес «1 + (x − порог)»;
вход по close d (сдвиг sig на день влево, Д2); вес по σ̂_{d+1}; lag = 0.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import t_obshchee as T
from fastbt.etap1_data import _one_bar
from fastbt.t_eval import TEvaluator, TInput

S252 = np.sqrt(252.0)


# ---------- z, Z ----------
def test_z_known_answer():
    c = np.array([100.0, 100.0, 90.0, 99.0])
    s = np.array([0.02, 0.01, 0.05, 0.05])
    z = T.z_ryad(c, s)
    assert np.isnan(z[0]) and z[1] == 0.0
    assert z[2] == pytest.approx(np.log(0.9) / 0.01)              # знаменатель σ̂_{d−1}=0.01, не σ̂_d=0.05
    assert z[3] == pytest.approx(np.log(1.1) / 0.05)
    assert T.Z_ryad(c, s) == pytest.approx(z, nan_ok=True)


def test_z_bad_inputs_nan():
    z = T.z_ryad([100.0, 101.0, np.nan, 102.0, 103.0], [0.01, 0.0, 0.01, np.nan, 0.01])
    assert np.isfinite(z[1]) and np.isnan(z[[0, 2, 3, 4]]).all()   # NaN-close и NaN-σ̂_{d−1} — нет значения, не ноль
    z = T.z_ryad([100.0, 101.0, 102.0], [0.01, 0.0, 0.01])
    assert np.isfinite(z[1]) and np.isnan(z[2])                    # σ̂_{d−1} = 0 — нет значения
    assert len(T.z_ryad([], [])) == 0
    with pytest.raises(ValueError):
        T.z_ryad([1.0, 2.0], [0.1])


def _z_with_current_sigma(c, s):                                   # мутация: σ̂_d вместо σ̂_{d−1}
    return T.z_ryad(c, np.r_[s[1:], s[-1]])


def _z_depends_on_sigma_d(fn):
    c, s = np.array([100.0, 99.0, 90.0, 91.0, 95.0]), np.full(5, 0.02)
    s2 = s.copy()
    s2[2] = 0.5                                                    # возмущение σ̂ в день d=2 не должно менять z[2]
    return not np.allclose(fn(c, s)[2], fn(c, s2)[2])


def test_mutatsiya_z_sigma_tekushchego_dnya_lovitsya():
    assert not _z_depends_on_sigma_d(T.z_ryad)
    assert _z_depends_on_sigma_d(_z_with_current_sigma)


# ---------- сила, вес ----------
def test_sila_and_ves_known_answer():
    assert T.sila(-3.5) == 3.5
    w = T.ves(T.sila(-3.5), 3.0, 0.02)
    assert w == pytest.approx((3.5 / 3.0) * 0.20 / (0.02 * S252))
    assert w == pytest.approx(0.7349, abs=1e-4)
    assert T.ves(3.0, 3.0, 0.02) == pytest.approx(0.20 / (0.02 * S252))     # на пороге множитель 1
    assert T.ves(3.5, 3.0, 0.001) == 1.0                                    # лимит L
    assert T.ves(3.5, 3.0, 0.0) == 0.0 and T.ves(np.nan, 3.0, 0.02) == 0.0 and T.ves(3.5, 3.0, np.nan) == 0.0
    with pytest.raises(ValueError):
        T.ves(3.0, 0.0, 0.02)


def _ves_not_proportional(fn):
    a = fn(3.5, 3.0, 0.05)
    b = fn(7.0, 3.0, 0.05)
    return not (0 < a and b == pytest.approx(2 * a))               # оба ниже L: вес обязан быть пропорционален силе


def _ves_one_plus(sila_x, porog, sigma_d, L=1.0):                  # мутация контракта #376: 1 + (x − порог)
    return min(L, (1.0 + (sila_x - porog)) * 0.20 / (sigma_d * S252))


def test_mutatsiya_ves_odin_plus_otklonenie_lovitsya():
    assert not _ves_not_proportional(T.ves)
    assert _ves_not_proportional(_ves_one_plus)


# ---------- postroit_sig ----------
SG = np.full(20, 0.02)


def test_sig_e_ne_ranshe_d_plus_1_i_pravilo_plus_2():
    ev = [(1, 1, 3.5), (2, 1, 4.0), (4, 1, 3.5), (5, 1, 3.5), (9, -1, 3.2)]
    sig, w, acc = T.postroit_sig(20, ev, 2, SG, 3.0)
    # d=1 → e=2, конец 4; d=2 (e=3, во время сделки) и d=4 (e=5 = конец+1) отброшены; d=5 → e=6 = конец+2 принято
    assert [(d, e, s) for d, e, s, _ in acc] == [(1, 2, 1), (5, 6, 1), (9, 10, -1)]
    assert sig[2:5].tolist() == [1, 1, 1] and sig[5] == 0 and sig[6:9].tolist() == [1, 1, 1] and sig[10:13].tolist() == [-1] * 3
    assert all(sig[d] == 0 for d, *_ in acc)                       # в день сигнала позиции нет (Д2)
    assert np.flatnonzero(w).tolist() == [2, 6, 10]                # вес — на день входа e
    assert w[2] == pytest.approx((3.5 / 3.0) * 0.20 / (0.02 * S252))


def test_sig_lag_nol_zapreshchen():
    with pytest.raises(ValueError):
        T.postroit_sig(20, [(1, 1, 3.5)], 2, SG, 3.0, lag=0)


def test_sig_obrezka_i_plokhie_sobytiya():
    sig, w, acc = T.postroit_sig(6, [(5, 1, 3.5)], 3, SG[:6], 3.0)
    assert acc == [] and not sig.any()                             # e = 6 ≥ n — события нет
    sig, _, acc = T.postroit_sig(6, [(4, 1, 3.5)], 3, SG[:6], 3.0)
    assert len(acc) == 1 and sig.tolist() == [0, 0, 0, 0, 0, 1]    # хвост обрезан по n
    sg = SG[:6].copy()
    sg[1] = np.nan
    _, _, acc = T.postroit_sig(6, [(1, 1, 3.5), (2, 0, 3.5), (3, 1, np.nan)], 1, sg, 3.0)
    assert acc == []                                               # σ̂ NaN, сторона 0, сила NaN — пропуск, не вес 0 в сделке
    assert T.postroit_sig(6, [], 2, SG[:6], 3.0)[0].sum() == 0


def _w_sigma_next(n, ev, tau, sg, porog):                          # мутация: вес по σ̂_{d+1}
    return T.postroit_sig(n, ev, tau, np.r_[sg[1:], sg[-1]], porog)


def _w_uses_next_sigma(fn):
    sg = np.full(20, 0.02)
    sg2 = sg.copy()
    sg2[2] = 0.2                                                   # σ̂_{d+1} при d=1 — вес события d=1 меняться не должен
    return fn(20, [(1, 1, 3.5)], 2, sg, 3.0)[1][2] != fn(20, [(1, 1, 3.5)], 2, sg2, 3.0)[1][2]


def test_mutatsiya_ves_sigma_d_plus_1_lovitsya():
    assert not _w_uses_next_sigma(T.postroit_sig)
    assert _w_uses_next_sigma(_w_sigma_next)


# ---------- TEvaluator ----------
class FakeEng:
    def __init__(self, n):
        self.dates = pd.bdate_range("2018-01-01", periods=n)

    def port_rows(self, g, sides, expo, comm=None):
        return pd.DataFrame(dict(gross=g, excess=g - sides * comm, exposure=expo, turnover=sides), index=self.dates)


CL = np.array([100, 101, 103, 102, 106, 104, 105, 108, 107, 110, 109, 111], float)   # b = 11


def _ev(sigma=0.02, events=None):
    n = len(CL)
    x = TInput([_one_bar(c) for c in CL], CL, np.full(n, sigma), 0.01, 0, n - 1)
    return TEvaluator({"X": x}, FakeEng(n - 1), 0.0004)


def test_tevaluator_vhod_po_close_e_vyhod_po_vremeni_i_pnl():
    ev = _ev()
    ebt = {"X": [(1, 1, 3.5)]}
    df, rs, acc = ev.rows(ebt, 2, 3.0)
    (d, e, side, w), = acc["X"]
    assert (d, e) == (1, 2)
    tr, = rs[0].trades
    assert tr.entry_day == e > d and tr.entry_px == CL[e]          # Д2: вход по close e=d+1, не по close d
    assert tr.exit_day == e + 2 and tr.exit_px == CL[e + 2] and tr.reason == "vremya"
    assert df["gross"].sum() == pytest.approx(w * np.log(CL[e + 2] / CL[e]))
    assert df["turnover"].sum() == pytest.approx(2 * w)           # вход и выход
    assert df["excess"].sum() == pytest.approx(df["gross"].sum() - 2 * w * 0.0004)
    d_df = ev.rows(ebt, 2, 3.0, dnevnoy=True)[0]                  # дневной путь на тех же барах = тот же результат
    assert d_df["gross"].to_numpy() == pytest.approx(df["gross"].to_numpy(), abs=1e-12)


def test_tevaluator_pustoy_nabor_nol():
    r = _ev().evaluate({}, 2, 3.0)
    assert r["sdelok"] == 0 and r["pnl"] == 0.0 and not r["excess"].any() and r["prinyato"] == {"X": 0}


def test_tevaluator_shorti_znak():
    df, rs, _ = _ev().rows({"X": [(4, -1, 3.5)]}, 1, 3.0)
    tr, = rs[0].trades
    assert (tr.entry_day, tr.exit_day, tr.side) == (5, 6, -1) and CL[6] > CL[5]
    assert df["gross"].sum() < 0                                   # шорт, цена выросла — убыток


def _entry_at_d(sig):                                              # мутация Д2: сигнал исполняется в баре d
    return np.r_[sig[1:], 0]


def _enters_on_signal_bar(fn):
    sig, _, acc = fn()
    return any(sig[d] != 0 for d, *_ in acc)


def test_mutatsiya_ispolnenie_v_bare_signala_lovitsya():
    f = lambda: T.postroit_sig(20, [(1, 1, 3.5)], 2, SG, 3.0)      # noqa: E731
    assert not _enters_on_signal_bar(f)
    mut = lambda: (_entry_at_d(f()[0]), *f()[1:])                  # noqa: E731
    assert _enters_on_signal_bar(mut)
