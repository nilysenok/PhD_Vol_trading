"""Л1.4 (Ж-146): семьи S7–S9 с интерфейсом сигнала S1–S6 поверх fastbt/kross_v7. Тесты написаны ДО кода.
Мутация M-d1: скор панели берёт c[d+1] — свойство «сигнал дня d не видит d+1» обязано упасть."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kross_v7 as kv
from fastbt.signals import v7_kross as vk
from fastbt.signals.interface import WARMUP, SignalSet, TickerData

N, T = 8, 320
IDX = pd.bdate_range("2017-01-02", periods=T)
COLS = [f"T{i}" for i in range(N)]
METKI = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * T, dtype=object).reshape(T, N)


def _ceny(seed=1):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, N)), axis=0)), index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, T))), index=IDX)
    return c, m


def _ctx(c=None, m=None):
    if c is None:
        c, m = _ceny()
    return vk.Panel(IDX.to_numpy("datetime64[ns]"), COLS, c, m, METKI, np.ones((T, N)), np.ones((T, N), bool))


def _td(c, j, idx=IDX):
    x = c[COLS[j]].to_numpy()
    return TickerData(COLS[j], idx.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(len(x)), np.zeros(len(x)))


GRID = [{"rezhim": "den", "porog": 0.5, "bufer": 0.0, "cap": 3}]


def _sig(ctx, sid, j=0, grid=GRID, c=None):
    c = ctx.c if c is None else c
    return vk.semya(sid, ctx)(_td(c, j), {}, grid)


def test_g1_forma_i_progrev():
    ctx = _ctx()
    grid = GRID + [{"rezhim": "nedelya", "porog": 0.3, "bufer": 0.1, "cap": 2}]
    for sid in ("S7", "S8", "S9"):
        ss = _sig(ctx, sid, 3, grid)
        assert isinstance(ss, SignalSet) and ss.k == 2 and ss.params == grid
        assert ss.sig.shape == (2, T) and ss.sig.dtype == np.int8
        assert ss.exit_l.shape == ss.exit_s.shape == ss.strength.shape == (2, T)
        assert ss.exit_l.dtype == ss.exit_s.dtype == bool and ss.split is True
        assert (ss.sig[:, :WARMUP] == 0).all() and np.isfinite(ss.strength).all() and (ss.strength >= 0).all()
        assert set(np.unique(ss.sig)) <= {-1, 0, 1}


def test_g1_semya_sdelka_est():
    """Не пустой результат: хотя бы на одном тикере после прогрева есть сигналы (иначе тест формы пуст)."""
    ctx = _ctx()
    assert any(np.abs(_sig(ctx, "S7", j).sig).sum() > 0 for j in range(N))


def test_g2_sig_ravno_storona_paneli():
    ctx = _ctx()
    for sid, rez in (("S7", "den"), ("S8", "nedelya"), ("S9", "nedelya")):
        p = {"rezhim": rez, "porog": 0.4, "bufer": 0.1, "cap": 2}
        sk = kv.rebal(kv.panel(ctx.c, ctx.imoex, ctx.metki, sid), ctx.daty, rez)
        ozhid = kv.storona(sk, ctx.sigma, ctx.metki, ctx.ok, 0.4, 0.1, 2)
        for j in range(N):
            ss = _sig(ctx, sid, j, [p])
            e = ozhid[:, j].copy()
            e[:WARMUP] = 0
            assert np.array_equal(ss.sig[0], e)
            assert np.array_equal(ss.exit_l[0], ss.sig[0] != 1) and np.array_equal(ss.exit_s[0], ss.sig[0] != -1)
            sg = np.abs(np.nan_to_num(sk[:, j])) / ctx.sigma[:, j]
            assert np.allclose(ss.strength[0][WARMUP:], sg[WARMUP:])


def test_g2_podmnozhestvo_dat_td():
    """td короче панели (даты — подмножество): сигнал берётся на датах td."""
    ctx = _ctx()
    sel = np.arange(0, T, 2)
    idx = IDX[sel]
    td = _td(ctx.c.iloc[sel], 1, idx)
    ss = vk.semya("S8", ctx)(td, {}, GRID)
    full = _sig(ctx, "S8", 1).sig[0]
    e = full[sel].copy()
    e[:WARMUP] = 0
    assert np.array_equal(ss.sig[0], e)


def test_g3_tiker_vne_paneli_keyerror():
    ctx = _ctx()
    td = _td(ctx.c, 0)
    td.ticker = "ZZZZ"
    with pytest.raises(KeyError):
        vk.semya("S7", ctx)(td, {}, GRID)


def test_g3_data_vne_paneli_keyerror():
    ctx = _ctx()
    idx = pd.bdate_range("2030-01-01", periods=T)
    with pytest.raises(KeyError):
        vk.semya("S7", ctx)(_td(ctx.c, 0, idx), {}, GRID)


def test_reestr_i_sid():
    ctx = _ctx()
    assert set(vk.REESTR_V7) == {"S7", "S8", "S9"}
    for sid, f in vk.REESTR_V7.items():
        assert isinstance(f(ctx)(_td(ctx.c, 0), {}, GRID), SignalSet)
    with pytest.raises(KeyError):
        vk.semya("S1", ctx)
    with pytest.raises(ValueError):
        vk.semya("S7", ctx)(_td(ctx.c, 0), {}, [{"rezhim": "x", "porog": 0.5, "bufer": 0.0, "cap": 3}])


# ---- G4: M-d1 ----
def _bez_budushchego(sid, rez, d0=250):
    """Портим цены после d0 -> сигнал на днях ≤ d0 не должен меняться. True, если не меняется."""
    c, m = _ceny()
    c2 = c.copy()
    c2.iloc[d0 + 1:] *= np.random.default_rng(9).uniform(0.7, 1.4, (T - d0 - 1, N))
    m2 = m.copy()
    m2.iloc[d0 + 1:] *= 1.3
    p = [{"rezhim": rez, "porog": 0.3, "bufer": 0.1, "cap": 3}]
    for j in range(N):
        a, b = _sig(_ctx(c, m), sid, j, p, c).sig[0], _sig(_ctx(c2, m2), sid, j, p, c2).sig[0]
        if not np.array_equal(a[:d0 + 1], b[:d0 + 1]):
            return False
    return True


@pytest.mark.parametrize("sid,rez", [("S7", "den"), ("S8", "nedelya"), ("S9", "nedelya"), ("S9", "den")])
def test_g4_signal_ne_vidit_d_plus_1(sid, rez):
    assert _bez_budushchego(sid, rez)


@pytest.mark.parametrize("sid,sdvig", [("S7", 1), ("S8", 25), ("S9", 1)])
def test_g4_mutaciya_d_plus_1_lovitsya(sid, sdvig, monkeypatch):
    """S8 (12−1) по построению не зависит от последних 21 дня, поэтому утечку в нём видно только при сдвиге ≥ 22."""
    orig = kv.panel

    def pan_d1(c, imoex, metki, s):
        return np.vstack([orig(c, imoex, metki, s)[sdvig:], np.full((sdvig, c.shape[1]), np.nan)])   # скор дня d+sdvig в строке d

    monkeypatch.setattr(kv, "panel", pan_d1)
    assert not _bez_budushchego(sid, "den")
