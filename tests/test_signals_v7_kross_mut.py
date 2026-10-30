"""Тестировщик #849: мутации на уровне обёртки Л1.4 (интерфейс SignalSet), которых нет в test_signals_v7_kross.py.
Мутации: S8 «окно включает день d (и d+1)» через обёртку; недельная ребалансировка на стыке ISO-годов (31.12.2018 — W01 2019);
exit_l/exit_s = «сторона снята». Только синтетика."""
import numpy as np
import pandas as pd
import pytest

from fastbt import kross_seksiya as ks
from fastbt import kross_v7 as kv
from fastbt.signals import v7_kross as vk
from fastbt.signals.interface import WARMUP, TickerData

N, T = 8, 320
IDX = pd.bdate_range("2018-01-01", periods=T)          # 31.12.2018 — понедельник W01 2019, индекс ≈ 260
COLS = [f"T{i}" for i in range(N)]
METKI = np.array(["A", "A", "A", "B", "B", "B", "C", ""] * T, dtype=object).reshape(T, N)


def _ceny(seed=3):
    r = np.random.default_rng(seed)
    c = pd.DataFrame(100 * np.exp(np.cumsum(r.normal(0, 0.01, (T, N)), axis=0)), index=IDX, columns=COLS)
    return c, pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, T))), index=IDX)


def _ctx(c, m):
    return vk.Panel(IDX.to_numpy("datetime64[ns]"), COLS, c, m, METKI, np.ones((T, N)), np.ones((T, N), bool))


def _sig(c, m, j, p, sid="S8"):
    x = c[COLS[j]].to_numpy()
    td = TickerData(COLS[j], IDX.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(T), np.zeros(T))
    return vk.semya(sid, _ctx(c, m))(td, {}, [p])


P_DEN = {"rezhim": "den", "porog": 0.3, "bufer": 0.0, "cap": 3}


def _okno_ne_vidit_d(d=290):
    """Портим закрытия d-20..d+2: sig на днях ≤ d обязан остаться. Возвращает (не изменился, изменился после d)."""
    c, m = _ceny()
    c2 = c.copy()
    c2.iloc[d - 20:d + 3] *= np.linspace(0.3, 3.7, N)[None, :] ** 2
    same, after = True, False
    for j in range(N):
        a, b = _sig(c, m, j, P_DEN).sig[0], _sig(c2, m, j, P_DEN).sig[0]
        same &= bool(np.array_equal(a[:d + 1], b[:d + 1]))
        after |= not np.array_equal(a[d + 1:], b[d + 1:])
    return same, after


def test_s8_cherez_obertku_ne_vidit_okno_do_d():
    same, after = _okno_ne_vidit_d()
    assert same and after            # after: возмущение не вхолостую — позже d сигнал меняется


@pytest.mark.parametrize("k", [0, 1])
def test_s8_mutaciya_okno_s_dnem_d_cherez_obertku_lovitsya(k, monkeypatch):
    monkeypatch.setattr(ks, "skor_s8", lambda c: 2 * ks.pct(np.log(c.shift(-k) / c.shift(252))) - 1)
    same, _ = _okno_ne_vidit_d()
    assert not same


def _chk_stik_goda(rebal):
    idx = pd.bdate_range("2018-12-24", "2019-01-11")
    sk = np.arange(len(idx), dtype=float)[:, None] * np.ones((1, 2))
    out = rebal(sk, idx, "nedelya")
    i = {d.strftime("%m-%d"): n for n, d in enumerate(idx)}
    assert (out[i["12-31"]:i["01-04"], 0] == sk[i["12-28"], 0]).all()      # W01 держит решение пятницы 28.12
    assert out[i["01-04"], 0] == sk[i["01-04"], 0] and (out[i["01-07"]:i["01-11"], 0] == sk[i["01-04"], 0]).all()


def test_nedelya_stik_goda_iso():
    _chk_stik_goda(kv.rebal)


def test_mutaciya_kalendarnyj_god_vmesto_iso_lovitsya():
    def mutant(skor, daty, rezhim):
        wk = pd.DatetimeIndex(daty).year * 100 + pd.DatetimeIndex(daty).isocalendar()["week"].to_numpy()
        posl = np.append(wk[1:] != wk[:-1], True)
        src = np.maximum.accumulate(np.where(posl, np.arange(len(wk)), -1))
        return np.asarray(skor, float)[np.maximum(src, 0)]

    with pytest.raises(AssertionError):
        _chk_stik_goda(mutant)


def test_obertka_nedelya_stik_goda():
    """Через обёртку: bufer = 0 → sig 31.12.2018–03.01.2019 равен sig пятницы 28.12 (решение недели не пересчитано)."""
    c, m = _ceny()
    p = {"rezhim": "nedelya", "porog": 0.2, "bufer": 0.0, "cap": 3}
    i28 = int(np.flatnonzero(IDX == pd.Timestamp("2018-12-28"))[0])
    assert i28 > WARMUP
    for sid in ("S7", "S8", "S9"):
        for j in range(N):
            s = _sig(c, m, j, p, sid).sig[0]
            assert (s[i28 + 1:i28 + 5] == s[i28]).all()


def test_exit_ravny_storone_snyata_i_sloy_vhoda():
    """exit_l = (sig ≠ +1), exit_s = (sig ≠ −1): при sig = +1 выход длинной не поднят, короткой — поднят."""
    c, m = _ceny()
    for j in range(N):
        ss = _sig(c, m, j, P_DEN)
        for v, a, b in ((1, False, True), (-1, True, False), (0, True, True)):
            mk = ss.sig[0] == v
            assert (ss.exit_l[0][mk] == a).all() and (ss.exit_s[0][mk] == b).all()


def test_poryadok_tochek_ne_vliyaet():
    c, m = _ceny()
    g = [P_DEN, {"rezhim": "nedelya", "porog": 0.4, "bufer": 0.1, "cap": 2}]
    x = c[COLS[2]].to_numpy()
    td = TickerData(COLS[2], IDX.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(T), np.zeros(T))
    f = vk.semya("S9", _ctx(c, m))
    a, b = f(td, {}, g), f(td, {}, g[::-1])
    assert np.array_equal(a.sig, b.sig[::-1]) and np.array_equal(a.strength, b.strength[::-1])
