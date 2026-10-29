"""Лист R-в для A_zashch (A (s9)): разложение lr на ночь/день, известные ответы на синтетике, без данных.
Мутация: части считаются по ценам дня d вместо d+1 (test_mutaciya_sdvig_dnya_lovitsya); мутации не подменяют модуль — функция передаётся аргументом."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "revers"))
import r1_v  # noqa: E402
import r1_v_a_zashch as R  # noqa: E402

C = np.array([100.0, 102.0, 101.0, 105.0])


def lr_of(c):
    out = np.zeros(len(c))
    out[:-1] = np.log(c[1:] / c[:-1])
    return out


def proverka(fn) -> bool:
    """Известные ответы: open[d+1] = close[d+1] → день 0; open[d+1] = close[d] → ночь 0; произвольный день — ln(c/o) дня d+1."""
    ok = np.allclose(fn(C, C), 0.0)                                   # цена открытия = закрытию: дневной части нет
    o = np.r_[np.nan, C[:-1]]                                         # open[d+1] = close[d]: гэпа нет, вся доходность — день
    I = fn(o, C)
    ok &= np.allclose(I, lr_of(C)) and np.allclose(lr_of(C) - I, 0.0)
    o2 = np.array([99.0, 101.0, 103.0, 104.0])
    return bool(ok and np.allclose(fn(o2, C)[:3], np.log(C[1:] / o2[1:])) and fn(o2, C)[3] == 0.0)


def test_vnutridnevnaya_izvestnye_otvety():
    assert proverka(R.vnutridnevnaya)


def test_mutaciya_sdvig_dnya_lovitsya():
    def sdvig(o, c):                       # части по ценам дня d вместо d+1
        I = np.zeros(len(c))
        ok = (np.asarray(o, float) > 0) & (c > 0)
        I[ok] = np.log(c[ok] / np.asarray(o, float)[ok])
        return I
    assert not proverka(sdvig)


def test_nan_open_ves_dohod_v_noch():
    I = R.vnutridnevnaya(np.array([1.0, np.nan, 1.0]), np.array([1.0, 2.0, 2.0]))
    assert I[0] == 0.0 and I[1] == pytest.approx(np.log(2.0))


def test_pusty_ryad():
    assert len(R.vnutridnevnaya(np.array([]), np.array([]))) == 0 and R.vnutridnevnaya(np.array([1.0]), np.array([1.0]))[0] == 0.0


def test_chasti_dnya_summa_i_nochnaya_poziciya():
    """Две пары на двух датах, обе сторона: N = pos × (lr − I), сумма = pos × lr, деление на cnt; позиция 0 — части 0."""
    lr, I_row = np.array([0.02, -0.01, 0.03, 0.0]), np.array([0.0, 0.0, 0.01, 0.0])
    pos, di, cnt = np.array([1.0, -1.0, 0.0, 1.0]), np.array([0, 0, 1, 1]), np.array([2.0, 2.0])
    I, N = R.chasti_dnya(lr, I_row, pos, di, cnt)
    assert np.allclose(I, [0.0, 0.0]) and np.allclose(N, [(0.02 + 0.01) / 2, 0.0])     # день 0: +0.02 и −(−0.01); день 1: поз 0 и lr 0
    assert np.allclose(I + N, np.bincount(di, pos * lr, 2) / cnt)
    assert R.doli_ravny(I, N, np.bincount(di, pos * lr, 2) / cnt) < 1e-15


def test_nochnaya_pozitsiya_dnevnaya_chast_nol():
    """Позиция, у которой весь рост приходится на гэп (open = close дня d+1), даёт дневную часть 0 и всю доходность в ночи."""
    c = np.array([100.0, 110.0, 110.0])
    I = R.vnutridnevnaya(c, c)
    ii, nn = R.chasti_dnya(lr_of(c), I, np.ones(3), np.arange(3), np.ones(3))
    assert np.allclose(ii, 0.0) and np.allclose(nn, lr_of(c))


def test_doli_ravny_lovit_raznitsu():
    with pytest.raises(AssertionError):
        R.doli_ravny(np.array([0.01]), np.array([0.01]), np.array([0.03]))


def test_pod_dvizhok_pervyj_bar_i_cnt():
    eng = SimpleNamespace(r=dict(pair=np.array([0, 0, 1, 1, 2, 2]), date=np.array([0, 1, 0, 1, 0, 1])),
                          lr_row=np.arange(6.0), dates=[0, 1], first=None, cnt=None)
    e = R.pod_dvizhok(eng, np.array([True, True, False, False, True, True]))
    assert list(e.first) == [True, False, True, False] and list(e.cnt) == [2.0, 2.0] and list(e.lr_row) == [0, 1, 4, 5]


def test_a_strogaya_put_ne_izmenen():
    """Умолчание --obekt — прежний расчёт; пути A_strogaya (PERIODS, ETALON, sverka) на месте."""
    import inspect
    assert inspect.signature(R.stroki).parameters["god"].default == 3.0
    assert 'default="A_strogaya"' in inspect.getsource(r1_v.main) and set(r1_v.PERIODS) == {"train4", "test1"}
