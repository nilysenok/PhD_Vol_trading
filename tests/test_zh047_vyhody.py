"""Ж-047, тесты 2.6–2.8, 2.13, 2.16 (контракт §2): трейлинг, безубыток, режим `zakrytie`, комиссия, повторный вход.
Синтетика; интерфейс К2 (PR #349): Params(k_tr=, be=, check=, reentry=), exec_e.run(close=). Мутации — monkeypatch."""
import numpy as np
import pytest

from fastbt import exec_e, exec_e_run
from test_zh047_pomoshch import F, P, SG, D, first, go, mutate

C = 0.0004


def sc_trail():                                                    # 2.6: M = max high ДО бара; трейл = M·(1 − σ)
    r = go([(100, 110, 100, 105), (105, 106, 102, 104), (104, 104, 104, 104)], p=P(R=None, k_tr=1.0))
    t = first(r)
    assert (t.exit_bar, t.exit_px, t.reason) == (1, 103.125, "trail")


def sc_bu():                                                       # 2.7
    r = go([(100, 106.25, 99, 100), (104, 104, 99, 100)], p=P(R=None, be=True))
    assert (first(r).exit_bar, first(r).exit_px, first(r).reason) == (1, 100.0, "bu"), "БУ — со следующего бара"
    r = go([(100, 106.0, 99, 100), (104, 104, 99, 100)], p=P(R=None, be=True))
    assert first(r).reason == "signal", "high < P(1+σ): безубытка нет"


def run_zak(close, sig):
    days = [D(F(100.0)), D([(100, 100, 90, 100), (100, 100, 99, 100)]), D(F(100.0, 2))]
    return exec_e.run("T", days, sig, np.full(3, SG), P(R=None, check="zakrytie"), close=close)


def sc_zakrytie():                                                 # 2.8
    r = run_zak([100.0, 101.0, 100.0], [1, 0, 0])
    assert (first(r).reason, first(r).exit_px) == ("signal", 101.0), "выход по close Н6, а не по 5-мин close"
    r = run_zak([100.0, 93.75, 100.0], [1, 1, 0])
    assert (first(r).reason, first(r).exit_px) == ("stop", 93.75), "стоп при C_d ≤ S"
    r = run_zak([100.0] * 3, [1, 1, 0])
    assert first(r).reason == "signal", "low = 90 внутри дня стопа в режиме zakrytie не даёт"


@pytest.mark.parametrize("sc", [sc_trail, sc_bu, sc_zakrytie])
def test_vyhod_zelenyy(sc):
    sc()


def test_scan_copy_equals_real(monkeypatch):
    mutate(monkeypatch, "")
    sc_trail()
    sc_bu()


def test_mut_trail_same_bar(monkeypatch):
    mutate(monkeypatch, "same_bar")
    with pytest.raises(AssertionError):
        sc_trail()


def test_mut_bu_same_bar(monkeypatch):
    mutate(monkeypatch, "same_bar")
    with pytest.raises(AssertionError):
        sc_bu()


def test_mut_zakrytie_by_5min_close(monkeypatch):                  # отметка по 5-мин close вместо Н6
    monkeypatch.setattr(exec_e_run.Run, "close_of", lambda s, i: float(s.days[i].c[-1]))
    with pytest.raises(AssertionError):
        sc_zakrytie()


# --- 2.13 комиссия ---
def sc_comm():
    days = [D(F(100.0)), D([(100, 100, 93.0, 100)]), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 0, 0], np.full(3, SG), exec_e.Params(b=1.0, comm=C))
    assert (r.cost[0], r.cost[1]) == (pytest.approx(2 * C), 0.0), "стоп внутри дня — в строке начала интервала"
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0))]
    r = exec_e.run("T", days, [1, 1, 1], np.full(3, SG), exec_e.Params(comm=C))
    assert r.cost[-1] == pytest.approx(C), "позиция на конце окна платит выход (Н5)"


def test_komissiya_granitsy_zelenyy():
    sc_comm()


def test_mut_comm_row_of_exit_day(monkeypatch):                    # стоимость стопа — в строке дня выхода
    monkeypatch.setattr(exec_e_run.Run, "_row", lambda s, i, j, at_close: i)
    with pytest.raises(AssertionError):
        sc_comm()


def test_mut_comm_window_end_unpaid(monkeypatch):                  # мутация № 5: выход на конце окна не платится
    orig = exec_e.run

    def no_end(*a, **kw):
        r = orig(*a, **kw)
        if r.pos[-1]:
            r.cost[-1] -= a[-1].comm
        return r
    monkeypatch.setattr(exec_e, "run", no_end)
    with pytest.raises(AssertionError):
        sc_comm()


# --- 2.16 повторный вход блокирован до новой сделки Sk ---
def sc_reentry():
    days = [D(F(100.0)), D([(100, 100, 93.0, 100)]), D(F(100.0)), D(F(100.0)), D(F(100.0, 2))]
    p = exec_e.Params(b=1.0, comm=C, stop_touch=True, reentry=False)
    r = exec_e.run("T", days, [1, 1, 1, 0, 1], np.full(5, SG), p)
    assert [t.entry_day for t in r.trades] == [0, 4], "после стопа при sig = 1 нового входа нет до сигнала 0"


def test_povtornyy_vhod_blokirovan():
    sc_reentry()


def test_mut_reentry_same_close(monkeypatch):                      # вход на том же close после стопа
    orig = exec_e.Params
    monkeypatch.setattr(exec_e, "Params", lambda **kw: orig(**{**kw, "reentry": True}))
    with pytest.raises(AssertionError):
        sc_reentry()
