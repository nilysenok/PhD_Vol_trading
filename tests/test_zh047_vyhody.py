"""Ж-047, тесты 2.6–2.8, 2.13, 2.16 (контракт §2): трейлинг, безубыток, режим `zakrytie`, комиссия, повторный вход.
Синтетика. Допущения об интерфейсе К2 — в test_zh047_pomoshch.py; `run_e1(ticker, days, sig, sigma, p, close=)`
(контракт §1, поток данных (2)) — имя и порядок аргументов сверить с zh047-k2. Мутации — monkeypatch."""
import numpy as np
import pytest

from fastbt import exec_e
from test_zh047_pomoshch import F, P, SG, WAIT, D, _absent, first, go, mutate, wait

C = 0.0004


def sc_trail():                                                    # 2.6: M = max high ДО бара; трейл = M·(1 − σ)
    r = go([(100, 110, 100, 105), (105, 106, 102, 104), (104, 104, 104, 104)], p=P(R=None, ktr=1.0))
    t = first(r)
    assert (t.exit_bar, t.exit_px, t.reason in ("trail", "stop")) == (1, 103.125, True)


def sc_bu():                                                       # 2.7
    r = go([(100, 106.25, 99, 100), (104, 104, 99, 100)], p=P(R=None, bu=True))
    assert (first(r).exit_bar, first(r).exit_px) == (1, 100.0), "БУ — со следующего бара, выход по P_вх"
    r = go([(100, 106.0, 99, 100), (104, 104, 99, 100)], p=P(R=None, bu=True))
    assert first(r).reason == "signal", "high < P(1+σ): безубытка нет"


def sc_zakrytie():                                                 # 2.8
    close = [100.0, 101.0, 100.0]
    days = [D(F(100.0)), D([(100, 100, 90, 100), (100, 100, 99, 100)]), D(F(100.0, 2))]
    r = exec_e.run_e1("T", days, [1, 0, 0], np.full(3, SG), P(R=None, rezhim="zakrytie"), close=close)
    assert (first(r).reason, first(r).exit_px) == ("signal", 101.0), "выход по close Н6, а не по 5-мин close"
    close = [100.0, 93.75, 100.0]
    r = exec_e.run_e1("T", days, [1, 1, 0], np.full(3, SG), P(R=None, rezhim="zakrytie"), close=close)
    assert (first(r).reason, first(r).exit_px) == ("stop", 93.75), "стоп при C_d ≤ S"
    r = exec_e.run_e1("T", days, [1, 1, 0], np.full(3, SG), P(R=None, rezhim="zakrytie"), close=[100.0] * 3)
    assert first(r).reason == "signal", "low = 90 внутри дня стопа в режиме zakrytie не даёт"


@pytest.mark.parametrize("sc", [sc_trail, sc_bu, sc_zakrytie])
@wait(2)
def test_vyhod_zelenyy(sc):
    sc()


@pytest.mark.parametrize("sc", [sc_trail, sc_bu])
def test_scenarii_soglasovany_s_etalonnym_prohodom(monkeypatch, sc):
    from test_zh047_ispolnenie import shim_params
    shim_params(monkeypatch)
    sc()


@wait(2)
def test_mut_trail_same_bar(monkeypatch):
    mutate(monkeypatch, "trail_same_bar")
    with pytest.raises(AssertionError):
        sc_trail()


@wait(2)
def test_mut_bu_same_bar(monkeypatch):
    mutate(monkeypatch, "be_same_bar")
    with pytest.raises(AssertionError):
        sc_bu()


@wait(2)
def test_mut_zakrytie_by_5min_close(monkeypatch):                  # отметка по 5-мин close вместо Н6
    orig = exec_e.run_e1
    monkeypatch.setattr(exec_e, "run_e1", lambda *a, **kw: orig(*a, **{**kw, "close": None}))
    with pytest.raises(AssertionError):
        sc_zakrytie()


# --- 2.13 комиссия (работает на действующем коде: живой тест, без маркера ожидания) ---
def sc_comm():
    days = [D(F(100.0)), D([(100, 100, 93.0, 100)]), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 0, 0], np.full(3, SG), exec_e.Params(b=1.0, comm=C))
    assert (r.cost[0], r.cost[1]) == (pytest.approx(2 * C), 0.0), "стоп внутри дня — в строке начала интервала"
    days = [D(F(100.0)), D(F(100.0)), D(F(100.0))]
    r = exec_e.run("T", days, [1, 1, 1], np.full(3, SG), exec_e.Params(comm=C))
    assert r.cost[-1] == pytest.approx(C), "позиция на конце окна платит выход (Н5)"


def test_komissiya_granitsy_zelenyy():
    sc_comm()


def test_mut_comm_row_of_exit_day(monkeypatch):                    # мутация: стоимость стопа — в строке дня выхода
    monkeypatch.setattr(exec_e._Run, "_row", lambda s, i, j, at_close: i)
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
@pytest.mark.xfail(_absent(2), strict=True, raises=(AssertionError,) + WAIT, reason="ждёт К2: сейчас вход на том же close (E §2)")
def test_povtornyy_vhod_blokirovan():
    days = [D(F(100.0)), D([(100, 100, 93.0, 100)]), D(F(100.0)), D(F(100.0)), D(F(100.0, 2))]
    r = exec_e.run("T", days, [1, 1, 1, 0, 1], np.full(5, SG), exec_e.Params(b=1.0, comm=C))
    assert [t.entry_day for t in r.trades] == [0, 4], "после стопа при sig = 1 нового входа нет до сигнала 0"
