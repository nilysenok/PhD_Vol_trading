"""4. Комиссия за сторону: net_t = pos_t·lr_t − |pos_t − pos_{t−1}|·c; вход c, выход c, разворот 2c.

Пути: s9.Engine.port (таблицы С9), msp.portfolio (МСП), approach_cd._eval (отбор C/D),
stats.sharpe_window (отбор фильтра оборота).
"""
import numpy as np
import pandas as pd

from fastbt import approach_cd, msp, s9, stats

POS = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 0.0])
LR = np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.005, 0.0])
C = 0.001
# |Δpos| вручную: 0,1,0,1,1,2,1,0 → 6 сторон; круг вход-выход = 2c
TURN = np.array([0, 1, 0, 1, 1, 2, 1, 0], float)
NET = POS * LR - TURN * C


def _half(f, pos_comm):
    def g(*a, **k):
        a = list(a)
        if "comm" in k:
            k["comm"] = k["comm"] / 2
        else:
            a[pos_comm] = a[pos_comm] / 2
        return f(*a, **k)
    return g


def _no_exit_at_end(f):
    """Мутация: прежнее соглашение — выход открытой на конце окна позиции не оплачивается."""
    def g(self, pos, comm=0.0005, *a, **k):
        p = f(self, pos, comm, *a, **k)
        p.iloc[-1, p.columns.get_loc("r")] += abs(pos[-1]) * comm
        return p
    return g


MUTATIONS = {"comm_no_exit_at_end": lambda mp: mp.setattr(s9.Engine, "port",
                                                          _no_exit_at_end(s9.Engine.port)),
             "comm_once_per_trade": lambda mp: (
    mp.setattr(s9.Engine, "port", _half(s9.Engine.port, 2)),
    mp.setattr(msp, "portfolio", _half(msp.portfolio, 2)),
    mp.setattr(approach_cd, "_eval", _half(approach_cd._eval, 5)),
    mp.setattr(stats, "sharpe_window", _half(stats.sharpe_window, 4)))}


def _sh(x):
    return np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)


def test_round_trip_costs_two_sides():
    assert TURN.sum() * C == 6 * C
    assert np.isclose((POS[:4] * LR[:4]).sum() - NET[:4].sum(), 2 * C)   # вход t=1, выход t=3


def test_open_position_at_window_end_pays_entry_and_exit():
    """Ъ владельца 26.10, 23:50 (аудит FA2a, Н5; было — соглашение спецификации п. 8 «без append»):
    позиция, открытая на последнем дне окна (30.12.2021 в разработке) и закрытая по close
    03.01.2022, платит вход и выход (10 сторон из 3 956, ΔШарп МСП порядка 1e-4)."""
    pos = np.array([0.0, 1.0, 1.0])
    n = len(pos)
    e = object.__new__(s9.Engine)
    e.r = {"date": np.arange(n)}
    e.first = np.r_[True, np.zeros(n - 1, bool)]
    e.lr_row, e.cnt = np.zeros(n), np.ones(n)
    e.dates = pd.bdate_range("2021-12-28", periods=n)
    e.kr = e.rf = np.zeros(n)
    e.cal = np.ones(n)
    p = s9.Engine.port(e, pos, C)
    assert np.isclose((p["gross"] - p["r"]).sum(), 2 * C)


def test_s9_engine_port():
    n = len(POS)
    e = object.__new__(s9.Engine)
    e.r = {"date": np.arange(n)}
    e.first = np.r_[True, np.zeros(n - 1, bool)]
    e.lr_row, e.cnt = LR, np.ones(n)
    e.dates = pd.bdate_range("2021-03-01", periods=n)
    e.kr = e.rf = np.zeros(n)
    e.cal = np.ones(n)
    p = s9.Engine.port(e, POS, C)
    assert np.allclose(p["r"], NET)
    assert np.allclose(p["gross"] - p["r"], TURN * C)
    assert np.allclose(p["turnover"], TURN)


def test_msp_portfolio():
    d = pd.DataFrame({"date": pd.bdate_range("2021-03-01", periods=len(POS)), "ticker": "ZZZ",
                      "strategy": "S1", "position": POS, "daily_gross_return": POS * LR})
    p = msp.portfolio(d, pd.Series([0.0], pd.to_datetime(["2020-01-01"])), C)
    assert np.allclose(p["r"], NET)


def test_selection_paths():
    m = np.ones((1, len(POS)), bool)
    sh, _, _ = approach_cd._eval(POS, m, LR, 0, len(POS), C)
    assert np.isclose(sh[0], _sh(NET))
    assert np.isclose(stats.sharpe_window(POS, LR, 0, len(POS), C), _sh(NET))
