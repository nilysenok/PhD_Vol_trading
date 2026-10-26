"""13. Сделка: вход по close сигнального бара, PnL = знак·ln(close[x]/close[e]) − комиссия.

Решение на закрытии d, позиция d на [вход, выход), доходность d→d+1 (спецификация пп. 2, 8, 14).

Пути: engine.lvl_run → stats.pos_from_trades (позиции финалистов, ансамбль, файл позиций) и
stats.seg_stats_trades (Шарп отбора A по сегментам), pipeline._rows (daily_gross_return).
Вход по close сигнального бара — правило спецификации (п. 2, как в c22a4b3), не заглядывание:
сигнал бара d считается по close[d], позиция d зарабатывает close[d]→close[d+1].
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from _eng import bars, mutant, pnl, run_lvl
from fastbt import pipeline, stats

C = 0.001
CLOSE = [100, 101, 102, 104, 103, 105, 107, 106, 108, 110, 109, 111.0]


def _pos_shift(k):
    def f(te, tx, tdir, nt, n):
        pos = np.zeros(n)
        for i in range(nt):
            for t in range(max(te[i] + k, 0), min(tx[i] + k, n)):
                pos[t] = float(tdir[i])
        return pos
    return f


MUTATIONS = {
    # позиция зарабатывает доходность (d−1, d] бара, на закрытии которого принято решение
    "pos_lookahead": lambda mp: mp.setattr(stats, "pos_from_trades", _pos_shift(-1)),
    # вход по close следующего бара (лишний лаг)
    "entry_next_close": lambda mp: mp.setattr(stats, "pos_from_trades", _pos_shift(1)),
    # отбор A: доходность бара сделки берётся со сдвигом назад (знание close[d] до решения d−1)
    "sel_stats_lookahead": lambda mp: mp.setattr(stats, "seg_stats_trades", mutant(
        stats.seg_stats_trades, [("d * lr[e]", "d * lr[e - 1]", 1),
                                 ("d * lr[t]", "d * lr[t - 1]", 1)])),
}


def _lr(c):
    lr = np.zeros(len(c))
    lr[:-1] = np.log(c[1:] / c[:-1])
    return lr


def _trades():
    c, h, lo = bars(CLOSE)
    # long на баре 2 (сигнальный выход на 6), short на 8 (выход на 10); стопы далеко
    return run_lvl(c, h, lo, longs=[2], shorts=[8], exl=[6], exs=[10], a1=5, a2=50), c


def _seg(trades, c):
    n = len(c)
    te = np.array([t[0] for t in trades], np.int64)
    tx = np.array([t[1] for t in trades], np.int64)
    td = np.array([t[2] for t in trades], np.int8)
    out = np.zeros((1, stats.NF))
    stats.seg_stats_trades(te, tx, td, len(te), _lr(c), np.array([0, n], np.int64),
                           np.zeros(n, np.int64), n, C, out)
    return out[0], te, tx, td


def test_trades_enter_on_signal_bar_exit_on_exit_bar():
    tr, _ = _trades()
    assert tr == [(2, 6, 1), (8, 10, -1)]


def test_position_series_and_pnl_by_hand():
    tr, c = _trades()
    _, te, tx, td = _seg(tr, c)
    pos = stats.pos_from_trades(te, tx, td, len(te), len(c))
    want = np.zeros(len(c))
    want[2:6] = 1.0
    want[8:10] = -1.0
    assert np.array_equal(pos, want)
    # ручной расчёт: long 102 → 107, short 108 → 109: ln(107/102) − ln(109/108)
    hand = np.log(107 / 102) - np.log(109 / 108)
    assert np.isclose((pos * _lr(c)).sum(), hand, atol=1e-15)
    assert np.allclose(pnl(tr, c), [np.log(107 / 102), -np.log(109 / 108)])
    # выход по close[x]: последняя доходность сделки — бар x−1 (close[x−1] → close[x])
    assert pos[5] == 1.0 and pos[6] == 0.0


def test_selection_stats_net_pnl():
    tr, c = _trades()
    seg, *_ = _seg(tr, c)
    hand = np.log(107 / 102) - np.log(109 / 108) - 4 * C     # 2 входа + 2 выхода
    assert np.isclose(seg[stats.F_S], hand, atol=1e-15)
    assert seg[stats.F_E] == 6 and seg[stats.F_R] == 2


def test_reversal_same_bar_pays_2c():
    """Сигнальный выход long и short-сигнал на баре 6: разворот на close[6], комиссия 2c."""
    c, h, lo = bars(CLOSE)
    tr = run_lvl(c, h, lo, longs=[2], shorts=[6], exl=[6], exs=[9], a1=5, a2=50)
    assert tr == [(2, 6, 1), (6, 9, -1)]
    seg, te, tx, td = _seg(tr, c)
    pos = stats.pos_from_trades(te, tx, td, len(te), len(c))
    hand = np.log(107 / 102) - np.log(110 / 107)
    assert np.isclose((pos * _lr(c)).sum(), hand, atol=1e-15)
    assert np.isclose(seg[stats.F_S], hand - 4 * C, atol=1e-15)   # c + 2c + c
    assert seg[stats.F_R] == 1                                     # вход из нуля — один


def test_open_trade_at_end_no_exit_commission():
    c, h, lo = bars(CLOSE)
    tr = run_lvl(c, h, lo, longs=[9], a1=50, a2=50)
    assert tr == [(9, len(c), 1)]
    seg, *_ = _seg(tr, c)
    assert np.isclose(seg[stats.F_S], np.log(111 / 110) - C, atol=1e-15)


def _engine2(pos_rows, lr_last=0.02):
    """s9.Engine на 2 парах × 3 дня (строки пары подряд); последний день окна — 30.12.2021,
    его доходность — до close 03.01.2022 (lr последней строки)."""
    from fastbt import msp, s9
    e = object.__new__(s9.Engine)
    e.r = {"date": np.array([0, 1, 2, 0, 1, 2])}
    e.first = np.array([True, False, False, True, False, False])
    e.lr_row = np.array([0.0, 0.0, lr_last, 0.0, 0.0, lr_last])
    e.cnt = np.full(3, 2.0)
    e.dates = pd.to_datetime(["2021-12-28", "2021-12-29", "2021-12-30"])
    e.kr = e.rf = np.zeros(3)
    e.cal = msp.cal_days(e.dates)
    return e


POS2 = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])   # пара 1 открыта на конце окна, пара 2 закрыта


@pytest.mark.xfail(strict=True, reason="Н5 (владелец 26.10, 23:50): позиция, открытая на последнем "
                                       "баре окна и закрытая по close первого бара после окна, "
                                       "должна платить комиссию выхода")
def test_window_end_exit_commission_s9():
    from fastbt import s9
    e = _engine2(POS2)
    p = s9.Engine.port(e, POS2, C)
    # пара 1: вход C + выход на конце окна C; пара 2: вход C + выход C → (2C + 2C)/2 пары
    assert np.isclose((p["gross"] - p["r"]).sum(), 2 * C)
    assert np.isclose((p["gross"] - p["r"]).iloc[-1], C / 2)     # выход пары 1 — в последний день
    assert np.isclose(p["gross"].iloc[-1], 0.02 / 2)              # доходность до 03.01 остаётся
    pw = s9.Engine.port(e, POS2, C, 1.0, np.full(6, 0.5))         # с весами (P1, B)
    assert np.isclose((pw["gross"] - pw["r"]).sum(), 4 * C * 0.5)


@pytest.mark.xfail(strict=True, reason="Н5: то же в msp.portfolio (табл. маржи, тест главы 4)")
def test_window_end_exit_commission_msp():
    from fastbt import msp
    dates = pd.to_datetime(["2021-12-28", "2021-12-29", "2021-12-30"])
    d = pd.DataFrame({"date": list(dates) * 2, "ticker": ["AAA"] * 3 + ["BBB"] * 3,
                      "strategy": "S1", "position": POS2,
                      "daily_gross_return": POS2 * np.array([0, 0, 0.02] * 2)})
    p = msp.portfolio(d, pd.Series([0.0], pd.to_datetime(["2020-01-01"])), C)
    assert np.isclose((p["r"] - (d.groupby("date")["daily_gross_return"].mean())).sum(), -2 * C)
    assert np.isclose(p["r"].iloc[-1], 0.02 / 2 - C / 2)


def test_pipeline_rows_gross_return():
    """Файл позиций: daily_gross_return[d] = position[d]·ln(close[d+1]/close[d])."""
    tr, c = _trades()
    _, te, tx, td = _seg(tr, c)
    pos = stats.pos_from_trades(te, tx, td, len(te), len(c))
    dates = pd.bdate_range("2021-03-01", periods=len(c)).values
    u = SimpleNamespace(dates=dates, lr=_lr(c), ticker="ZZZ")
    df = pipeline._rows(u, "S1", "A", {2021: (None, pos)}, [2021])[0]
    assert np.allclose(df["position"], want := pos) and len(want) == len(c)
    assert np.allclose(df["daily_gross_return"], pos * _lr(c), atol=1e-8)
    assert np.isclose(df["daily_gross_return"].sum(), np.log(107 / 102) - np.log(109 / 108),
                      atol=1e-7)
