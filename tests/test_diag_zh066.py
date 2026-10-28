"""Ж-066 п. 1: диагностические таблицы — синтетика с известным ответом + мутации (testy.md п. 1, mutations.md № 2).
Мутации подкладываются monkeypatch'ем; проверка «ловится» — известный ответ обязан перестать сходиться."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fastbt import diag_zh066 as dz
from fastbt import exec_e, exec_e_run
from fastbt.diag_zh066_run import log_orders
from fastbt.exec_e_types import Day, Params


def tr(sila, pnl, side=1, **kw):
    return dict(tiker="AAA", d_in="2018-01-02", d_out="2018-01-03", side=side, pnl=pnl, sila=sila, d_sig="2018-01-01", **kw)


# ---------- (а) децили ----------
def test_deciles_known_answer():
    trades = [tr(float(i), 0.01 * (1 if i % 2 else -1) * i) for i in range(1, 21)]      # сила 1..20 вперемешку по порядку
    rows = dz.deciles(trades[::-1])                                                      # порядок на входе не важен
    assert [r[1] for r in rows] == [2] * 10
    assert (rows[0][2], rows[0][3]) == (1.0, 2.0) and (rows[9][2], rows[9][3]) == (19.0, 20.0)
    assert rows[0][4] == pytest.approx(100 * (0.01 * 1 - 0.01 * 2))                    # Σpnl % дециля 1: +1 и −2
    assert rows[0][6] == 0.5 and all(r[7] == "шум" for r in rows)                      # n = 2 < 10


def test_deciles_sum_equals_total_and_nan_row():
    trades = [tr(float(i), 0.001 * i) for i in range(1, 101)] + [tr(float("nan"), 0.5)]
    rows = dz.deciles(trades)
    assert sum(r[1] for r in rows) == 101 and rows[-1][0] == "нет"
    assert sum(r[4] for r in rows) == pytest.approx(100 * sum(t["pnl"] for t in trades), abs=1e-9)
    assert all(r[7] == "" for r in rows[:10])                                            # по 10 сделок — не шум


def test_deciles_ties_in_strength_flagged():
    trades = [tr(0.5 + 0.01 * i, 0.01) for i in range(10)] + [tr(1.0, 0.01) for _ in range(30)]      # 30 сделок с силой, равной 1
    rows = dz.deciles(trades)
    assert "ничья" not in rows[0][7] and all("ничья по силе" in r[7] for r in rows[3:])


def test_deciles_empty_tape_no_crash():
    assert all(r[1] == 0 for r in dz.deciles([])[:10])


# ---------- (в) лонг/шорт ----------
def _check_long_short():
    t = [tr(0.5, 0.02, 1), tr(0.5, -0.01, 1), tr(0.5, 0.03, -1), tr(0.5, 0.05, -1), tr(0.5, -0.02, -1)]
    lg, sh, al = dz.long_short(t)
    assert (lg[0], lg[1], lg[4]) == ("long", 2, 0.5) and lg[2] == pytest.approx(1.0) and lg[3] == pytest.approx(0.5)
    assert (sh[1], sh[4]) == (3, pytest.approx(2 / 3)) and sh[2] == pytest.approx(6.0) and sh[3] == pytest.approx(2.0)
    assert al[1] == 5 and al[2] == pytest.approx(lg[2] + sh[2])


def test_long_short_known_answer():
    _check_long_short()


def test_mutation_short_in_long_column_caught(monkeypatch):
    def bad(trades):                                    # сделки шорта попадают в колонку лонга
        return [(n, *dz._stat([t["pnl"] for t in trades if sel(t["side"])]))
                for n, sel in (("long", lambda s: s != 0), ("short", lambda s: s < 0), ("все", lambda s: True))]
    monkeypatch.setattr(dz, "long_short", bad)
    with pytest.raises(AssertionError):
        _check_long_short()


# ---------- (б) корреляции ----------
def test_corr_symmetric_unit_diagonal_and_known_values():
    s1 = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    ser = {"S1": dz.pd.Series(s1), "S2": dz.pd.Series(2 * s1), "S3": dz.pd.Series(-s1)}
    for m in ("pearson", "spearman"):
        c = dz.corr_matrix(ser, m)
        assert np.allclose(c, c.T) and np.allclose(np.diag(c), 1.0)
        assert c.loc["S1", "S2"] == pytest.approx(1.0) and c.loc["S1", "S3"] == pytest.approx(-1.0)


# ---------- (г) проскальзывание и доля исполнения ----------
def order(side, L, px, close_d=100.0, isp=True, tiker="AAA"):
    return dict(tiker=tiker, d_sig="2018-01-01", side=side, L=L, close_d=close_d, isp=isp, px=px if isp else float("nan"))


def _check_limits():
    assert dz.slip(order(1, 98.0, 98.0)) == pytest.approx(-2.0)        # лонг купил дешевле решения — улучшение
    assert dz.slip(order(-1, 102.0, 102.0)) == pytest.approx(-2.0)     # шорт продал дороже решения — тоже улучшение
    assert dz.slip(order(-1, 102.0, 99.0)) == pytest.approx(+1.0)      # шорт продал дешевле решения — против сделки, плюс
    assert dz.slip(order(1, 98.0, 101.0)) == pytest.approx(+1.0)       # лонг купил дороже решения — против сделки, плюс
    assert dz.gap_gain(order(1, 98.0, 97.0)) == pytest.approx(100 * 1.0 / 98.0)
    assert dz.gap_gain(order(-1, 102.0, 103.0)) == pytest.approx(100 * 1.0 / 102.0)
    rows = dz.limit_rows([order(1, 98, 98), order(1, 98, 0, isp=False), order(-1, 102, 103, tiker="BBB"),
                          order(-1, 102, 0, isp=False, tiker="BBB"), order(-1, 102, 0, isp=False, tiker="BBB")])
    by = {r[0]: r for r in rows}
    assert by["AAA"][1:4] == (2, 1, 0.5) and by["BBB"][1:4] == (3, 1, pytest.approx(1 / 3))
    assert by["все"][1:3] == (5, 2) and by["BBB"][5] == 1 and by["AAA"][5] == 0 and math.isnan(by["AAA"][6])


def test_limits_known_answer():
    _check_limits()


def test_mutation_short_slip_sign_flipped_caught(monkeypatch):
    monkeypatch.setattr(dz, "slip", lambda o: 100.0 * (o["px"] - o["close_d"]) / o["close_d"])   # знак шорта не перевёрнут
    with pytest.raises(AssertionError):
        _check_limits()


# ---------- журнал заявок из реального движка ----------
def day(o, h, l, c, n=3):
    return Day(*[np.full(n, float(x)) for x in (o, h, l, c)])


def _run(side, d1):
    days = [day(100, 100.5, 99.5, 100), d1, day(100, 100.5, 99.5, 100)]
    sink: list = []
    with log_orders(sink):
        exec_e.run("AAA", days, np.array([side, 0, 0]), np.full(3, 0.02),
                   Params(k=1.0, tick=0.1, check="bary", comm=0.0, exit_signal=True))
    return sink


def test_orders_logged_fill_pass_and_touch():
    (p,) = _run(1, day(100, 100.5, 97.9, 99))                     # лонг: L = 98.0, low 97.9 = сквозь на 1 тик
    assert p["isp"] and p["px"] == pytest.approx(98.0) and p["close_d"] == 100.0 and p["L"] == pytest.approx(98.0)
    (t,) = _run(1, day(100, 100.5, 98.0, 99))                     # касание L без прохода на тик — не исполнено
    assert not t["isp"] and math.isnan(t["px"])
    (g,) = _run(1, day(97.0, 100.5, 96.5, 99))                    # гэп вниз: исполнение по open 97 лучше L
    assert g["px"] == pytest.approx(97.0) and dz.gap_gain(g) > 0
    (s,) = _run(-1, day(100, 102.1, 99.5, 101))                   # шорт: L = 102.0, high 102.1
    assert s["isp"] and s["side"] == -1 and dz.slip(s) == pytest.approx(-2.0)


def test_log_orders_restores_engine():
    before = (exec_e_run.Run._place, exec_e_run.Run._try_order)
    _run(1, day(100, 100.5, 97.9, 99))
    assert (exec_e_run.Run._place, exec_e_run.Run._try_order) == before


def test_mutation_touch_counted_as_fill_caught(monkeypatch):
    monkeypatch.setattr(exec_e_run, "limit_mask", lambda sd, h, l, L, tick: (l <= L) if sd > 0 else (h >= L))
    (t,) = _run(1, day(100, 100.5, 98.0, 99))
    with pytest.raises(AssertionError):
        assert not t["isp"]                                       # касание ошибочно стало исполнением -> тест роняется


# ---------- сверка с журналом ----------
def test_reconcile_detects_changed_pnl_and_missing_trade():
    j = [dict(tiker="AAA", d_in="2018-01-02", d_out="2018-01-03", pnl=0.01), dict(tiker="BBB", d_in="2018-01-02", d_out="2018-01-04", pnl=-0.02)]
    t = [dict(j[0]), dict(j[1])]
    assert dz.reconcile(t, j) == 0.0
    with pytest.raises(AssertionError):
        dz.reconcile([t[0], dict(t[1], pnl=-0.0200001)], j)
    with pytest.raises(AssertionError):
        dz.reconcile(t[:1], j)
