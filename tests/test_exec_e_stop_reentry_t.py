"""Т-v611c (тестировщик): добавка к test_exec_e_stop_reentry.py — reentry_den на шорте по ручной ленте и «≤ 2 входов в день» по каждому seed.
Шорт: пара вечера d−1 = buy-stop 102 / sell-stop 98; 1σ = 0,01·цена, b=1 -> защитный стоп шорта 98·1,01 = 98,98.
Мутации: пара дня d вместо d−1; повторный вход в баре выхода; снят лимит «один повторный в день».
python3 -m pytest tests/test_exec_e_stop_reentry_t.py -q"""
import numpy as np
import pytest
from test_exec_e_stop_osi import N0, NAN, P, _day, _flat, _rnd

from fastbt import exec_e

BARS_S = [(98.5, 98.5, 97.6, 97.9),    # 0: sell-stop 98 -> вход шорта по 98 (high ниже стопа 98,98)
          (97.9, 99.2, 97.8, 99.1),    # 1: стоп 98,98 -> выход (open 97,9 лучше уровня); пара активна только с бара 2
          (98.5, 98.6, 97.5, 97.8),    # 2: sell-stop 98 -> повторный вход по 98 (не в баре выхода 1, хотя low бара 1 тоже ниже 98)
          (97.8, 99.1, 97.7, 99.0),    # 3: стоп 98,98 -> выход
          (99.0, 99.0, 97.0, 97.5)]    # 4: пара снова коснулась, но второго повторного за день нет


def _go_s(bars, p):
    days = [_flat() for _ in range(N0)] + [_day(*zip(*bars))]
    n = len(days)
    u = {k: np.full(n, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][N0 - 1], u["dn"][N0 - 1] = 102.0, 98.0
    u["up"][N0], u["dn"][N0] = 130.0, 90.0          # пара дня d: sell-stop 90 не достижим, повторный вход по ней невозможен
    return exec_e.run("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u)


def test_reentry_short_pare_vechera_d_minus_1():
    r = _go_s(BARS_S, P(b=1.0, reentry_den=True))
    a, b = r.trades
    assert (a.side, a.entry_bar, a.entry_px, a.exit_bar, a.reason, round(a.exit_px, 9)) == (-1, 0, 98.0, 1, "stop", 98.98)
    assert (b.side, b.entry_day, b.entry_bar, b.entry_px, b.exit_bar, b.reason, b.d_sig) == (-1, a.entry_day, 2, 98.0, 3, "stop", N0 - 1)
    assert len(r.trades) == 2 and (r.n_orders, r.n_fills) == (2, 2)
    assert len(_go_s(BARS_S, P(b=1.0)).trades) == 1  # без оси повторного входа нет


def test_reentry_short_gep_po_open():
    bars = BARS_S[:2] + [(97.0, 97.5, 96.5, 97.2)]    # open 97 ниже sell-stop 98 -> вход по open
    r = _go_s(bars, P(b=1.0, reentry_den=True))
    assert (len(r.trades), r.trades[1].entry_bar, r.trades[1].entry_px) == (2, 2, 97.0)


@pytest.mark.parametrize("seed", range(8))
def test_ne_bolshe_dvukh_vkhodov_v_den_na_kazhdom_seed(seed):
    p = P(b=2.0, a=3.0, b0=1.0, pir=0.5, ed_s="atr", chand=1.5, k_e=0.1, reentry_den=True, comm=0.0004)
    B, u = _rnd(seed)
    r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), p, urovni=u)
    by_day = {}
    for t in r.trades:
        if t.reason != "pir":
            by_day[t.entry_day] = by_day.get(t.entry_day, 0) + 1
    assert not by_day or max(by_day.values()) <= 2


def test_est_dni_s_dvumya_vkhodami_sredi_seed():
    p = P(b=2.0, a=3.0, b0=1.0, pir=0.5, ed_s="atr", chand=1.5, k_e=0.1, reentry_den=True, comm=0.0004)
    k = 0
    for seed in range(8):
        B, u = _rnd(seed)
        r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), p, urovni=u)
        days = [t.entry_day for t in r.trades if t.reason != "pir"]
        k += len(days) - len(set(days))
    assert k > 0
