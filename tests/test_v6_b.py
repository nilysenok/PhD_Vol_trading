"""V6-b (Ж-111 пп. 7, 8б, 10; Ж-114 п. 5): безубыток b0 / b0_c, «стоп в плюс», ступенчатый трейлинг, пирамидинг
(fastbt/exec_e_v6.py). Синтетика с известным ответом, без данных. check_* — на правильном коде; test_*_mut — подложенная
ошибка (check обязан упасть).

Лента: день 0 — одна свеча 100, sig = 1 на всех днях, вход по close 100 (k = None). σ = 0,01, b = 2: начальный стоп 98,
R = 2, 1σ = 1 рубль."""
import numpy as np
import pytest

from fastbt import exec_e_v6, fast_etap1
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

COMM = 0.001


def day(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, None)


def go(*dd, rang=None, side=1, b=2.0, **kw):
    days = [day([(100, 100, 100, 100)])] + [day(d) for d in dd]
    return run("T", days, [1] * len(days), np.full(len(days), 0.01), Params(b=b, comm=COMM, side=side, reentry=False, **kw), rang=rang)


def check_b0():
    d = [(100, 101.5, 100, 101), (101, 101, 99.9, 100)]
    t = go(d, b0=1.0).trades[0]
    assert (t.reason, t.exit_bar, t.exit_px) == ("bu", 1, pytest.approx(100.0))
    assert go(d, b0=2.0).trades[0].reason is None          # порог 2σ = 102 не достигнут
    assert go([(100, 101.5, 99.5, 101)], b0=1.0).trades[0].reason is None   # действует со следующего бара, не в том же


def check_b0_c():
    d = [(100, 101.5, 100, 101), (101, 101, 99.9, 100)]
    assert go(d, b0_c=2.0, rang=[0.5, 0]).trades[0].reason == "bu"          # порог 2·0,5 = 1σ (ранг дня сигнала 0)
    assert go(d, b0_c=2.0, rang=[1.0, 0]).trades[0].reason is None          # 2σ не достигнуто
    assert go(d, b0_c=2.0, rang=[np.nan, 0]).trades[0].reason is None       # нет ранга — безубытка нет
    assert go(d, b0_c=2.0, rang=[0.0, 0]).trades[0].reason == "bu"          # ранг 0: порог 0, сразу со второго бара


def check_sp():
    t = go([(100, 102.5, 100, 102), (102, 102, 100.5, 101)], sp=(2.0, 1.0)).trades[0]
    assert (t.reason, t.exit_px) == ("trail", pytest.approx(101.0))            # после +2σ стоп на +1σ
    assert go([(100, 101.9, 100, 102), (102, 102, 100.5, 101)], sp=(2.0, 1.0)).trades[0].reason is None


def check_step():
    t = go([(100, 103.2, 100, 103), (103, 103, 100.9, 101)], step_s=1.0).trades[0]
    assert (t.reason, t.exit_px) == ("trail", pytest.approx(101.0))            # 3 шага: 98 + 3
    assert go([(100, 100.9, 100, 100), (100, 100, 97.9, 99)], step_s=1.0).trades[0].exit_px == pytest.approx(98.0)  # 0 шагов
    t = go([(100, 101.2, 100, 101), (101, 101, 98.9, 99)], step_s=0.5).trades[0]
    assert t.exit_px == pytest.approx(99.0)                                    # 2 шага по 0,5: 98 + 1


def check_pir_basic():
    """+1σ = 101 пройден (h 101,5): добавка 0,5 по 101. Гэп 97 на день 2 — стоп 98 → выход всех ног по 97."""
    r = go([(100, 101.5, 100, 101.5)], [(97, 97, 97, 97)], pir=0.5)
    par, leg = r.trades
    assert (leg.reason, leg.parent, leg.frac, leg.entry_px, leg.entry_bar) == ("pir", 0, 0.5, pytest.approx(101.0), 0)
    assert leg.fee_in == pytest.approx(0.5 * COMM) and leg.fee_out == pytest.approx(0.5 * COMM)
    assert (leg.exit_day, leg.exit_px, par.exit_px) == (2, pytest.approx(97.0), pytest.approx(97.0))
    assert par.fee_out == pytest.approx(COMM) and len([t for t in r.trades if t.reason != "pir"]) == 1
    assert r.pos[1] == pytest.approx(1.5)
    assert r.gross.sum() == pytest.approx(np.log(1.01) + 1.5 * np.log(97 / 101))
    assert r.cost.sum() == pytest.approx(3 * COMM)                             # вход 1 + добавка 0,5 + выход 1,5


def check_pir_risk_cap():
    """a = 1 при риске базы 2 (до стопа 98) и добавке по 101 (риск 3·a): потолок 2ρ = 4 → a = (4 − 2)/3."""
    r = go([(100, 101.5, 100, 101.5)], [(97, 97, 97, 97)], pir=1.0)
    assert r.trades[1].frac == pytest.approx(2 / 3)
    risk = 1.0 * 2.0 + r.trades[1].frac * (101.0 - 98.0)
    assert risk <= 4.0 + 1e-9


def check_pir_two_max():
    """b = 4 (стоп 96, R = 4), pir 0,2: +1σ, +2σ → по 0,2; на +3σ третьей добавки нет. Риск ног ≤ 2ρ = 8."""
    r = go([(100, 101.5, 100, 101.5)], [(101.5, 102.5, 101.5, 102.5)], [(102.5, 103.5, 102.5, 103.5)], b=4.0, pir=0.2)
    legs = [t for t in r.trades if t.reason == "pir"]
    assert [round(t.frac, 6) for t in legs] == [0.2, 0.2]
    assert 4.0 + sum(t.frac * (t.entry_px - 96.0) for t in legs) <= 8.0 + 1e-9
    assert r.pos[3] == pytest.approx(1.4)


def check_pir_against():
    """Падение на −1σ (99) без стопа: добавки против позиции нет."""
    r = go([(100, 100, 99, 99.5)], [(99.5, 99.8, 99.2, 99.5)], pir=1.0)
    assert len(r.trades) == 1 and r.pos[1] == 1.0


def check_pir_short():
    r = go([(100, 100, 98.5, 98.5)], [(103, 103, 103, 103)], side=-1, pir=0.5)       # стоп 102, гэп 103
    par, leg = r.trades
    assert (leg.side, leg.entry_px, leg.frac) == (-1, pytest.approx(99.0), 0.5) and par.side == -1
    assert r.pos[1] == pytest.approx(-1.5)
    assert r.gross.sum() == pytest.approx(-np.log(99 / 100) - 1.5 * np.log(103 / 99))


def check_pir_not_touch_and_entry_bar():
    r = go([(100, 101.0, 100, 100.5)], pir=0.5)               # касание +1σ (h = 101) — не проход
    assert len(r.trades) == 1


def test_b0_sp_step():
    check_b0()
    check_b0_c()
    check_sp()
    check_step()


def test_pir():
    check_pir_basic()
    check_pir_risk_cap()
    check_pir_two_max()
    check_pir_against()
    check_pir_short()
    check_pir_not_touch_and_entry_bar()


def test_pir_stop_before_add():
    """Бар и с проходом +1σ, и с падением до стопа 98 — стоп сначала (открытие 100 → низ 97): добавки нет, выход по 98."""
    r = go([(100, 101.5, 97, 98)], pir=1.0)
    assert len(r.trades) == 1 and r.trades[0].reason == "stop"


def test_default_off_identical():
    """Без новых полей — одна сделка, frac 1, parent None, комиссия 2·COMM; поля по умолчанию выключены."""
    p = Params()
    assert (p.b0, p.b0_c, p.sp, p.step_s, p.pir) == (None, None, None, None, 0.0)
    r = go([(100, 101.5, 100, 101)], [(97, 97, 97, 97)])
    assert len(r.trades) == 1 and r.trades[0].frac == 1.0 and r.trades[0].parent is None
    assert r.cost.sum() == pytest.approx(2 * COMM)


def test_validation():
    base = [(100, 100, 100, 100)]
    for kw in (dict(b=None, pir=0.5), dict(b0=1.0, be=True), dict(b0_c=1.0), dict(sp=(1.0, 1.0)), dict(step_s=0.0),
               dict(pir=1.5), dict(b0=1.0, b0_c=1.0, rang=[0.5, 0.5]), dict(b0_c=1.0, rang=[0.5])):
        with pytest.raises(ValueError):
            go(base, **kw)


def test_fast_route_slow_and_equal():
    p = Params(b=2.0, comm=COMM, pir=0.5, step_s=1.0)
    assert fast_etap1._slow_only(p, None) and not fast_etap1._slow_only(Params(b=2.0), None)
    days = [day([(100, 100, 100, 100)]), day([(100, 103.2, 100, 103)]), day([(103, 103, 100.9, 101)])]
    a = run("T", days, [1, 1, 1], np.full(3, 0.01), p)
    b = fast_etap1.run("T", days, np.array([1, 1, 1]), np.full(3, 0.01), p)
    assert np.array_equal(a.gross, b.gross) and np.array_equal(a.cost, b.cost) and len(a.trades) == len(b.trades)


# --- подложенные ошибки: check обязан упасть ---
def mut(monkeypatch, name, fn, check):
    monkeypatch.setattr(exec_e_v6, name, fn)
    with pytest.raises(AssertionError):
        check()


def test_mut_b0_ignored(monkeypatch):
    mut(monkeypatch, "be_mult", lambda p, r: 1.0 if p.be else None, check_b0)


def test_mut_rang_ignored(monkeypatch):
    mut(monkeypatch, "be_mult", lambda p, r: p.b0_c, check_b0_c)


def test_mut_sp_uses_p(monkeypatch):
    mut(monkeypatch, "plus_level", lambda sd, px, sg, q: px * (1 + sd * 2 * sg), check_sp)


def test_mut_step_off_by_one(monkeypatch):
    mut(monkeypatch, "step_level", lambda sd, px, sg, st, ext, b: px * (1 - sd * b * sg) + sd * 4 * st * sg * px, check_step)


def test_mut_no_risk_cap(monkeypatch):
    mut(monkeypatch, "pir_size", lambda sd, pa, S, legs, r0, a: a, check_pir_risk_cap)


def test_mut_pir_against(monkeypatch):
    mut(monkeypatch, "pir_level", lambda sd, px, sg, m: px * (1 - sd * m * sg), check_pir_against)


def test_mut_pir_touch(monkeypatch):
    monkeypatch.setattr(exec_e_v6, "take_hit", lambda sd, o, h, l, T, tick: (h >= T, max(o, T)))
    with pytest.raises(AssertionError):
        check_pir_not_touch_and_entry_bar()


def test_mut_three_adds(monkeypatch):
    monkeypatch.setattr(exec_e_v6, "PIR_MAX", 3)
    with pytest.raises(AssertionError):
        check_pir_two_max()
