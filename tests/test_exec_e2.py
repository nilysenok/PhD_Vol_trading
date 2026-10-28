"""Л-2: новые оси исполнения (fastbt/exec_e2.py, Run2). Синтетика с известным ответом, без данных.
check_* — на правильном коде; *_mut — подложенная ошибка (check обязан упасть). Мутации: № 2, 3 (mutations.md) и из §7 контракта.
Время баров (Л-3, fastbt/exec_e_vremya) подменено заглушкой по контракту: bar_do ищет бар, начатый за 5 минут до hhmm."""
import sys
import types

import numpy as np
import pytest

from fastbt import exec_e2, exec_e_run
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

TK = 0.01
LN = np.log


def D(rows, t0=None):
    a = np.array(rows, float).reshape(-1, 4)
    t = None if t0 is None else t0 + 5 * np.arange(len(a))
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False, t)


def F(x, n=1):
    return [(x, x, x, x)] * n


def bars(n, **at):
    """n плоских баров по 100; at={индекс: (o, h, l, c)}."""
    r = F(100.0, n)
    for b, v in at.items():
        r[int(b[1:])] = v
    return r


def rn(days, sig, sigma=0.02, close=None, **kw):
    kw.setdefault("comm", 0.0)
    return run("T", days, sig, np.full(len(days), sigma), Params(**kw), close=close)


# --- время баров: заглушка Л-3 по контракту ---
def _bar_do(day, hhmm):
    m = (int(hhmm) // 100) * 60 + int(hhmm) % 100 - 5
    ix = np.flatnonzero(day.t == m)
    return int(ix[0]) if ix.size else None


def _plan(t_vyhod, prichina):
    if t_vyhod == "open2":
        return 1, None
    return (0 if prichina == "vremya" else 1), t_vyhod


@pytest.fixture(autouse=True)
def vremya(monkeypatch):
    m = types.ModuleType("fastbt.exec_e_vremya")
    m.bar_do, m.vyhod_plan = _bar_do, _plan
    monkeypatch.setitem(sys.modules, "fastbt.exec_e_vremya", m)
    monkeypatch.setattr("fastbt.exec_e_vremya", m, raising=False)
    return m


T0 = 870                                                      # 14:30: бар b кончается в 14:35 + 5b; 15:00 → b = 5; 17:00 → нет


def L0(k=0.5, s=0.02):
    return 100.0 * (1 - k * s)


# --- умолчания и ed ---
def _lenta(seed):
    r = np.random.default_rng(seed)
    days = []
    for _ in range(14):
        c = 100 * np.exp(np.cumsum(r.normal(0, 0.004, 8)))
        o = np.r_[c[0], c[:-1]] * (1 + r.normal(0, 0.001, 8))
        days.append(Day(o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c))
    return days, r.integers(0, 2, 14), r.uniform(0.01, 0.03, 14)


def _same(a, b):
    assert a.trades == b.trades and (a.n_orders, a.n_fills, a.x_t) == (b.n_orders, b.n_fills, b.x_t)
    for f in ("gross", "cost", "pos", "free", "fdr"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


def check_ed_equals_sigma_30():
    for seed in range(30):
        days, sig, sg = _lenta(seed)
        kw = dict(k=0.5, a=1.0, b=1.0, tau=3, comm=0.0004, f=("propusk", "zakrytie_d1", "povtor")[seed % 3])
        _same(run("T", days, sig, sg, Params(**kw), ed=sg), run("T", days, sig, sg, Params(**kw)))


def test_ed_equals_sigma_30():
    check_ed_equals_sigma_30()


def test_ed_changes_only_limit():
    days = [D(F(100.0)), D([(100, 100, 97.9, 100)]), D(F(100.0))]
    ed = np.array([0.04, 0.0, 0.0])
    r = run("T", days, [1, 1, 1], np.full(3, 0.02), Params(k=0.5, b=1.0, comm=0.0), ed=ed)
    assert r.trades[0].entry_px == pytest.approx(98.0)         # L = 100·(1 − 0.5·0.04); стоп 98·0.98 = 96.04 (по σ̂) не задет
    assert r.trades[0].reason is None
    with pytest.raises(ValueError):
        run("T", days, [1, 0, 0], np.full(3, 0.02), Params(k=0.5), ed=ed[:2])


def check_ed():
    test_ed_changes_only_limit()


def test_ed_mut(monkeypatch):
    monkeypatch.setattr(exec_e2.Run2, "_place", exec_e_run.Run._place)       # лимит по σ̂, а не по ed
    with pytest.raises(AssertionError):
        check_ed()


# --- zhizn ---
def _zh(b_low, zhizn):
    days = [D(F(100.0)), D(bars(20, **{f"b{b_low}": (100, 100, 98.5, 100)})), D(F(100.0))]
    return rn(days, [1, 0, 0], k=0.5, zhizn=zhizn).trades


def check_zhizn():
    assert len(_zh(11, 12)) == 1 and _zh(11, 12)[0].entry_bar == 11            # b < N
    assert _zh(12, 12) == [] and _zh(13, 12) == []                             # b ≥ N — заявка снята
    assert len(_zh(12, 36)) == 1 and len(_zh(15, None)) == 1


def test_zhizn():
    check_zhizn()


def test_zhizn_mut(monkeypatch):
    monkeypatch.setattr(exec_e2, "zhizn_hits", lambda hit, n: hit)             # жизнь не ограничена
    with pytest.raises(AssertionError):
        check_zhizn()
    monkeypatch.setattr(exec_e2, "zhizn_hits", lambda hit, n: hit if n is None else hit[hit <= n])   # b ≤ N
    with pytest.raises(AssertionError):
        check_zhizn()


def test_zhizn_no_fill_blocks_like_propusk():
    days = [D(F(100.0)), D(bars(20, b12=(100, 100, 98.5, 100))), D(F(100.0)), D(F(100.0))]
    assert rn(days, [1, 1, 1, 0], k=0.5, zhizn=12).trades == []


# --- gap ---
def _gp(o0, **kw):
    L = L0()
    days = [D(F(100.0)), D([(o0, 100, min(o0, L - 0.5), 100)]), D(F(100.0))]
    return rn(days, [1, 0, 0], k=0.5, tick=TK, **kw).trades


def check_gap():
    L = L0()
    assert _gp(L - TK, gap_propusk=True) == []                                 # o₀ ≤ L − tick → снята
    assert _gp(L - 1.0, gap_propusk=True) == []
    assert _gp(L - 1.0)[0].entry_px == L - 1.0                                 # без propusk — по open
    t = _gp(L, gap_propusk=True)                                               # o₀ = L: не за уровнем — по уровню
    assert len(t) == 1 and t[0].entry_px == L


def test_gap():
    check_gap()


def test_gap_mut(monkeypatch):
    monkeypatch.setattr(exec_e2, "gap_over", lambda *a: False)
    with pytest.raises(AssertionError):
        check_gap()


def test_gap_blocks_reentry():
    days = [D(F(100.0)), D([(97.0, 97.0, 96.0, 97.0)]), D(F(100.0)), D(F(100.0))]
    assert rn(days, [1, 1, 1, 0], k=0.5, tick=TK, gap_propusk=True).trades == []


# --- t_vyhod ---
def _tv(tv, d3, sig=(1, 1, 0, 0, 0), **kw):
    days = [D(F(100.0, 20), T0), D(F(100.0, 20), T0), D(F(100.0, 20), T0), D(d3, T0), D(F(100.0, 20), T0)]
    return rn(days, list(sig), t_vyhod=tv, comm=0.001, **kw)


D3 = bars(20, b5=(100, 101.5, 100, 101), b0=(100.5, 100.5, 100, 100), b19=(100, 100, 99, 99))


def check_t_vyhod_signal():
    r = _tv("close", D3)
    assert (r.trades[0].exit_day, r.x_t) == (2, 0)
    for tv, bar, px in (("1500", 5, 101.0), ("open2", 0, 100.5)):
        r = _tv(tv, D3)
        t = r.trades[0]
        assert (t.exit_day, t.exit_bar, t.exit_px, t.reason, r.x_t) == (3, bar, px, "signal", 1)
        assert r.gross[2] == pytest.approx(LN(px / 100.0)) and r.cost[2] == pytest.approx(0.001)   # выход — в начале интервала
    r = _tv("1700", D3)                                                         # бара 17:00 нет: close Н6 дня x+1, флаг прежний
    t = r.trades[0]
    assert (t.exit_day, t.exit_px, t.reason, r.x_t, r.cost[3]) == (3, 99.0, "signal", 0, pytest.approx(0.001))


def test_t_vyhod_signal():
    check_t_vyhod_signal()


def test_t_vyhod_mut_signal_same_day(monkeypatch, vremya):
    monkeypatch.setattr(vremya, "vyhod_plan", lambda tv, pr: (0, tv))            # выход по сигналу в 15:00 дня x
    with pytest.raises(AssertionError):
        check_t_vyhod_signal()


def test_t_vyhod_stop_first_before_bar_do():
    d3 = bars(20, b2=(100, 100, 97, 100), b5=(100, 101.5, 100, 101))
    r = _tv("1500", d3, b=1.0)
    t = r.trades[0]
    assert (t.exit_bar, t.exit_px, t.reason, r.x_t) == (2, 98.0, "stop", 0)


def check_stop_in_bar_do():
    d3 = bars(20, b5=(100, 100, 97, 99))                        # стоп 98 внутри бара 15:00, close бара 99
    t = _tv("1500", d3, b=1.0).trades[0]
    assert (t.exit_bar, t.exit_px, t.reason) == (5, 98.0, "stop")
    t = _tv("1500", bars(20, b5=(100, 100, 99, 99)), b=1.0).trades[0]     # стопа нет — выход по c[bar_do]
    assert (t.exit_bar, t.exit_px, t.reason) == (5, 99.0, "signal")


def test_stop_in_bar_do_first():
    check_stop_in_bar_do()


def test_stop_in_bar_do_mut(monkeypatch):
    monkeypatch.setattr(exec_e2, "scan_end", lambda cut: cut)     # bar_do исключён из скана стопа
    with pytest.raises(AssertionError):
        check_stop_in_bar_do()


def test_t_vyhod_tau_same_day():
    days = [D(F(100.0, 20), T0), D(F(100.0, 20), T0), D(bars(20, b5=(100, 100.5, 100, 100.4)), T0), D(F(100.0, 20), T0)]
    r = rn(days, [1, 1, 1, 1], t_vyhod="1500", tau=2, comm=0.001)
    t = r.trades[0]
    assert (t.exit_day, t.exit_bar, t.exit_px, t.reason, r.x_t) == (2, 5, 100.4, "vremya", 1)
    assert r.cost[1] == pytest.approx(0.001)
    r = rn(days, [1, 1, 1, 1], t_vyhod="1700", tau=2, comm=0.001)             # бара нет: close этого дня
    assert (r.trades[0].exit_day, r.x_t) == (2, 0)
    stp = days[:2] + [D(bars(20, b2=(100, 100, 97, 100)), T0)] + days[3:]
    t = rn(stp, [1, 1, 1, 1], t_vyhod="1500", tau=2, b=1.0).trades[0]         # стоп раньше среза
    assert (t.exit_bar, t.reason) == (2, "stop")


# --- bar_plus ---
def _bpl(o_next, kind="entry"):
    if kind == "entry":
        days = [D(F(100.0)), D([(100, 100, 100, 100), (100, 100, 98.5, 98.8), (o_next, 100, 98, 99), (99, 99, 99, 99)]), D(F(100.0))]
        return rn(days, [1, 0, 0], k=0.5, bar_plus=1).trades[0]
    days = [D(F(100.0)), D([(100, 100, 97, 98), (o_next, o_next, o_next, o_next)]), D(F(100.0))]
    return rn(days, [1, 1, 1], b=1.0, bar_plus=1).trades[0]


def check_bar_plus():
    L = L0()
    assert _bpl(99.5).entry_px == 99.5 and _bpl(98.0).entry_px == L           # лонг: хуже — выше
    assert _bpl(96.0, "stop").exit_px == 96.0 and _bpl(99.0, "stop").exit_px == 98.0
    days = [D(F(100.0)), D([(100, 100, 98.5, 98.8)]), D(F(100.0))]            # последний бар дня — close Н6
    assert rn(days, [1, 0, 0], k=0.5, bar_plus=1, close=[100, 98.8, 100]).trades[0].entry_px == 98.8
    t = rn(days, [1, 0, 0], k=0.5, bar_plus=0).trades[0]
    assert t.entry_px == L                                                    # 0 — как раньше


def test_bar_plus():
    check_bar_plus()


def test_bar_plus_short():
    days = [D(F(100.0)), D([(100, 103, 100, 100), (101.0, 101, 101, 101)]), D(F(100.0))]
    t = rn(days, [1, 1, 1], side=-1, b=1.0, bar_plus=1).trades[0]             # стоп шорта 102: open b+1 = 101 лучше → по уровню
    assert t.exit_px == 102.0
    days[1] = D([(100, 103, 100, 100), (104.0, 104, 104, 104)])
    assert rn(days, [1, 1, 1], side=-1, b=1.0, bar_plus=1).trades[0].exit_px == 104.0


def test_bar_plus_mut_same_bar(monkeypatch):
    monkeypatch.setattr(exec_e2.Run2, "_bp", lambda s, i, b, level, sell: (min if sell else max)(level, s.days[i].o[b]))
    with pytest.raises(AssertionError):
        check_bar_plus()
    monkeypatch.setattr(exec_e2.Run2, "_bp", lambda s, i, b, level, sell: level)       # задержки нет
    with pytest.raises(AssertionError):
        check_bar_plus()


# --- stop_tik ---
def _st(**kw):
    days = [D(F(100.0)), D([(100, 100, 96, 100)]), D(F(100.0))]
    return rn(days, [1, 1, 1], tick=TK, **kw).trades[0]


def _st_short(**kw):
    days = [D(F(100.0)), D([(100, 104, 99, 100)]), D(F(100.0))]
    return rn(days, [1, 1, 1], tick=TK, side=-1, **kw).trades[0]


def check_stop_tik():
    assert _st(b=1.0).exit_px == pytest.approx(98.0)
    assert _st(b=1.0, stop_tik=1).exit_px == pytest.approx(97.99)             # лонг: хуже — ниже
    assert _st_short(b=1.0, stop_tik=1).exit_px == pytest.approx(102.01)      # шорт — выше
    assert _st(b=1.0, stop_tik=2).exit_px == pytest.approx(97.98)


def test_stop_tik():
    check_stop_tik()


def test_stop_tik_mut(monkeypatch):
    monkeypatch.setattr(exec_e2, "tik_stop", lambda sd, S, tick, n: S + sd * n * tick)   # в лучшую сторону
    with pytest.raises(AssertionError):
        check_stop_tik()


# --- мутации № 2, 3 на Run2 ---
def check_touch():
    L = L0()
    days = [D(F(100.0)), D([(100, 100, L, 100)]), D(F(100.0))]
    assert rn(days, [1, 0, 0], k=0.5, zhizn=60).trades == []
    days[1] = D([(100, 100, L - 1e-9, 100)])
    assert rn(days, [1, 0, 0], k=0.5, zhizn=60).trades[0].entry_px == L


def test_touch_is_not_fill():
    check_touch()


def test_touch_mut(monkeypatch):
    touch = lambda sd, h, l, L, tick: (l <= L) if sd > 0 else (h >= L)
    monkeypatch.setattr(exec_e_run, "limit_mask", touch)
    monkeypatch.setattr(exec_e2, "limit_mask", touch)
    with pytest.raises(AssertionError):
        check_touch()


def check_stop_before_take():
    days = [D(F(100.0)), D([(100, 110, 90, 100)]), D(F(100.0))]
    t = rn(days, [1, 1, 1], a=1.0, b=1.0, zhizn=60).trades[0]
    assert (t.reason, t.exit_px) == ("stop", 98.0)


def test_stop_before_take():
    check_stop_before_take()


def test_stop_before_take_mut(monkeypatch):
    monkeypatch.setattr(exec_e_run, "stop_hit", lambda sd, o, h, l, S, touch: (False, S))    # стоп после тейка
    with pytest.raises(AssertionError):
        check_stop_before_take()


# --- проверки параметров ---
@pytest.mark.parametrize("kw", [dict(t_vyhod="1600"), dict(zhizn=24), dict(bar_plus=-1), dict(stop_tik=-1),
                                dict(stop_tik=1), dict(bar_plus=1, check="zakrytie")])
def test_check_params(kw):
    days = [D(F(100.0))] * 2
    with pytest.raises(ValueError):
        run("T", days, [1, 0], np.full(2, 0.02), Params(k=0.5, **kw))
