"""V6-11a (Ж-132, И16): оси v5 «единица стопа/тейка» ed_s, ed_t ∈ {sigma, atr, pct} и Chandelier на стоп-заявке тренда (vhod_tip = stop_zayavka).
Известный ответ: плоские дни 0..14 (close 100, ATR(14) = 2, единица atr = 0,02), buy-stop 102 / sell-stop 98 вечером дня 14, бары дня 15–16.
«Выкл.» = v5 побайтно: слепки (sha256) посчитаны кодом базы до правки. Гэп через стоп — по open (мутация № 4); цена исполнения в [low; high].
Мутации: RunStop без рядов единиц; ряд единицы дня d+1; быстрое ядро берёт точку с новой осью на стоп-заявке (ловит маршрут в эталон).
python3 -m pytest tests/test_exec_e_stop_osi.py -q"""
import dataclasses
import hashlib

import numpy as np
import pytest

from fastbt import exec_e, exec_e_stop, fast_etap1, fast_exec
from fastbt.exec_e_rules import atr_series
from fastbt.exec_e_types import Day, Params

N0, NAN = 15, float("nan")


def _day(o, h, l, c):
    return Day(np.array(o, float), np.array(h, float), np.array(l, float), np.array(c, float), False)


def _flat():
    return _day([100, 100], [101, 101], [99, 99], [100, 100])


def P(**kw):
    return Params(**{"vhod_tip": "stop_zayavka", "stop_touch": True, "exit_signal": False, "comm": 0.0, **kw})


def _go(bars, p, up=102.0, dn=98.0, f=exec_e.run):
    days = [_flat() for _ in range(N0)] + [_day(*zip(*b)) for b in bars]
    n = len(days)
    u = {k: np.full(n, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][N0 - 1], u["dn"][N0 - 1] = up, dn
    return f("T", days, np.zeros(n), np.full(n, 0.01), p, urovni=u), days


def _t(bars, p, **kw):
    return _go(bars, p, **kw)[0].trades[0]


LONG = [(100, 103, 100, 102.5)]                                   # день 15: buy-stop 102 исполнен по 102
SHORT = [(100, 100, 97.5, 98)]                                    # день 15: sell-stop 98 исполнен по 98


@pytest.mark.parametrize("p,low,px", [
    (P(b=0.03, ed_s="pct"), 98.0, 102 * 0.97),                    # S = 98,94
    (P(b=2.0, ed_s="atr"), 97.0, 102 * (1 - 2 * 0.02)),           # S = 97,92 (σ-стоп был бы 99,96)
    (P(b=2.0), 97.0, 102 * (1 - 2 * 0.01)),                       # единица σ: путь прежний
])
def test_stop_v_edinitsakh_long(p, low, px):
    t = _t([LONG, [(102, 102, low, 99)]], p)
    assert (t.side, t.reason, t.exit_day) == (1, "stop", N0 + 1) and t.exit_px == pytest.approx(px, abs=1e-9)


def test_stop_pct_short_zerkalo():
    t = _t([SHORT, [(98, 101.5, 98, 101)]], P(b=0.03, ed_s="pct"))
    assert (t.side, t.reason) == (-1, "stop") and t.exit_px == pytest.approx(98 * 1.03, abs=1e-9)


@pytest.mark.parametrize("p,px", [(P(b=5.0, a=0.03, ed_t="pct"), 102 * 1.03), (P(b=5.0, a=3.0, ed_t="atr"), 102 * (1 + 3 * 0.02))])
def test_take_v_edinitsakh(p, px):
    t = _t([LONG, [(102, 109, 101, 105)]], p)
    assert t.reason == "take" and t.exit_px == pytest.approx(px, abs=1e-9)


def test_gep_cherez_stop_po_open():
    """Мутация № 4: гэп вниз через стоп 98,94 — исполнение по open 97 (худшая из open и уровня), а не по уровню."""
    t = _t([LONG, [(97, 98, 96, 97)]], P(b=0.03, ed_s="pct"))
    assert (t.reason, t.exit_px) == ("stop", 97.0)


def test_chandelier_na_stop_zayavke():
    bars = [[(100, 110, 100, 110)], [(108, 108, 100, 100)]]
    p = P(b=5.0, ed_s="atr", chand=2.0)                          # начальный стоп 91,8; Chandelier: максимум 11 дней до дня 16 − 2·ATR
    t, days = _go(bars, p)[0].trades[0], _go(bars, p)[1]
    assert (t.reason, t.exit_day) == ("trail", N0 + 1)
    assert t.exit_px == pytest.approx(110.0 - 2.0 * atr_series(days)[N0], abs=1e-9)
    assert _t(bars, dataclasses.replace(p, chand=None)).reason != "trail"


def test_ed_pustoy_atr_stopa_net():
    """Нет ATR (меньше 14 дней) → единица nan → уровня стопа нет; вход есть (как у Run2 с ed_s = atr)."""
    days = [_flat() for _ in range(5)] + [_day(*zip(*LONG))]
    u = {k: np.full(6, NAN) for k in ("up", "dn", "xl", "xs")}
    u["up"][4] = 102.0
    r = exec_e.run("T", days, np.zeros(6), np.full(6, 0.01), P(b=2.0, ed_s="atr"), urovni=u)
    assert r.trades[0].reason is None


def _rnd(seed, n=90):
    g, c, B = np.random.default_rng(seed), 100.0, []
    for _ in range(n):
        cl = c * (1 + g.normal(0, 0.012, 4).cumsum() / 2)
        op = np.r_[c, cl[:-1]]
        B.append(_day(op, np.maximum(op, cl) * (1 + abs(g.normal(0, 0.004, 4))), np.minimum(op, cl) * (1 - abs(g.normal(0, 0.004, 4))), cl))
        c = cl[-1]
    h, l = np.array([max(d.h) for d in B]), np.array([min(d.l) for d in B])
    roll = lambda f, a, w: np.array([f(a[max(0, i - w + 1):i + 1]) for i in range(n)])      # noqa: E731
    u = dict(up=roll(np.max, h, 5), dn=roll(np.min, l, 5), xl=roll(np.min, l, 3), xs=roll(np.max, h, 3))
    for v in u.values():
        v[:20] = NAN
    return B, u


def _digest(r):
    return hashlib.sha256(r.gross.tobytes() + r.cost.tobytes() + r.pos.tobytes() + repr(r.trades).encode()).hexdigest()[:16]


PV5 = (P(b=1.5, a=2.0, tau=6), P(b=2.0, atr_m=2.0, chast=(0.5, 1.0), k_e=0.25, comm=0.0004))
SLEPOK = {(0, 0): "635cd755dd86c9ef", (0, 1): "db07f32081553add", (1, 0): "46d29c347e609c4b", (1, 1): "6714ccc0f5a8c6e8"}     # заполнено кодом базы до правки


@pytest.mark.parametrize("seed,k", sorted(SLEPOK))
def test_vykl_ravno_v5_pobaytno(seed, k):
    B, u = _rnd(seed)
    r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), PV5[k], urovni=u)
    assert r.trades and _digest(r) == SLEPOK[(seed, k)]


def test_tsena_ispolneniya_v_low_high():
    """Инвариант [low; high]: все новые оси сразу, 8 путей; цены входа/выхода внутри бара (кроме close дня)."""
    p = P(b=2.0, a=3.0, ed_s="atr", ed_t="atr", chand=1.5, k_e=0.1, comm=0.0004)
    n = 0
    for seed in range(8):
        B, u = _rnd(seed)
        r = exec_e.run("T", B, np.zeros(len(B)), np.full(len(B), 0.012), p, urovni=u)
        for t in r.trades:
            for dy, br, px in ((t.entry_day, t.entry_bar, t.entry_px), (t.exit_day, t.exit_bar, t.exit_px)):
                if dy is not None and px != B[dy].c[-1]:
                    assert B[dy].l[br] - 1e-9 <= px <= B[dy].h[br] + 1e-9
                    n += 1
    assert n > 20


def test_mutaciya_runstop_bez_ryadov_edinits_lovitsya(monkeypatch):
    bars = [LONG, [(102, 102, 97, 99)]]
    p = P(b=2.0, ed_s="atr")
    ok = _t(bars, p).exit_px
    orig = exec_e_stop.RunStop.__init__
    monkeypatch.setattr(exec_e_stop.RunStop, "__init__", lambda s, *a, **kw: orig(s, *a, **{k: v for k, v in kw.items() if k not in ("us", "ut")}))
    assert _t(bars, p).exit_px != ok


def test_mutaciya_edinitsa_dnya_d_plyus_1_lovitsya(monkeypatch):
    bars = [LONG, [(102, 102, 97, 99)]]
    p = P(b=2.0, ed_s="atr")
    ok = _t(bars, p).exit_px
    orig = exec_e.unit_series
    monkeypatch.setattr(exec_e, "unit_series", lambda *a: (lambda x: None if x is None else np.r_[x[1:], 0.5])(orig(*a)))
    assert _t(bars, p).exit_px != ok


def test_novaya_os_na_stop_zayavke_idet_yadrom():
    """Ф-vhod_open: ed_s / chand на стоп-заявке считает ядро (прежде — эталон); побайтно равно эталону."""
    p = P(b=2.0, ed_s="atr", chand=1.5)
    assert fast_exec.ne_znaet(p) == [] and fast_etap1._yadro_v5(p, None, None, None)
    bars = [LONG, [(102, 102, 97, 99)]]
    assert _go(bars, p, f=fast_etap1.run)[0].trades == _go(bars, p)[0].trades
    assert fast_exec.ne_znaet(P(b=2.0, a=2.0)) == []                  # у прежних точек маршрут прежний


def test_T_i_smena_na_stop_zayavke_ne_prinimayutsya():
    for bad in (dict(T=3), dict(exit_signal=True)):
        with pytest.raises(ValueError):
            _go([LONG], dataclasses.replace(P(b=2.0), **bad))
