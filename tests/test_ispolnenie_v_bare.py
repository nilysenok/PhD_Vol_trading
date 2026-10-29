"""Инвариант исполнения (БЛОКЕР #596): цена любого исполнения — вход, стоп, тейк, лимит-выход, ½, добавка — лежит в [low; high]
своего бара (close дня — цена рынка); стоп, сработавший в баре входа, не закрывает позицию лучше цены входа.
Тест-свойство: 1000 случайных путей дневного автомата × режимы (σ, ed_s/ed_t, chand, chast, parab, лимит, стоп-заявка + k_e,
vyhod_lim + reentry_den) и 1000 сессий tf 1h (tf_ispolnenie_v5). Проверка — по барам в тесте, независимо от Run.vne.
Мутация: стоп в баре входа исполнен по уровню S (прежний код) — свойство и ValueError в sessiya её ловят. Синтетика, сеть не нужна.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fastbt import exec_e
from fastbt import tf_ispolnenie_v5 as X
from fastbt.exec_e_rules import stop_hit
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day, Params

EPS = 1e-9
STOPY = ("stop", "trail", "bu", "kanal")
N_PUT, N_DEN, N_BAR = 1000, 16, 6
REZHIMY = {
    "sigma": dict(k=0.25, b=0.3, a=2.0, stop_touch=True),
    "ed_atr_pct": dict(k=0.25, b=1.0, a=0.03, ed_s="atr", ed_t="pct", stop_touch=True),
    "chand": dict(k=0.25, b=2.0, a=2.0, chand=3.0, stop_touch=True),
    "chand_tight": dict(k=0.5, b=3.0, chand=1.0, stop_touch=True, reentry=False),
    "chast": dict(k=0.25, b=1.0, chast=(0.5, 1.0)),
    "parab": dict(k=0.25, b=2.0, parab=(0.02, 0.2), stop_touch=True),
    "vyhod_lim_re": dict(k=0.25, b=1.5, a=2.0, vyhod_lim=True, reentry_den=True, chand=2.0),
    "stop_zayavka_ke": dict(vhod_tip="stop_zayavka", b=0.3, k_e=-0.5, exit_signal=False, stop_touch=True),
}


def _put(rng):
    """Дни из N_BAR баров случайного блуждания (σ шага 0,4 %), сигнал −1/0/1 держится 2 дня; уровни стоп-заявок — экстремум дня."""
    x = np.log(100.0) + np.cumsum(rng.normal(0, 0.004, (N_DEN * N_BAR * 4 + 1)))
    days = []
    for d in range(N_DEN):
        seg = [x[(d * N_BAR + b) * 4:(d * N_BAR + b) * 4 + 5] for b in range(N_BAR)]
        days.append(Day(*(np.exp([f(s) for s in seg]) for f in (lambda s: s[0], np.max, np.min, lambda s: s[-1]))))
    sig = np.repeat(rng.integers(-1, 2, N_DEN // 2 + 1), 2)[:N_DEN].astype(float)
    hi, lo = np.array([d.h.max() for d in days]), np.array([d.l.min() for d in days])
    ur = dict(up=hi, dn=lo, xl=lo * 0.995, xs=hi * 1.005)
    return days, sig, np.full(N_DEN, 0.004 * np.sqrt(4 * N_BAR)), ur


def _narusheniya(trades, bar_of):
    """Исполнения вне [low; high] своего бара и стопы в баре входа лучше цены входа. bar_of(день, бар) -> (low, high, close дня)."""
    plohie = []
    for t in trades:
        for den, bar, px, chto in ((t.entry_day, t.entry_bar, t.entry_px, "vhod"), (t.exit_day, t.exit_bar, t.exit_px, t.reason)):
            if den is None or px is None:
                continue
            lo, hi, c = bar_of(den, bar)
            if px != c and not (lo - EPS * lo <= px <= hi + EPS * hi):
                plohie.append((chto, den, bar, px, lo, hi))
        if (t.parent is None and t.reason in STOPY and (t.exit_day, t.exit_bar) == (t.entry_day, t.entry_bar)
                and t.side * (t.exit_px - t.entry_px) > EPS * t.entry_px):
            plohie.append(("stop_luchshe_vhoda", t.entry_day, t.entry_bar, t.entry_px, t.exit_px, t.reason))
    return plohie


def _dnevnoy(rezhim, seed=0, n=N_PUT):
    rng, plohie, vse = np.random.default_rng(seed), [], 0
    p = Params(**REZHIMY[rezhim])
    for _ in range(n):
        days, sig, sg, ur = _put(rng)
        kw = dict(urovni=ur) if p.vhod_tip == "stop_zayavka" else {}
        r = exec_e.run("RW", days, sig, sg, p, **kw)
        bar_of = lambda den, bar: (days[den].l[min(bar, N_BAR - 1)], days[den].h[min(bar, N_BAR - 1)], days[den].c[-1])  # noqa: E731
        plohie += _narusheniya(r.trades, bar_of)
        vse += len(r.trades)
    return plohie, vse


@pytest.mark.parametrize("rezhim", sorted(REZHIMY))
def test_dnevnoy_avtomat_ispolneniya_v_bare(rezhim):
    plohie, vse = _dnevnoy(rezhim)
    assert vse > 300 and not plohie, (len(plohie), vse, plohie[:3])


def _sessii(seed, n=N_PUT, n_bar=9, sub=6):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x = np.log(100.0) + np.cumsum(rng.normal(0, 0.002, n_bar * sub + 1))
        seg = [x[k * sub:(k + 1) * sub + 1] for k in range(n_bar)]
        o, h, lo, c = (np.exp([f(s) for s in seg]) for f in (lambda s: s[0], np.max, np.min, lambda s: s[-1]))
        st = np.repeat(rng.integers(-1, 2, n_bar // 3 + 1), 3)[:n_bar].astype(np.int8)
        st[0] = 0
        yield o, h, lo, c, st, np.full(n_bar, 0.002 * np.sqrt(sub)), np.full(n_bar, float(np.mean(h - lo)))


@pytest.mark.parametrize("chand", [3.0, 1.0, None])
def test_tf_1h_ispolneniya_v_bare(chand):
    p, plohie, vse = dataclasses.replace(X.PARAMS, chand=chand), [], 0
    for o, h, lo, c, st, sg, atr in _sessii(11):
        x = X.sessiya("RW", o, h, lo, c, st, sg, p, atr)          # нарушение в коде — ValueError (инвариант sessiya)
        plohie += _narusheniya(x.trades, lambda den, bar: (lo[den], h[den], c[-1] if den == len(c) - 1 else np.nan))
        vse += len(x.trades)
    assert vse > 300 and not plohie, (len(plohie), vse, plohie[:3])


def _staryj_stop_vhoda(s, i, j, d, b, px, S):
    """Мутация (прежний код до #596): стоп в баре входа — по open бара и выход ровно по уровню S."""
    if S is None:
        return False
    hit = stop_hit(s.tr.side, d.o[b], d.h[b], d.l[b], S, s.p.stop_touch)[0]
    if hit:
        s._exit(i, j, b, S, s.sr)
    return hit


def test_mutaciya_stop_vhoda_po_urovnyu_lovitsya_svojstvom(monkeypatch):
    monkeypatch.setattr(Run, "_stop_vhoda", _staryj_stop_vhoda)
    try:
        plohie, vse = _dnevnoy("chand_tight", n=300)
    except ValueError as e:                                   # поймана кодом (#617): исполнение вне бара
        assert "вне [low; high]" in str(e), str(e)
        return
    assert plohie, "мутация «стоп в баре входа по S» не поймана ни свойством, ни ValueError"


def test_mutaciya_stop_vhoda_po_urovnyu_lovitsya_v_kode(monkeypatch):
    monkeypatch.setattr(Run, "_stop_vhoda", _staryj_stop_vhoda)
    with pytest.raises(ValueError, match="вне \\[low; high\\]"):
        for o, h, lo, c, st, sg, atr in _sessii(11, n=300):
            X.sessiya("RW", o, h, lo, c, st, sg, X.PARAMS, atr)


def test_stop_uzhe_probityj_v_bare_vhoda_vyhod_po_cene_vhoda():
    """Известный ответ: лимит 99,75 в баре 4 (low 99,5); Chandelier 107 ≥ цены входа — выход по 99,75, брутто 0."""
    o = np.array([110.0, 108.0, 104.0, 101.0, 100.0, 100.0])
    h, lo = np.array([110.0, 108.0, 104.0, 101.0, 100.5, 100.0]), np.array([108.0, 104.0, 101.0, 100.0, 99.5, 100.0])
    c = np.array([108.0, 104.0, 101.0, 100.0, 100.0, 100.0])
    x = X.sessiya("SYN", o, h, lo, c, np.array([0, 0, 0, 1, 1, 1], np.int8), np.full(6, 0.01), X.PARAMS, np.full(6, 1.0))
    t, = x.trades
    assert (t.entry_px, t.exit_px, t.reason, t.exit_day) == (99.75, 99.75, "trail", 4) and x.gross == 0.0
