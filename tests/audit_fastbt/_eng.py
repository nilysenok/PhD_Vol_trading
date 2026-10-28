"""Помощники аудита сделок (FA2a): вызов движков на синтетике и мутанты numba-функций.

mutant(fn, [(старое, новое, число_вхождений)]) — копия функции с правкой исходника, исполняемая
как обычный Python (без numba); исходники fastbt не меняются, мутант ставится через monkeypatch.
Тесты вызывают движки через атрибут модуля (engine.lvl_run, engine_trend.trend_run), поэтому
monkeypatch подменяет именно то, что проверяется. Каждая правка обязана найтись в исходнике
заданное число раз — иначе мутация была бы пустой.
"""
from __future__ import annotations

import inspect
import textwrap

import numpy as np

from fastbt import engine, engine_trend


def mutant(fn, repl, overrides=None):
    py = getattr(fn, "py_func", fn)
    src = textwrap.dedent(inspect.getsource(py))
    lines = src.splitlines()
    while lines and lines[0].lstrip().startswith("@"):
        lines.pop(0)
    src = "\n".join(lines) + "\n"
    for old, new, cnt in repl:
        got = src.count(old)
        assert got == cnt, f"мутация {py.__name__}: «{old}» найдено {got} раз, ждали {cnt}"
        src = src.replace(old, new)
    g = dict(py.__globals__)
    g.update(overrides or {})
    exec(compile(src, f"<мутант {py.__name__}>", "exec"), g)
    return g[py.__name__]


def bars(close, high=None, low=None):
    c = np.asarray(close, float)
    h = c + 0.5 if high is None else np.asarray(high, float)
    lo = c - 0.5 if low is None else np.asarray(low, float)
    return c, h, lo


def _sig(n, longs=(), shorts=()):
    s = np.zeros(n, np.int8)
    for t in longs:
        s[t] = 1
    for t in shorts:
        s[t] = -1
    return s


def _flags(n, idx=()):
    f = np.zeros(n, np.bool_)
    for t in idx:
        f[t] = True
    return f


def run_lvl(close, high, low, longs=(), shorts=(), exl=(), exs=(), kind=0, a1=1.0, a2=2.0,
            mh=100.0, be=-1.0, cd=0, tdec=0, adx_on=0, gamma=-1.0, med=0.0, atr=1.0,
            sigma=None, adx=None, sd=None, tdist=None, start=0, end=None):
    """Уровневый движок (контртренд/диапазон). Возвращает [(вход, выход, знак)]."""
    n = len(close)
    end = n if end is None else end
    sig = _sig(n, longs, shorts)
    nxt = engine.next_signal(sig)
    atr0 = np.full(n, float(atr)) if np.isscalar(atr) else np.asarray(atr, float)
    sg = np.full(n, 0.01) if sigma is None else np.asarray(sigma, float)
    ad = np.full(n, 10.0) if adx is None else np.asarray(adx, float)
    sdd = np.full(n, 1.0) if sd is None else np.asarray(sd, float)
    tdd = np.full(n, 2.0) if tdist is None else np.asarray(tdist, float)
    te, tx = np.empty(n, np.int64), np.empty(n, np.int64)
    td = np.empty(n, np.int8)
    nt = engine.lvl_run(sig, _flags(n, exl), _flags(n, exs), nxt, close, high, low, ad, atr0,
                        sg, sdd, sdd, tdd, tdd, int(kind), float(a1), float(a2), float(mh),
                        float(be), int(cd), int(tdec), int(adx_on), float(gamma), float(med),
                        int(start), int(end), te, tx, td)
    return [(int(te[i]), int(tx[i]), int(td[i])) for i in range(nt)]


def run_trend(close, high, low, longs=(), shorts=(), exl=(), exs=(), kind=0, isl=2.0, tmult=99.0,
              bth=1e12, tt=0, ps=0.02, pm=0.2, cd=0, base=30.0, gamma=-1.0, med=0.0, atr=1.0,
              sigma=None, rh=None, rl=None, start=0, end=None):
    """Трендовый движок. rh/rl по умолчанию — max high / min low за 11 баров с текущим."""
    import pandas as pd
    n = len(close)
    end = n if end is None else end
    sig = _sig(n, longs, shorts)
    nxt = engine.next_signal(sig)
    a = np.full(n, float(atr)) if np.isscalar(atr) else np.asarray(atr, float)
    a0 = np.where(np.isnan(a), 0.0, a)
    sg = np.full(n, 0.01) if sigma is None else np.asarray(sigma, float)
    if rh is None:
        rh = pd.Series(high).rolling(11, min_periods=1).max().values
    if rl is None:
        rl = pd.Series(low).rolling(11, min_periods=1).min().values
    te, tx = np.empty(n, np.int64), np.empty(n, np.int64)
    td = np.empty(n, np.int8)
    nt = engine_trend.trend_run(sig, _flags(n, exl), _flags(n, exs), nxt, close, high, low, a, a0,
                                sg, np.asarray(rh, float), np.asarray(rl, float), int(kind),
                                float(isl), float(tmult), float(bth), int(tt), float(ps),
                                float(pm), int(cd), float(base), float(gamma), float(med),
                                int(start), int(end), te, tx, td)
    return [(int(te[i]), int(tx[i]), int(td[i])) for i in range(nt)]


def pnl(trades, close):
    """Ручной расчёт: сделка d на [e, x) — вход и выход по close (спецификация п. 2)."""
    return [d * np.log(close[x] / close[e]) for e, x, d in trades]
