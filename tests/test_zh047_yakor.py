"""Ж-047, тесты 2.14 (синтетический вариант), 2.15, 2.17, 2.18 (контракт §2): якорь, port_rows ≡ port, вес на сделку, ключ.
Только синтетика. Допущения об интерфейсе (К2/К4 ещё не слиты; сверить при слиянии, XPASS/красный подскажет):
  exec_e.run_e1(ticker, days, sig, sigma, p, close=None, w=None) -> объект с .g, .sides, .expo (по строкам t);
  s9.Engine.port_rows(g, sides, expo, comm, cap=1); etap1.E1Inputs(...) с .digest (свойство или метод).
2.14 на реальных данных (SkEvaluator.pnl = ekviv_s9.run_sk, S1–S6, 2018/2019, 0,04/0,035 %) здесь НЕ написан:
реальные данные тестировщику запрещены — это тест К4 (`tests/test_etap1_yakor.py`)."""
import dataclasses
import importlib

import numpy as np
import pandas as pd
import pytest

from fastbt import exec_e
from test_zh047_pomoshch import D, F, P, SG, WAIT, wait

C = 0.0004
wait_port = pytest.mark.xfail(
    importlib.util.find_spec("fastbt.s9") is None or not hasattr(importlib.import_module("fastbt.s9").Engine, "port_rows"),
    strict=True, raises=WAIT, reason="ждёт К4: Engine.port_rows")


def engine(npair=4, n=12, seed=3):
    """Синтетический Engine: npair пар на одних n датах (одно окно на пару), без чтения данных."""
    from fastbt.s9 import Engine
    e = Engine.__new__(Engine)
    rng = np.random.default_rng(seed)
    e.r = {"date": np.tile(np.arange(n), npair), "pair": np.repeat(np.arange(npair), n)}
    e.first = np.r_[True, e.r["pair"][1:] != e.r["pair"][:-1]]
    e.lr_row = rng.normal(0, 0.02, n * npair)
    e.dates = pd.bdate_range("2019-01-02", periods=n)
    e.cnt = np.full(n, float(npair))
    e.kr = np.full(n, 0.07)
    e.rf = e.kr / 365.0
    e.cal = np.ones(n)
    return e, rng.integers(-1, 2, n * npair).astype(float)


def rows_from_pos(e, pos, comm):
    dp = np.diff(pos, prepend=0.0)
    dp[e.first] = pos[e.first]
    last = np.r_[e.first[1:], True]
    return pos * e.lr_row, np.abs(dp) + np.where(last, np.abs(pos), 0.0), np.abs(pos)


def sc_port_rows(e, pos):                                          # 2.15
    a = e.port(pos, C)
    g, sides, expo = rows_from_pos(e, pos, C)
    b = e.port_rows(g, sides, expo, C, 1.0)
    for col in a.columns:
        assert np.array_equal(a[col].values, b[col].values), f"port_rows ≠ port побайтно: {col}"


@wait_port
def test_215_port_rows_equals_port():
    e, pos = engine()
    sc_port_rows(e, pos)


@wait_port
@pytest.mark.parametrize("mut", ["comm_eps", "first_row_sides_zero", "sum_order"])
def test_215_mutations_caught(monkeypatch, mut):
    from fastbt.s9 import Engine
    e, pos = engine()
    orig = Engine.port_rows

    def bad(self, g, sides, expo, comm, cap=1.0):
        if mut == "comm_eps":
            comm = comm * (1 + 1e-12)
        if mut == "first_row_sides_zero":                           # 2.14: стороны первой строки окна → 0
            sides = np.where(self.first, 0.0, sides)
        if mut == "sum_order":                                      # другой порядок суммирования по датам
            o = np.argsort(-np.arange(len(g)))
            saved = self.r["date"]
            self.r = {**self.r, "date": saved[o]}
            try:
                return orig(self, g[o], sides[o], expo[o], comm, cap)
            finally:
                self.r = {**self.r, "date": saved}
        return orig(self, g, sides, expo, comm, cap)
    monkeypatch.setattr(Engine, "port_rows", bad)
    with pytest.raises(AssertionError):
        sc_port_rows(e, pos)


# --- 2.14 (синтетика): оси = якорь → g = pos·lr, expo = |pos|, стороны как в port ---
PX = np.array([100, 101, 99, 102, 103, 101, 104, 105, 103, 106, 107, 105, 108.0])
SIGP = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1.0])


def sc_anchor():
    days = [D(F(x)) for x in PX]
    r = exec_e.run_e1("T", days, SIGP, np.full(len(PX), SG), P(R=None, b=None, check="bary"))
    n = len(PX) - 1
    lr = np.log(PX[1:] / PX[:-1])
    pos = SIGP[:n]
    assert np.allclose(np.asarray(r.g)[:n], pos * lr, rtol=0, atol=1e-12), "g ≠ pos·lr"
    assert np.allclose(np.asarray(r.expo)[:n], np.abs(pos), rtol=0, atol=1e-12)
    dp = np.diff(pos, prepend=0.0)
    assert np.allclose(np.asarray(r.sides)[1:n - 1], np.abs(dp)[1:n - 1], rtol=0, atol=1e-12), "стороны якоря ≠ |Δpos|"


@wait(4)
def test_214_anchor_synthetic():
    sc_anchor()


@wait(4)
def test_214_mutation_exits_on_caught(monkeypatch):                # выходы включены при «якоре» → якорь рвётся
    orig = exec_e.Params
    monkeypatch.setattr(exec_e, "Params", lambda **kw: orig(**{**kw, "tau": 1}))
    with pytest.raises(AssertionError):
        sc_anchor()


# --- 2.17 вес фиксирован на сделку ---
def sc_weight():
    px = np.array([100, 101, 102, 103, 104, 100.0])
    days = [D(F(x)) for x in px]
    w = np.array([0.5, 0.9, 0.2, 0.7, 0.4, 0.4])                   # вес по дню сигнала d: σ меняется посреди сделки
    r = exec_e.run_e1("T", days, np.array([1, 1, 1, 1, 0, 0.0]), np.full(6, SG), P(R=None, b=None), w=w)
    lr = np.log(px[1:] / px[:-1])
    assert np.allclose(np.asarray(r.g)[:4], 0.5 * lr[:4], atol=1e-15), "вес сделки = вес дня входа"
    s = np.asarray(r.sides)
    assert (s[0], s[1], s[2], s[3], s[4]) == (0.5, 0, 0, 0, 0.5), "стороны: 2·w за сделку, без ребалансировок"


@wait(4)
def test_217_weight_fixed_per_trade():
    sc_weight()


@wait(4)
def test_217_mutation_daily_rebalance_caught(monkeypatch):
    orig = exec_e.run_e1

    def daily(*a, w=None, **kw):                                   # ежедневная перебалансировка: вес дня t в каждой строке
        r = orig(*a, w=None, **kw)
        r.g, r.expo = np.asarray(r.g) * w[: len(r.g)], np.asarray(r.expo) * w[: len(r.g)]
        r.sides = np.abs(np.diff(np.asarray(r.expo), prepend=0.0))
        return r
    monkeypatch.setattr(exec_e, "run_e1", daily)
    with pytest.raises(AssertionError):
        sc_weight()


# --- 2.18 ключ кэша (Д5, mutations.md № 7) ---
def inputs(et, **ch):
    day = D(F(100.0, 3))
    base = dict(sk=None, sigma={"T": {"rv5": np.ones(5)}}, bars={"T": [day]}, close_n6={"T": np.ones(5)},
                rezhim={"ma200": np.ones(5, bool)}, sobytiya={"cbr": {"T": np.zeros(5, bool)}},
                tick={"T": {2019: 0.01}}, rate=pd.Series([7.0], index=pd.to_datetime(["2019-01-01"])))
    base.update(ch)
    return et.E1Inputs(**base)


def dg(x):
    return x.digest() if callable(x.digest) else x.digest


def sc_key(et):
    k0 = dg(inputs(et))
    assert dg(inputs(et)) == k0, "ключ детерминирован"
    sub = {"sigma": {"T": {"rv5": np.ones(5) + 1e-9}}, "bars": {"T": [D(F(100.0, 2) + [(100, 100, 100, 100.01)])]},
           "sobytiya": {"cbr": {"T": np.r_[True, np.zeros(4, bool)]}}, "rezhim": {"ma200": np.zeros(5, bool)},
           "close_n6": {"T": np.ones(5) * 1.0000001}}
    for name, v in sub.items():
        assert dg(inputs(et, **{name: v})) != k0, f"подмена {name} при том же ключе"


@wait(4)
def test_218_cache_key_tracks_inputs():
    sc_key(importlib.import_module("fastbt.etap1"))


@wait(4)
def test_218_mutation_constant_key_caught(monkeypatch):
    et = importlib.import_module("fastbt.etap1")
    monkeypatch.setattr(et.E1Inputs, "digest", "const")
    with pytest.raises(AssertionError):
        sc_key(et)
