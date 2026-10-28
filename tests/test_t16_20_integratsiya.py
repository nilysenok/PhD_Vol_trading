"""Интеграция детекторов T16–T20 с драйвером `zh051_iter_t` (monkeypatch `vhod`): план строится, `porog` вызывается,
событие на пороге даёт вес с множителем 1. Мутация: порог = 1,0 вместо `porog(osi)` (у модуля нет `porog`) — вес меняется.
T17: порога в условии нет, `porog = 1,0` (базой веса — пробой на 1σ), мутация для него неразличима и не проверяется.
"""
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest

from fastbt import zh051_iter_t as D
from fastbt.t_obshchee import TARGET, postroit_sig

SG = 0.05                                     # σ̂ такая, что вес на пороге (0,25) ниже лимита L = 1
W_THR = TARGET / (SG * np.sqrt(252.0))        # вес события на пороге (множитель 1)
EPS = 1e-9


class _Stop(Exception):
    pass


class Tev:
    """Двойник TEvaluator: на первом вызове строит sig/вес как `postroit_sig` и останавливает прогон."""
    def __init__(self, n):
        self.n, self.seen = n, {}

    def rows(self, events, tau, porog, stop_b=None, dnevnoy=False, lag=1):
        evs = events.get("AAA", [])
        _, _, acc = postroit_sig(self.n, evs, tau, np.full(self.n, SG), porog, lag)
        self.seen = dict(events=evs, porog=porog, acc=acc)
        raise _Stop


def _t16(osi):
    c = np.full(60, 100.0)
    c[20] = 100.0 * np.exp(-(osi["z0"] + EPS) * SG)
    return dict(close=c, sigma=np.full(60, SG), mkt_close=np.full(60, 100.0), mkt_sigma=np.full(60, SG))


def _t17(osi):
    n = 100
    c, h, lo = np.full(n, 100.0), np.full(n, 120.0), np.full(n, 80.0)
    h[70:80], lo[70:80] = 100.5, 99.5
    c[80] = 100.5 * np.exp(SG * (1 + EPS))                                  # сила пробоя = 1σ
    return dict(close=c, high=h, low=lo, sigma=np.full(n, SG))


def _t19(osi):
    s5 = np.full(60, SG)
    s5[20] = SG * np.sqrt(osi["q"] + EPS)                                   # V_c = q + ε; на d=21 σ5 снижается
    return dict(sigma5=s5, sigma66=np.full(60, SG))


def _t20(osi):
    v = np.full(60, 100.0)
    v[25] = 100.0 * (osi["v0"] + EPS)
    c = np.full(60, 100.0)
    c[25] = 100.0 * np.exp(0.1 * SG)                                        # |z| = 0,1 < r0
    return dict(close=c, vol=v, sigma=np.full(60, SG))


CASES = [(16, "t16_shok", _t16), (17, "t17_szhatie", _t17), (19, "t19_rv_povorot", _t19), (20, "t20_obyom", _t20)]


def _run(it, mod, make, tmp):
    osi = D.plan(it)[0]["osi"]
    x = make(osi)
    n = len(next(iter(x.values())))
    inp = {"dates": pd.bdate_range("2018-01-02", periods=n).to_numpy("datetime64[D]"), "tickers": {"AAA": x}}
    mod.vhod = lambda: inp
    (tmp / "SHA256SUMS.txt").write_text("", encoding="utf-8")
    tev = Tev(n)
    with pytest.raises(_Stop):
        D.run_it(it, D.Ctx(tev, {"AAA": np.full(n, 2018)}, tmp), mod, out=tmp)
    return osi, tev.seen


@pytest.mark.parametrize("it,name,make", CASES)
def test_sobytie_na_poroge_dayot_ves_mnozhitel_1(it, name, make, tmp_path, monkeypatch):
    mod = importlib.import_module(f"fastbt.{name}")
    monkeypatch.setattr(mod, "vhod", mod.vhod)                              # откат подмены после теста
    osi, seen = _run(it, mod, make, tmp_path)
    assert seen["porog"] == mod.porog(osi) > 0
    assert len(seen["events"]) == 1                                         # одно событие на пороге
    assert seen["acc"][0][3] == pytest.approx(W_THR, rel=1e-6)              # множитель 1


@pytest.mark.parametrize("it,name,make", [c for c in CASES if c[0] != 17])
def test_mutatsiya_porog_1_vmesto_porog_osi_lovitsya(it, name, make, tmp_path, monkeypatch):
    mod = importlib.import_module(f"fastbt.{name}")
    monkeypatch.setattr(mod, "vhod", mod.vhod)
    monkeypatch.delattr(mod, "porog")                                       # драйвер берёт 1,0
    _, seen = _run(it, mod, make, tmp_path)
    assert seen["porog"] == 1.0
    assert seen["acc"][0][3] != pytest.approx(W_THR, rel=1e-6)


def test_t18_porog_bolshe_nulya():
    from fastbt import t18_div_gap
    assert t18_div_gap.porog({"g0": 0.0}) > 0 and t18_div_gap.porog({"g0": 2.0}) == 2.0


@pytest.mark.skipif(importlib.util.find_spec("fastbt.dividendy") is None, reason="fastbt/dividendy.py (#370) ещё не слит")
def test_t18_vhod_beret_tablitsu_iz_dividendy():
    from fastbt import dividendy, t18_div_gap
    assert dividendy.TABLE.exists() and callable(t18_div_gap.vhod)
