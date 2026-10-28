"""Л-4: ось `filtr` (fastbt.filtr_e1, fastbt.filtr_e1_rynok). Синтетика с известным ответом, мутации «d+1» и «сторона», Д1.
python3 -m pytest tests/test_filtr_e1.py -q -s"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import e1_sektor, filtr_e1 as f, filtr_e1_rynok as fr, rezhim_e1

N, T, SIG = 300, "SBER", 0.01
DATES = pd.bdate_range("2016-01-04", periods=N).to_numpy("datetime64[D]")
Z0 = np.zeros(N)


def make(r=Z0, vol=None, w=None, imoex_r=Z0, rv5=None, ticker=T):
    c = 100 * np.exp(np.cumsum(r))
    w = np.full(N, 0.01) if w is None else np.asarray(w, float)
    td = SimpleNamespace(dates=DATES, close=c, high=c * (1 + w), low=c * (1 - w),
                         volume=np.ones(N) if vol is None else np.asarray(vol, float))
    im = pd.DataFrame({"date": pd.DatetimeIndex(DATES), "close": 3000 * np.exp(np.cumsum(imoex_r)), "sig_mkt": SIG, "nbars": 50})
    s = lambda x: {ticker: np.full(N, SIG) if x is None else np.asarray(x, float)}  # noqa: E731
    inp = SimpleNamespace(sk=SimpleNamespace(_td={ticker: (td, None)}), sigma={"rv5": s(rv5), "rv22": s(None), "rv66": s(None)},
                          sobytiya={"cbr": {ticker: np.arange(N) % 7 == 0}, "div": {ticker: np.arange(N) % 11 == 0}})
    return inp, im


def run(name, inp, im, sd, ticker=T, mp=None, shift=0):
    """shift=-1 — мутация «данные d+1»: все ряды тикера сдвинуты на -1 (день d видит d+1)."""
    mp.setattr(rezhim_e1, "imoex_daily", lambda path=None: im)
    if shift:
        td = inp.sk._td[ticker][0]
        for k in ("close", "high", "low", "volume"):
            setattr(td, k, np.roll(getattr(td, k), shift))
        for k in inp.sigma:
            inp.sigma[k][ticker] = np.roll(inp.sigma[k][ticker], shift)
    return fr.razresheno(name, ticker, inp, np.full(N, float(sd)), DATES)


@pytest.fixture
def mp(monkeypatch):
    return monkeypatch


def spike(k, v, base=0.0):
    x = np.full(N, base)
    x[k] = v
    return x


def scen(name):
    """(kwargs make, sd, день события): на honest в этот день True (кроме szhatie — там True до события)."""
    if name == "post_shok":
        return dict(r=spike(150, -0.035)), 1
    if name == "szhatie":
        return dict(w=np.where(np.arange(N) < 200, 0.05 - 0.00015 * np.arange(N), 0.2)), 1
    if name == "rv_perevorot":
        rv = np.full(N, SIG)
        rv[100], rv[101] = 0.02, 0.0105
        return dict(rv5=rv), 1
    if name == "obyom":
        return dict(vol=spike(100, 4.0, 1.0)), 1
    return dict(r=np.where((np.arange(N) >= 40) & (np.arange(N) < 45), 0.01, 0.0)), 1


NEW = ("post_shok", "szhatie", "rv_perevorot", "obyom", "rel_sila_imoex")


def test_00_d1_maksimum_na_sdvige_0(mp):
    """Д1: корреляция ряда с исходной дневной величиной при сдвигах -2…+1; окончательные ряды — по приращениям."""
    rng = np.random.default_rng(7)
    r, Rm = rng.normal(0, SIG, N), rng.normal(0, SIG, N)
    sig = SIG * np.exp(rng.normal(0, 0.2, N))
    vol, w = np.exp(rng.normal(0, 0.5, N)), 0.01 + 0.01 * rng.random(N)
    c = 100 * np.exp(np.cumsum(r))
    rg = f.rg_series(c * (1 + w), c * (1 - w), c)
    rv5, rv66 = sig * (1 + rng.random(N)), sig
    inc = lambda x, k=1: x - f.prev(x, k)  # noqa: E731
    rel = fr.rel_series(r, Rm, sig)
    cases = {"z": (f.z_series(r, sig), r), "V": (f.v_series(vol), vol), "U": (f.u_series(rv5, rv66), rv5),
             "Rg": (inc(rg), inc(2 * w, 20)), "rel": (inc(rel * f.prev(sig) * np.sqrt(5)), inc(r - Rm, 5))}
    res = {}
    for k, (x, y) in cases.items():
        res[k] = {s: float(pd.Series(x).corr(pd.Series(np.roll(y, -s)))) for s in (-2, -1, 0, 1)}
    print("Д1 корреляции " + "; ".join(f"{k}: " + " ".join(f"{s:+d}={v:.2f}" for s, v in d.items()) for k, d in res.items()))
    for k, d in res.items():
        assert max(d, key=d.get) == 0, (k, d)


def test_znacheniya_net_cbr_div(mp):
    inp, im = make()
    assert fr.ZNACHENIYA == ("net", "cbr", "div", "post_shok", "szhatie", "rv_perevorot", "obyom", "rel_sila_imoex", "sektor_trend")
    assert run("net", inp, im, 1, mp=mp).all()
    assert run("cbr", inp, im, 1, mp=mp).tobytes() == (~inp.sobytiya["cbr"][T]).tobytes()
    assert run("div", inp, im, 1, mp=mp).tobytes() == (~inp.sobytiya["div"][T]).tobytes()
    del inp.sobytiya["div"]
    assert run("div", inp, im, 1, mp=mp) is None            # даты не сверены → ne_schitana, не net
    with pytest.raises(ValueError):
        run("nope", inp, im, 1, mp=mp)


def test_otvety_post_shok_obe_storony(mp):
    inp, im = make(r=spike(150, -0.035))
    L, S = run("post_shok", inp, im, 1, mp=mp), run("post_shok", inp, im, -1, mp=mp)
    assert L[150:155].all() and not L[149] and not L[155] and L.sum() == 5 and not S.any()
    inp, im = make(r=spike(150, 0.035))
    assert run("post_shok", inp, im, -1, mp=mp)[150:155].all() and not run("post_shok", inp, im, 1, mp=mp).any()
    inp, im = make(r=spike(150, -0.035), imoex_r=spike(150, -0.03))     # |Z| = 3: шок рынка
    assert not run("post_shok", inp, im, 1, mp=mp).any()
    inp, im = make(r=spike(150, -0.035))
    assert not run("post_shok", inp, im, 0, mp=mp).any()               # sd = 0: нет стороны


def test_otvety_szhatie_rv_obyom_rel(mp):
    d = np.arange(N)
    inp, im = make(w=0.05 - 0.00015 * d)
    a = run("szhatie", inp, im, 1, mp=mp)
    assert a[139:].all() and not a[:138].any()                          # окно 120 значений Rg → с d = 20+119-1
    inp, im = make(w=0.01 + 0.00015 * d)
    assert not run("szhatie", inp, im, -1, mp=mp)[140:].any()
    kw, _ = scen("rv_perevorot")
    inp, im = make(**kw)
    a = run("rv_perevorot", inp, im, -1, mp=mp)
    assert not a[100] and a[101] and a[102] and a.sum() == 2
    kw, _ = scen("obyom")
    inp, im = make(**kw)
    a = run("obyom", inp, im, -1, mp=mp)
    assert a[100] and a.sum() == 1 and not a[:21].any()
    inp, im = make(r=spike(100, 0.01), vol=spike(100, 4.0, 1.0))
    assert not run("obyom", inp, im, 1, mp=mp)[100]                     # |z| = 1
    inp, im = make(**scen("rel_sila_imoex")[0])
    assert run("rel_sila_imoex", inp, im, 1, mp=mp)[44] and not run("rel_sila_imoex", inp, im, -1, mp=mp)[44]
    assert not run("rel_sila_imoex", inp, im, 1, mp=mp)[20]


@pytest.mark.parametrize("name", NEW)
def test_mutaciya_dannye_d_plus_1(name, mp):
    kw, sd = scen(name)
    ok = run(name, *make(**kw), sd, mp=mp)
    bad = run(name, *make(**kw), sd, mp=mp, shift=-1)
    assert ok.tobytes() != bad.tobytes()


@pytest.mark.parametrize("name", NEW)
def test_mutaciya_storona(name, mp):
    kw, sd = scen(name)
    a, b = run(name, *make(**kw), sd, mp=mp), run(name, *make(**kw), -sd, mp=mp)
    if name in ("post_shok", "rel_sila_imoex"):
        assert a.tobytes() != b.tobytes()                              # перепутанная сторона даёт другой ответ
    else:
        assert a.tobytes() == b.tobytes()                              # фильтр по контракту от стороны не зависит


def _sektor_inp(up_from, mp, ticker=T):
    inp, im = make(ticker=ticker)
    idx = pd.Series(np.where(np.arange(N) == up_from, 101.0, 100.0), index=pd.DatetimeIndex(DATES))
    cl = {ticker: pd.Series(inp.sk._td[ticker][0].close, index=pd.DatetimeIndex(DATES))}
    inp._filtr_sektor_ctx = e1_sektor.Ctx({"IDX": idx}, cl, {ticker: "IDX"})
    return inp, im


def test_sektor_trend_storony_i_d_plus_1(mp):
    inp, im = _sektor_inp(250, mp)
    L, S = run("sektor_trend", inp, im, 1, mp=mp), run("sektor_trend", inp, im, -1, mp=mp)
    assert L[250] and not L[249] and not L[:199].any() and S[251] and not S[250] and not S[:199].any()
    assert L.tobytes() != S.tobytes()                                   # сторона перепутана — ловится
    idx = inp._filtr_sektor_ctx.idx["IDX"]
    inp._filtr_sektor_ctx.idx["IDX"] = pd.Series(np.roll(idx.to_numpy(), -1), index=idx.index)    # мутация d+1
    assert run("sektor_trend", inp, im, 1, mp=mp)[249] and not run("sektor_trend", inp, im, 1, mp=mp)[250]


@pytest.mark.parametrize("tk", ["AFLT", "LSRG"])
def test_sektor_trend_aflt_lsrg_ne_dejstvuet(tk, mp):
    inp, im = make(ticker=tk)
    assert run("sektor_trend", inp, im, -1, ticker=tk, mp=mp).all()


@pytest.mark.parametrize("name", NEW)
def test_nechestnaya_istoriya_zapreshchena(name, mp):
    inp, im = make()
    a = run(name, inp, im, 1, mp=mp)
    assert not a[:5].any()
