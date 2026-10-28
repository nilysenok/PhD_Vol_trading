"""Smoke быстрого слоя этапа 1 (Ж-083), синтетика, без данных и сети.
1) fast_etap1.run ≡ exec_e.run побайтно на случайных лентах и параметрах (стоп, тейк, трейл, безубыток, лимит,
   povtor / zakrytie_d1, vyhod_lim, тик None / 0 / шаг, дивиденды, дни pusto, close Н6) — известный ответ: эталон.
2) Подмена входа после digest — новый хэш (Д5): кэш digest не отдаёт старое значение.
Эквивалентность на реальных данных и мутации — у тестировщика (ветка skorost-test).
python3 -m pytest tests/test_fast_etap1_smoke.py -q"""
import dataclasses
import re

import numpy as np
import pytest

from fastbt import exec_e, fast_etap1
from fastbt.exec_e_types import Day, Params

TICK = 0.01


def _days(rng, n):
    out, px = [], 100.0
    for i in range(n):
        if i and rng.random() < 0.08:
            e = np.empty(0)
            out.append(Day(e, e, e, e, True))
            continue
        k = int(rng.integers(1, 7))
        c = np.round(px * np.exp(np.cumsum(rng.normal(0, 0.01, k))) / TICK) * TICK
        o = np.r_[np.round(px * np.exp(rng.normal(0, 0.01)) / TICK) * TICK, c[:-1]]
        h = np.maximum(o, c) + TICK * rng.integers(0, 30, k)
        lo = np.minimum(o, c) - TICK * rng.integers(0, 30, k)
        out.append(Day(o, h, lo, c, False))
        px = float(c[-1])
    return out


def _params(rng):
    ch = lambda *v: v[int(rng.integers(len(v)))]                       # noqa: E731
    b = ch(None, 1.0, 2.0)
    tau = ch(None, 1, 3, 5)
    R = ch(None, 1.5, 3.0) if b is not None else None
    k = ch(None, 0.25, 0.5, 1.0)
    check = ch("bary", "zakrytie")
    return Params(k=k, f=ch("propusk", "zakrytie_d1", "povtor"), a=None if R is not None else ch(None, 2.0),
                  b=b, tau=tau, s=ch("fiks", "ezhedn"), comm=0.0004, exit_signal=ch(True, False), side=ch(1, -1),
                  tick=ch(None, 0.0, TICK), stop_touch=ch(True, False), R=R, h=ch(None, 4.0) if R else None,
                  k_tr=ch(None, 1.5) if b is not None else None, be=ch(False, True) if b is not None else False,
                  check=check, reentry=ch(True, False), vyhod_lim=bool(k is not None and check == "bary" and ch(0, 1)))


def _case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 80))
    days = _days(rng, n)
    sig = np.repeat(rng.choice([-1, 0, 1], n // 3 + 1), 3)[:n].astype(float)
    sigma = rng.uniform(0.005, 0.03, n)
    sigma[rng.random(n) < 0.05] = np.nan
    close = np.array([d.c[-1] * (1 + rng.normal(0, 0.002)) if not d.pusto else 1.0 for d in days])
    div = np.where(rng.random(n) < 0.05, rng.uniform(0.5, 3.0, n), 0.0)
    return days, sig, sigma, _params(rng), (close if rng.random() < 0.7 else None), (div if rng.random() < 0.5 else None)


def _same(r1, r2):
    for f in ("gross", "cost", "pos", "free", "fdr"):
        assert getattr(r1, f).tobytes() == getattr(r2, f).tobytes(), f
    assert (r1.n_orders, r1.n_fills, r1.x_orders, r1.x_fills) == (r2.n_orders, r2.n_fills, r2.x_orders, r2.x_fills)
    assert [dataclasses.astuple(t) for t in r1.trades] == [dataclasses.astuple(t) for t in r2.trades]


@pytest.mark.parametrize("seed", range(400))
def test_fast_run_ravno_medlennomu(seed):
    days, sig, sigma, p, close, div = _case(seed)
    try:
        ref = exec_e.run("T", days, sig, sigma, p, close=close, div=div)
    except ValueError as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            fast_etap1.run("T", days, sig, sigma, p, close=close, div=div)
        return
    _same(ref, fast_etap1.run("T", days, sig, sigma, p, close=close, div=div))


@pytest.mark.parametrize("bad", ["bar_nan", "bar_pusto", "close_nan"])
def test_plohie_bary_ta_zhe_oshibka(bad):
    days, sig, sigma, _, close, _ = _case(3)
    close = np.array([d.c[-1] if not d.pusto else 1.0 for d in days])
    i = next(i for i, d in enumerate(days) if not d.pusto and i > 1)
    e = np.empty(0)
    if bad == "bar_nan":
        days[i] = Day(days[i].o, days[i].h, np.full(len(days[i].c), np.nan), days[i].c, False)
    elif bad == "bar_pusto":
        days[i] = Day(e, e, e, e, False)
    else:
        close[i] = np.nan
    with pytest.raises(ValueError) as ref:
        exec_e.run("T", days, sig, sigma, Params(), close=close)
    with pytest.raises(ValueError, match=re.escape(str(ref.value))):
        fast_etap1.run("T", days, sig, sigma, Params(), close=close)


@pytest.mark.parametrize("seed", range(40))
def test_osi_run2_idut_etalonom(seed):
    """Оси Л-2 (zhizn, gap_propusk, ed) ядро не повторяет: результат — тот же, что у exec_e.run (через эталон)."""
    days, sig, sigma, p, close, div = _case(seed)
    if p.check != "bary" or p.vyhod_lim or p.k is None or (p.R is not None and p.h is None and p.tau is None):
        p = dataclasses.replace(p, check="bary", vyhod_lim=False, k=0.5, R=None, h=None)
    p = dataclasses.replace(p, zhizn=12, gap_propusk=bool(seed % 2))
    _same(exec_e.run("T", days, sig, sigma, p, close=close, div=div),
          fast_etap1.run("T", days, sig, sigma, p, close=close, div=div))


def test_kesh_golovy_ne_putaet_lenty():
    """Тот же список дней с другим последним днём и другая лента той же длины — свой результат, как у эталона."""
    days, sig, sigma, p, close, div = _case(7)
    other, *_ = _case(8)
    for d in (days, days[:-1] + other[-1:], other[:len(days)]):
        if len(d) == len(days):
            _same(exec_e.run("T", d, sig, sigma, p), fast_etap1.run("T", d, sig, sigma, p))


def test_digest_podmena_vhoda_menyaet_hesh():
    from fastbt import etap1
    rng = np.random.default_rng(1)
    days = _days(rng, 10)
    inp = etap1.E1Inputs(None, {"rv22": {"T": rng.random(10)}}, {"T": days}, {"T": rng.random(10)}, {}, {}, {"T": TICK},
                         __import__("pandas").Series([1.0]), meta={"T": {"dates": np.arange(10)}}, div={"T": np.zeros(10)})
    with fast_etap1.patched():
        d0 = inp.digest()
        assert inp.digest() == d0
        inp.sigma["rv22"]["T"] = inp.sigma["rv22"]["T"] * 2
        d1 = inp.digest()
        inp.bars["T"][3] = days[4]
        d2 = inp.digest()
    assert len({d0, d1, d2}) == 3
    assert d2 == etap1.E1Inputs.digest(inp)
