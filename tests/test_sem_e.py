"""Оценщик sem_e (C/E) под контракт sweep_evaluate v1.1: синтетика с известным ответом, без сети и реальных данных."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import cache, msp, sem_e
from fastbt.exec_e import Day, shift_krug
from fastbt.sem_e import Inputs, Pair, cache_key, evaluate, make_evaluate
from fastbt import sweep_drivers as sd

KEYS = ("v sharpe by_year sr_d T skew kurt msr sortino mdd oborot_y dni_v_rynke sdelok_y period okno_s okno_po "
        "chestnaya comm_pct excess_d").split()
OBJ = {"kind": "E", "sigma": "naive", "u": "all"}


def build(seed=0, tickers=("AAA", "BBB"), strength=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-03-01", "2019-12-31")
    n, pairs, sigma = len(dates), [], {"naive": {}, "const": {}}
    for t in tickers:
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
        days = []
        for x in c:
            b = np.array([[x, x * (1 + rng.uniform(0, .02)), x * (1 - rng.uniform(0, .04)), x * (1 + rng.normal(0, .004))]
                          for _ in range(4)])
            b[:, 1] = np.maximum(b[:, 1], b[:, [0, 3]].max(1))
            b[:, 2] = np.minimum(b[:, 2], b[:, [0, 3]].min(1))
            days.append(Day(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
        cl = np.array([d.c[-1] for d in days])
        lr = np.r_[np.diff(np.log(cl)), 0.0]
        sigma["naive"][t] = rng.uniform(0.01, 0.03, n)
        sigma["const"][t] = np.full(n, 0.02)
        for sid, cat in (("S1", 1), ("S3", 0)):
            pos = rng.choice([-1, 0, 1], n, p=[.15, .5, .35]).astype(float)
            pairs.append(Pair(t, sid, cat, dates, pos, lr, rng.random(n) if strength else None, days))
    rate = pd.Series([0.07], pd.to_datetime(["2017-01-01"]))
    return Inputs(pairs, sigma, rate, {2017: ["AAA"], 2018: ["BBB"]}, rvw={t: np.full(n, 0.02) for t in tickers})


def test_kontrakt_keys_and_values():
    inp = build()
    r = evaluate(OBJ, {"q": 0.5, "k": 0.5, "f": "zakrytie_d1"}, "tune2018", 0.04, inp)
    assert all(k in r for k in KEYS) and r["v"] == "1.1" and r["period"] == "tune2018"
    assert r["T"] == len(pd.bdate_range("2018-01-01", "2018-12-31")) == len(r["excess_d"])
    assert list(r["by_year"]) == [2018] and r["chestnaya"] == "net" and r["comm_pct"] == 0.04
    assert (r["okno_s"], r["okno_po"]) == ("2018-01-01", "2018-12-31") and np.isfinite(r["sharpe"])
    assert r["kurt"] > 1 and 0 <= r["mdd"] < 1 and 0 < r["dni_v_rynke"] <= 1 and r["sdelok_y"] > 0
    assert evaluate(OBJ, {}, "ctrl2019", 0.04, inp)["chestnaya"] == "kontrol"
    assert sorted(evaluate(OBJ, {}, "train", 0.04, inp)["by_year"]) == [2018, 2019]
    assert evaluate(OBJ, {}, "setup", 0.04, inp)["period"] == "tune2018"


def test_kontrakt_commission_costs_money():
    inp = build()
    a, b = (evaluate(OBJ, {"k": 0.5}, "train", c, inp) for c in (0.0, 0.1))
    assert b["sharpe"] < a["sharpe"] and sum(b["excess_d"]) < sum(a["excess_d"])


def test_drivers_oat_and_factorial_run_on_sem_e():
    inp = build()
    ev = make_evaluate(OBJ, inp, 0.04)
    base = {"q": 0.0, "k": 0.5, "f": "propusk", "a": 1.0, "b": 1.0, "tau": 3, "s": "fiks"}
    oat = sd.oat_sweep(ev, base, {"k": [0.25, 0.75], "tau": [1, 5]}, "train")
    assert oat["n_trials"] == 5 and np.isfinite([r["objective"] for r in oat["rows"]]).all()
    fac = sd.factorial_sweep(ev, base, {"k": [0.25, 0.75], "a": [1.0, 2.0]}, "tune2018")
    assert fac["n_trials"] == 4 and all("sharpe_2018" in r for r in fac["rows"])


def test_e0_q0_equals_a_on_long_side():
    inp = build()
    r = evaluate(OBJ, {"q": 0.0}, "train", 0.04, inp)
    rows = [pd.DataFrame(dict(date=p.dates, ticker=p.ticker, strategy=p.sid, position=(p.pos > 0) * 1.0,
                              daily_gross_return=(p.pos > 0) * p.lr)) for p in inp.pairs]
    ex = msp.portfolio(pd.concat(rows), inp.rate, 0.0004)["excess"]
    ex = ex[ex.index >= "2018-01-01"]
    assert np.allclose(r["excess_d"], ex.values, atol=1e-12, rtol=0)


def test_q_positive_needs_strength_and_changes_result():
    with pytest.raises(ValueError, match="нет силы"):
        evaluate(OBJ, {"q": 0.5}, "train", 0.04, build(strength=False))
    inp = build()
    a, b = (evaluate(OBJ, {"q": q}, "train", 0.04, inp) for q in (0.0, 0.67))
    assert a["excess_d"] != b["excess_d"] and b["dni_v_rynke"] < a["dni_v_rynke"]


def test_mask_cache_keyed_by_q():
    inp = build()
    a, b = (evaluate(OBJ, {"q": q}, "train", 0.04, inp)["dni_v_rynke"] for q in (0.5, 0.75))
    assert b < a                                     # разные q не делят одну закэшированную маску


def test_k2_lag0_equals_original_and_lag_changes_sharpe():
    inp = build()
    par = {"k": 0.5, "a": 1.0, "b": 1.0}

    def lagged(lag):
        s = {t: shift_krug(v, lag) for t, v in inp.sigma["naive"].items()}
        return dataclasses.replace(inp, sigma={**inp.sigma, "naive": s})
    r0 = evaluate(OBJ, par, "train", 0.04, inp)
    assert evaluate(OBJ, par, "train", 0.04, lagged(0))["excess_d"] == r0["excess_d"]
    assert evaluate(OBJ, par, "train", 0.04, lagged(37))["excess_d"] != r0["excess_d"]


def test_kljuch_cache_changes_with_every_input_and_code(monkeypatch):
    inp, par = build(), {"q": 0.5, "k": 0.5}
    k0 = cache_key(OBJ, par, inp, 0.04)
    assert k0 == cache_key(OBJ, dict(par), build(), 0.04)
    sg = {**inp.sigma, "naive": {t: v + 1e-6 for t, v in inp.sigma["naive"].items()}}
    assert cache_key(OBJ, par, dataclasses.replace(inp, sigma=sg), 0.04) != k0          # σ̂
    p0 = inp.pairs[0]
    d1 = list(p0.days)
    d1[5] = Day(d1[5].o, d1[5].h, d1[5].l - 1e-3, d1[5].c)
    for alt in (dataclasses.replace(p0, days=d1), dataclasses.replace(p0, strength=p0.strength + 1e-6),
                dataclasses.replace(p0, pos=p0.pos * 0)):                                # бары, сила, позиции
        assert cache_key(OBJ, par, dataclasses.replace(inp, pairs=[alt] + inp.pairs[1:]), 0.04) != k0
    assert cache_key({**OBJ, "sigma": "const"}, par, inp, 0.04) != k0
    assert cache_key({**OBJ, "u": "top10"}, par, inp, 0.04) != k0
    assert cache_key(OBJ, {**par, "q": 0.67}, inp, 0.04) != k0 and cache_key(OBJ, par, inp, 0.035) != k0
    monkeypatch.setattr(cache, "code_hash", lambda: "другой код")
    assert cache_key(OBJ, par, build(), 0.04) != k0                                      # код


def test_period_guards():
    inp = build()
    with pytest.raises(PermissionError, match="validation"):
        evaluate(OBJ, {}, "validation", 0.04, inp)
    with pytest.raises(ValueError, match="пустой"):      # открыт, но данных 2020–2021 в синтетике нет — не PermissionError
        evaluate(OBJ, {}, "validation", 0.04, dataclasses.replace(inp, validation_open=True))
    with pytest.raises(ValueError):
        evaluate(OBJ, {}, "test", 0.04, inp)
    late = dataclasses.replace(inp.pairs[0], dates=inp.pairs[0].dates + pd.Timedelta(days=800))
    with pytest.raises(PermissionError, match="2022"):
        evaluate(OBJ, {}, "train", 0.04, dataclasses.replace(inp, pairs=[late]))


def test_u_top10_uses_previous_year_turnover():
    inp = build()
    full = evaluate(OBJ, {}, "tune2018", 0.04, inp)
    top = evaluate({**OBJ, "u": "top10"}, {}, "tune2018", 0.04, inp)     # 2018: состав по обороту 2017 — только AAA
    assert top["dni_v_rynke"] < full["dni_v_rynke"]
    swap = dataclasses.replace(inp, top10={2017: ["BBB"], 2018: ["AAA"]})
    assert evaluate({**OBJ, "u": "top10"}, {}, "tune2018", 0.04, swap)["excess_d"] != top["excess_d"]
    cur = dataclasses.replace(inp, top10={2018: ["AAA"]})                # оборот текущего года не используется
    assert evaluate({**OBJ, "u": "top10"}, {}, "tune2018", 0.04, cur)["dni_v_rynke"] == 0.0


def test_rate_without_coverage_raises_not_bfill():
    inp = build()
    late = pd.Series([0.07], pd.to_datetime(["2018-06-01"]))
    with pytest.raises(ValueError, match="bfill"):
        evaluate(OBJ, {}, "tune2018", 0.04, dataclasses.replace(inp, rate=late))


def test_c_family_gates_and_star():
    inp = build()
    c = {"kind": "C", "sigma": "naive"}
    a = evaluate({"kind": "C", "sigma": "naive"}, {"W": 126, "p": 0.999999}, "train", 0.04, inp)
    b = evaluate(c, {"W": 126, "p": 0.5}, "train", 0.04, inp)
    d = evaluate(c, {"W": 126, "p1": 0.6, "p2": 0.8}, "train", 0.04, inp)
    e = evaluate(c, {"W": 126, "p": 0.5, "star": True}, "train", 0.04, inp)
    assert b["dni_v_rynke"] < a["dni_v_rynke"] and np.isfinite([x["sharpe"] for x in (a, b, d, e)]).all()
    assert d["excess_d"] != b["excess_d"] and e["T"] == a["T"]


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        evaluate({"kind": "Z", "sigma": "naive"}, {}, "train", 0.04, build())


def test_module_lines_within_limit():
    assert len(open(sem_e.__file__, encoding="utf-8").read().splitlines()) <= 200
