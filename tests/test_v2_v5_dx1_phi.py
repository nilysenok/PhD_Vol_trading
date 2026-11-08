"""V5, лист 1b, G16: φ₁…φ₆ известны на конец дня d; стандартизация — только по прошлому. Мутации: стандартизация по всей выборке; значения после дня D меняют φ до D."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_dx1_phi as PH


def toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2016-01-04", periods=n + 1)
    b, c = [], []
    for t in ("A", "B"):
        rv = rng.lognormal(0.0, 0.5, len(d))
        b.append(pd.DataFrame({"ticker": t, "date": d, "rv": rv, "j": rv * rng.uniform(0, 0.3, len(d))}))
        f = rv[:-1] * rng.lognormal(0.0, 0.2, n)
        c.append(pd.DataFrame({"ticker": t, "date": d[:-1], "t_plus": d[1:], "rv_tplus": rv[1:], "K8": f, "k1": f * 1.1, "M": f * 0.9}))
    return pd.concat(b, ignore_index=True), pd.concat(c, ignore_index=True)


def test_expanding_z_known_answer_and_skip():
    x = pd.Series(np.arange(100, dtype=float) % 7 + np.arange(100) * 0.01)
    z = PH.expanding_z(x)
    assert z.iloc[:60].isna().all() and z.iloc[60:].notna().all()
    p = x.iloc[:70]
    assert np.isclose(z.iloc[70], (x.iloc[70] - p.mean()) / p.std(ddof=1), atol=1e-12)


def test_expanding_z_ignores_future_and_whole_sample_mutation_caught():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=300))
    y = x.copy()
    y.iloc[200:] += 50.0
    assert np.allclose(PH.expanding_z(x).iloc[:200].dropna(), PH.expanding_z(y).iloc[:200].dropna(), atol=0, rtol=0)
    zx, zy = PH.expanding_z(x, whole=True), PH.expanding_z(y, whole=True)
    assert np.abs(zx.iloc[70:200] - zy.iloc[70:200]).max() > 0.1            # мутация «стандартизация по всей выборке» видна как зависимость от будущего


def test_daily_raw_known_answer():
    d = pd.bdate_range("2016-01-04", periods=300)
    rv = np.ones(len(d))
    rv[-1] = np.e ** 2
    b = pd.DataFrame({"ticker": "A", "date": d, "rv": rv, "j": 0.25 * rv})
    r = PH.daily_raw(b).set_index("date")
    assert np.isclose(r["x1"].iloc[-1], 2.0) and np.isclose(r["x2"].iloc[-1], 2.0)         # ln(e²) − среднее ln прошлых (0)
    assert np.isclose(r["x3"].iloc[-1], 0.25) and np.isnan(r["x1"].iloc[100])              # EWMA постоянной доли; нет 252 прошлых дней


def test_phi_table_uses_nothing_after_day_D_and_whole_sample_mutation_caught():
    b, c = toy()
    cut = pd.Timestamp("2017-12-29")
    full = PH.phi_table(b, c)
    b2, c2 = b.copy(), c.copy()
    b2.loc[b2["date"] > cut, ["rv", "j"]] *= 7.0
    c2.loc[c2["date"] >= cut, ["rv_tplus", "K8", "M", "k1"]] *= 5.0
    chg = PH.phi_table(b2, c2)
    a = full[full["date"] <= cut].reset_index(drop=True)
    e = chg[chg["date"] <= cut].reset_index(drop=True)
    assert a[PH.PHI].notna().any(axis=None)
    for p in PH.PHI:
        assert np.allclose(a[p], e[p], equal_nan=True, atol=0, rtol=0), p
    mut = PH.phi_table(b, c, whole=True)
    mut2 = PH.phi_table(b2, c2, whole=True)
    assert max(np.nanmax(np.abs(mut[p][mut["date"] <= cut] - mut2[p][mut2["date"] <= cut])) for p in PH.PHI) > 0.01


def test_phi_phi5_phi6_rows_start_after_skip():
    b, c = toy()
    t = PH.phi_table(b, c)
    g = t[t["ticker"] == "A"].reset_index(drop=True)
    first = g["phi5"].first_valid_index()
    assert first >= PH.MIN_OBS and g["phi5"].iloc[:first].isna().all()
