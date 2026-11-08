"""V4-RQC (G35): K8 = sqrt(H·M), RQC1/RQC2 на известных ответах, тождество a ≡ 1, обрезка (PIT), мутации через боевой путь `v4_rqc_run.forecast`. Тест в сеть и к данным проекта не ходит."""
from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_rqc_calib as CAL, v4_rqc_run as R


def synth(n=320, tickers=("AAA", "BBB", "CCC"), seed=0, ratio=None, start="2017-01-02", sig=0.3):
    """Панель: rv = K8·ratio (ratio — число, функция (i, t) или шум lognormal)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n + 1)
    rows, zr = [], []
    for t in tickers:
        h, k2, k3 = (np.exp(rng.normal(-1.0, 0.2, n)) for _ in range(3))
        k8 = CAL.k8(h, k2, k3)
        r = np.exp(rng.normal(0.0, sig, n)) if ratio is None else np.array([ratio(i, d) if callable(ratio) else ratio for i, d in enumerate(dates[:n])], float)
        rows.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv": k8 * r, "h": h, "k2": k2, "k3": k3}))
        zr.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "jshare": rng.uniform(0.0, 0.3, n), "dlrv": rng.normal(0.0, 0.3, n)}))
    return pd.concat(rows, ignore_index=True), pd.concat(zr, ignore_index=True)


def test_k8_known_answer_and_bitwise():
    assert CAL.k8([4.0], [1.0], [1.0])[0] == 2.0
    h, k2, k3 = np.array([0.37, 1.9, 5.2]), np.array([0.5, 2.2, 3.1]), np.array([0.41, 1.1, 7.7])
    ref = [math.sqrt(a * math.sqrt(b * c)) for a, b, c in zip(h, k2, k3)]
    assert np.array_equal(CAL.k8(h, k2, k3), np.array(ref))


def test_rqc1_constant_ratio_known_answer():
    P, Z = synth(ratio=1.3)
    out = R.forecast("RQC1", P, Z)[0]
    d = out.sort_values(["ticker", "date"])
    a = d[d["ticker"] == "AAA"]["a"].to_numpy()
    assert np.all(a[:20] == 1.0) and np.allclose(a[20:], 1.3, rtol=1e-12)      # к дню r закрыто 3r пар; MIN_PAIRS = 60 -> a = 1 при r ≤ 19


def test_rqc1_clip_to_bounds():
    for ratio, want in ((3.0, 1.6), (0.1, 0.6)):
        P, Z = synth(ratio=ratio)
        assert R.forecast("RQC1", P, Z)[0]["a"].iloc[-1] == want


def test_rqc2_yearly_known_answer_uses_previous_year_only():
    P, Z = synth(n=800, ratio=lambda i, d: 1.2 if d.year == 2017 else (0.8 if d.year == 2018 else 1.0))
    o = R.forecast("RQC2", P, Z)[0]
    o["y"] = o["date"].dt.year
    assert (o[o["y"] == 2017]["a"] == 1.0).all()
    tol = dict(atol=5e-3)      # год пары — по t⁺: одна пара на границе года (дата в прошлом году) входит в соседний год
    assert np.allclose(o[o["y"] == 2018]["a"], 1.2, **tol) and np.allclose(o[o["y"] == 2019]["a"], 0.8, **tol) and np.allclose(o[o["y"] == 2020]["a"], 1.0, **tol)
    assert o[o["y"] == 2019]["a"].nunique() == 1 and o[o["y"] == 2019]["a"].iloc[0] != 0.8     # постоянна в году; граничная пара учтена по t⁺, а не по дате


def test_identity_a_equals_one_gives_k8_bitwise():
    P, Z = synth()
    k8 = R.forecast("K8", P, Z)[0]
    for cfg in ("RQC1", "RQC2", "RQC3"):
        o = R.forecast(cfg, P, Z, zero_a=True)[0]
        assert np.array_equal(o["F"].to_numpy(), k8["F"].to_numpy())
    assert np.array_equal(k8["F"].to_numpy(), k8["F0"].to_numpy())


@pytest.mark.parametrize("cfg", ["RQC1", "RQC3"])
def test_truncation_honest_and_mutation_closed_plus_one_is_caught(cfg):
    P, Z = synth()
    day = P["date"].iloc[200]
    assert not R.pit_violation(cfg, P, Z, day)
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):
        assert R.pit_violation(cfg, P, Z, day)


def test_mutation_closed_plus_one_changes_rqc2_at_year_boundary():
    """Для RQC2 «закрыто + 1» меняет год пар на границе года (пара декабря с t⁺ в январе): прогноз отличается от честного побитно."""
    P, Z = synth(n=800)
    honest = R.forecast("RQC2", P, Z)[0]["F"].to_numpy()
    with mock.patch.object(CAL, "closed_date", lambda tp, d: d):
        bad = R.forecast("RQC2", P, Z)[0]["F"].to_numpy()
    assert not np.array_equal(honest, bad)


def test_mutation_rqc2_year_y_instead_of_previous_is_caught():
    P, Z = synth(n=800)
    day = pd.Timestamp("2018-06-01")
    assert not R.pit_violation("RQC2", P, Z, day)
    with mock.patch.object(CAL, "ref_year", lambda y: y):
        assert R.pit_violation("RQC2", P, Z, day)


@pytest.mark.parametrize("cfg", ["RQC1", "RQC2", "RQC3"])
def test_loto_excluded_ticker_does_not_enter_panel(cfg):
    P, Z = synth(n=700)
    Q = P.copy()
    Q.loc[Q["ticker"] == "BBB", "rv"] *= 5.0
    a, b = R.forecast(cfg, P, Z, exclude="BBB")[0], R.forecast(cfg, Q, Z, exclude="BBB")[0]
    m = a["ticker"] != "BBB"
    assert np.array_equal(a.loc[m, "F"].to_numpy(), b.loc[m, "F"].to_numpy())
    c, d = R.forecast(cfg, P, Z)[0], R.forecast(cfg, Q, Z)[0]
    assert not np.array_equal(c.loc[c["ticker"] != "BBB", "F"].to_numpy(), d.loc[d["ticker"] != "BBB", "F"].to_numpy())      # контроль: без исключения влияет


def test_precomputed_pre_equals_inline():
    P, Z = synth()
    pre = R.make_pre(P, Z)
    assert np.array_equal(R.forecast("RQC3", P, Z, pre=pre)[0]["F"].to_numpy(), R.forecast("RQC3", P, Z)[0]["F"].to_numpy())
