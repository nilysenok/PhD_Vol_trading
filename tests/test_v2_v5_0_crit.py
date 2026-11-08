"""V5-0, G2/G3: веса годов `crit_panel` — аналитическая формула; мутации: неверный штраф, неверный вес года, пропуски дат."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s6.xgb.d_comp import crit_wf
from models_v2.s12 import v5_0_crit as CR, v5_0_io as IO


def synth(seed=1, years=(2017, 2018, 2019, 2020, 2021), n=60):
    rng = np.random.default_rng(seed)
    d = pd.DatetimeIndex([pd.Timestamp("%d-03-01" % y) + pd.Timedelta(days=i) for y in years for i in range(n)])
    q = rng.gamma(2.0, 0.1, len(d)) * np.repeat(rng.uniform(0.5, 2.0, len(years)), n)
    return q, d


def test_weights_known_answer():
    w = CR.weights(5)
    assert np.allclose(w, np.array([25, 13, 7, 3, 0]) / 48.0, atol=1e-15) and abs(w.sum() - 1) < 1e-15


def test_formula_equals_crit_wf_on_synthetic():
    for seed in range(5):
        q, d = synth(seed)
        ann = pd.Series(q).groupby(d.year).mean().to_numpy()
        assert abs(CR.pen_parts(ann)[2] - crit_wf(q, d)) <= 1e-12


def test_base_part_is_weighted_mean_of_years_2021_weighs_zero():
    q, d = synth(3)
    ann = pd.Series(q).groupby(d.year).mean().to_numpy()
    base = CR.pen_parts(ann)[0]
    assert abs(base - float(CR.weights(5) @ ann)) <= 1e-12
    ann2 = ann.copy()
    ann2[-1] += 10.0                       # год 2021 не входит в критерий
    assert CR.pen_parts(ann2)[2] == CR.pen_parts(ann)[2]


def test_mutation_wrong_penalty_is_caught(monkeypatch):
    q, d = synth(2)
    ann = pd.Series(q).groupby(d.year).mean().to_numpy()
    monkeypatch.setattr(CR, "PENALTY", 0.5)
    assert abs(CR.pen_parts(ann)[2] - crit_wf(q, d)) > 1e-4


def test_mutation_equal_year_weights_is_caught():
    q, d = synth(4)
    ann = pd.Series(q).groupby(d.year).mean().to_numpy()
    assert abs(CR.pen_parts(ann)[0] - float(np.full(5, 0.2) @ ann)) > 1e-4


def test_gaps_in_dates_do_not_move_year_weights():
    """Пропуск половины дат года меняет годовое среднее, но не веса: формула и crit_wf остаются равны на урезанном ряду."""
    q, d = synth(5)
    keep = np.ones(len(d), bool)
    keep[(d.year == 2017) & (np.arange(len(d)) % 2 == 0)] = False
    ann = pd.Series(q[keep]).groupby(d[keep].year).mean().to_numpy()
    assert abs(CR.pen_parts(ann)[2] - crit_wf(q[keep], d[keep])) <= 1e-12


def test_ticker_with_one_year_excluded_in_annual():
    k = pd.DataFrame({"ticker": ["A"] * 4 + ["B"] * 4, "t_plus": pd.to_datetime(["2017-02-01", "2017-03-01", "2018-02-01", "2018-03-01"] * 2), "group": "M16",
                      "rv_tplus": np.arange(1, 9, dtype=float) / 3, "k1": 1.0, "k2": 1.0})
    k.loc[k["ticker"] == "B", "t_plus"] = pd.to_datetime(["2017-02-01", "2017-03-01", "2017-04-01", "2017-05-01"])
    a = CR.annual(k, ["k1"])
    assert set(a["ticker"]) == {"A"}


@pytest.mark.parametrize("seed", [0])
def test_real_panel_reproduces_crit_panel_and_weights(seed):
    k = IO.frame()
    cols = ["k1", "k2", "k3", "d2", "K8", "I2", "M"]
    tab, ver, _ = CR.audit_tables(k, cols)
    assert ver["max_abs_diff"] <= 1e-12 and ver["years_all"] and ver["kind"] == "определение"
    assert abs(ver["w2017"] - 25 / 48) < 1e-12 and ver["w2021"] == 0.0
