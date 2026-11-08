"""V5, лист 1b, G16: моменты m_k, НУ-t, Вальд, правило п. 6.3. Известный ответ на синтетике, пустая лента, нулевая выборка, перепутанный знак."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_0_dx_ols as O, v5_dx1_mom as MO, v5_dx1_phi as PH


def synth(beta=0.0, n=800, seed=0, tick=("A", "B", "C")):
    """rv_tplus = F·(1 + beta·φ₁ + шум): m₁ = beta·E[φ₁²] ≈ beta; K8 = M = 1."""
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2017-01-02", periods=n)
    rows, ph = [], []
    for t in tick:
        phi = rng.normal(size=(n, 6))
        s = beta * phi[:, 0] + 0.3 * rng.normal(size=n)
        rows.append(pd.DataFrame({"ticker": t, "date": d, "t_plus": d + pd.offsets.BDay(1), "rv_tplus": np.maximum(1.0 + s, 0.05), "K8": 1.0, "M": 1.0}))
        ph.append(pd.DataFrame(phi, columns=PH.PHI).assign(ticker=t, date=d))
    return pd.concat(rows, ignore_index=True), pd.concat(ph, ignore_index=True)


def test_known_answer_moment_and_sign():
    c, ph = synth(beta=0.5)
    r = MO.wald(MO.date_series(c, ph, "K8"))
    assert abs(r["m"][1] - 0.5) < 0.06 and r["t"][1] > 5 and r["p"] < 1e-6
    c2, ph2 = c.copy(), ph.copy()
    ph2[PH.PHI] = -ph2[PH.PHI]                 # перепутанный знак φ -> знак m_k меняется
    assert MO.wald(MO.date_series(c2, ph2, "K8"))["m"][1] < -0.4


def test_null_not_rejected_and_t_matches_nw_ols():
    c, ph = synth(beta=0.0, seed=3)
    U = MO.date_series(c, ph, "K8")
    r = MO.wald(U)
    assert r["p"] > 0.01 and r["df"] == 7 and r["n"] == len(U)
    for k in range(7):
        t = O.nw_ols(U.iloc[:, k].to_numpy(), np.ones((len(U), 1)), U.index.to_numpy())["t"][0]
        assert np.isclose(r["t"][k], t, rtol=1e-9)


def test_wald_empty_and_short_series_are_nan():
    assert np.isnan(MO.wald(pd.DataFrame(np.zeros((0, 7)), columns=MO.NAMES))["p"])
    assert np.isnan(MO.wald(pd.DataFrame(np.ones((10, 7)), columns=MO.NAMES))["W"])


def test_rows_with_missing_phi_are_dropped_everywhere():
    c, ph = synth(beta=0.5)
    ph.loc[ph["date"] < "2018-06-01", "phi3"] = np.nan
    U = MO.date_series(c, ph, "K8")
    assert U.index.min() >= pd.Timestamp("2018-06-01")


def test_decision_rule_boundary():
    def tab(p):
        return pd.DataFrame([dict(base="K8", window="2017_2021", item="const", m=0.0, t_nw=0.1, p_wald=np.nan), dict(base="K8", window="2017_2021", item="phi2", m=0.1, t_nw=-3.0, p_wald=np.nan),
                             dict(base="K8", window="2017_2021", item="wald", m=np.nan, t_nw=np.nan, p_wald=p), dict(base="K8", window="2022_2025", item="wald", m=np.nan, t_nw=np.nan, p_wald=1e-9)])
    assert MO.decision(tab(0.009))["verdict"] == "считать" and MO.decision(tab(0.011))["verdict"] == "закрыть"
    assert MO.decision(tab(0.011))["big_phi"] == "phi2"                         # 2022–2025 в решение не входит (p = 1e-9 там)


def test_table_shape_and_mutation_window_year_is_caught():
    c, ph = synth(beta=0.5, n=2900, tick=("A", "B"))
    t = MO.table(c, ph)
    assert len(t) == 2 * 11 * 8 and set(t["base"]) == {"K8", "M"}
    g = t[(t["window"] == "2017_2021") & (t["base"] == "K8") & (t["item"] == "wald")]
    assert g["n_dates"].iloc[0] > 1200 and g["p_wald"].iloc[0] < 1e-6
