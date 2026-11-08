"""V4-I (G42): разложение RV = C + J и признаки компонент — известные ответы на синтетике, тождество, мутация «C := BV» (двойной счёт скачка, класс № 111), окна внутри сегмента. Данные проекта не читаются."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_i_comp as IC


def daily(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(1.0, 0.4, n))
    bv = rv * rng.uniform(0.6, 1.3, n)                       # bv > rv у части дней: J = 0 (как в daily.build_daily)
    j = np.maximum(rv - bv, 0.0)
    return pd.DataFrame({"rv": rv, "bv": bv, "j": j, "seg": 0}, index=pd.bdate_range("2017-01-02", periods=n))


def test_split_known_answer():
    assert np.array_equal(IC.split([4.0, 3.0], [1.0, 0.0]), [3.0, 3.0])


def test_identity_holds_for_battle_split_and_ulp_tolerance():
    d = daily(400, 1)
    c = IC.split(d["rv"], d["j"])
    assert IC.identity_ok(d["rv"], c, d["j"])
    assert (d["j"] > 0).any() and (d["j"] == 0).any()


def test_mutation_c_equals_bv_is_caught_by_identity():
    """Двойной счёт: C := BV вместо RV − J ломает C + J = RV там, где J > 0."""
    d = daily(400, 2)
    assert not IC.identity_ok(d["rv"], d["bv"], d["j"])
    assert not IC.identity_ok(d["rv"], d["bv"].to_numpy(), d["j"].to_numpy())


def test_mutation_c_equals_bv_via_battle_features(monkeypatch):
    """Подмена боевой функции разложения: признаки строятся на C := BV, тождество на тех же рядах не проходит."""
    d = daily(100, 3)
    monkeypatch.setattr(IC, "split", lambda rv, j: d["bv"].to_numpy())
    f = IC.daily_features(d)
    assert not IC.identity_ok(f["rv"], f["c"], f["j"])


def test_identity_catches_sign_swap_and_wrong_sum():
    d = daily(50, 4)
    assert not IC.identity_ok(d["rv"], IC.split(d["rv"], d["j"]) + 1e-6, d["j"])
    assert not IC.identity_ok(d["rv"], d["rv"] + d["j"], d["j"])


def test_features_known_answer_windows_event_ewma():
    d = daily(60, 5)
    f = IC.daily_features(d)
    c = d["rv"] - d["j"]
    assert np.isclose(f["lc"].iloc[10], np.log(c.iloc[10]))
    assert np.isclose(f["lcw"].iloc[10], np.log(c.iloc[6:11].mean()))
    assert np.isclose(f["lcm"].iloc[30], np.log(c.iloc[9:31].mean()))
    assert np.isnan(f["lcw"].iloc[3]) and np.isnan(f["lcm"].iloc[20])
    assert np.isclose(f["ewf"].iloc[0], float(d["j"].iloc[0] > 0))
    assert np.array_equal(f["n_ev"].to_numpy(), np.cumsum((d["j"] > 0).to_numpy()))


def test_windows_do_not_cross_segment_break():
    d = daily(60, 6)
    d["seg"] = (np.arange(60) >= 30).astype(int)
    f = IC.daily_features(d)
    assert np.isnan(f["lcw"].iloc[30:34]).all() and np.isfinite(f["lcw"].iloc[34])


def test_lead_shifts_features_to_next_day_and_targets_are_at_t_plus():
    d = daily(60, 7)
    f0, f1 = IC.daily_features(d), IC.daily_features(d, lead=1)
    assert np.isclose(f1["ewf"].iloc[10], f0["ewf"].iloc[11])
    P = pd.DataFrame({"ticker": "AAA", "date": d.index[20:30], "t_plus": d.index[21:31]})
    b = IC.build(P, {"AAA": d})
    assert np.array_equal(b["c_t"].to_numpy(), (d["rv"] - d["j"]).iloc[21:31].to_numpy()) and np.array_equal(b["j_t"].to_numpy(), d["j"].iloc[21:31].to_numpy())
    assert np.array_equal(b["ev_t"].to_numpy(), (d["j"].iloc[21:31] > 0).to_numpy(float))
