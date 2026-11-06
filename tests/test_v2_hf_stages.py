"""Ж-211 (GATES_HF G1): стадии bound / inter / weights / final. Синтетика, без обучения. Мутации: порча будущего ловится главным тестом обрезки; пороги режима по всей выборке;
N поправки Холма = число тестов; лимит trials; уровни 1/2 (выигрыш только 2017–2018 не даёт L2)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import pit
from models_v2.s6.e import rules
from models_v2.s7 import hf_bound as B, hf_cmp as C, hf_inter as I, hf_ledger as L, hf_levels as V, hf_rules as R, hf_wstage as W  # noqa: E741

N = 900


def synth(seed=7):
    rng = np.random.default_rng(seed)
    d = (np.datetime64("2017-03-01") + np.arange(N).astype("timedelta64[D]")).astype("datetime64[ns]")
    har = np.exp(rng.normal(0, 0.4, N))
    g = rng.normal(0, 0.2, N)
    return d, (d + 1).astype("datetime64[ns]"), har * np.exp(0.5 * g + rng.normal(0, 0.3, N)), har, g


def test_truncation_wf_zero_and_mutation_caught():
    d, td, rv, har, g = synth()
    ok = C.truncation_max_diff(B.wf_fn(R.cfg_grid()), rv, har, g, d, td, cuts=(400, 700))
    assert ok == 0.0
    cfgs = R.cfg_grid()
    bad = lambda rv_, h_, g_, d_, td_: R.wf_select(rv_, h_, g_, R.closed_of(rv_, d_, td_, extra=60), cfgs)[0]     # мутация: закрыто + 60
    assert C.truncation_max_diff(bad, rv, har, g, d, td, cuts=(400, 700)) > 0.0


def test_truncation_rowwise_clip_and_shrink_zero():
    d, td, rv, har, g = synth()
    for f in (lambda r, h, g_, d_, t_: R.apply(h, g_, 0.2, 1.0), lambda r, h, g_, d_, t_: R.apply(h, g_, None, 0.5)):
        assert C.truncation_max_diff(f, rv, har, g, d, td, cuts=(300, 600)) == 0.0


def test_weights_stage_truncation_zero_and_mutation_caught():
    d, td, rv, har, g = synth()
    S = np.column_stack([har, har * np.exp(g)])
    ctx = {"rv": rv, "d": d, "td": td, "st_g4": np.zeros(N, bool)}
    for kind, par in (("ema", 21), ("rate", 0.02)):
        assert W.trunc_diff(S, "roll_126", kind, par, ctx, cuts=(400, 700)) == 0.0
    base = W.sigma_of(S, rules.weights("roll_126", S, ctx, extra=80), "ema", 21)[0]          # мутация: целевой вес по закрытым + 80
    rv2, S2 = rv.copy(), S.copy()
    rv2[500:] *= 41.0
    S2[500:] *= 0.25
    bad = W.sigma_of(S2, rules.weights("roll_126", S2, {**ctx, "rv": rv2}, extra=80), "ema", 21)[0]
    assert np.abs(bad[:501] - base[:501]).max() > 0.0


def test_wald_detects_group_effect_and_null_is_flat():
    rng = np.random.default_rng(1)
    grp = rng.integers(0, 3, 1500)
    p_null = I.wald_nw(rng.normal(size=1500), grp)[1]
    p_eff = I.wald_nw(rng.normal(size=1500) + 0.5 * (grp == 2), grp)[1]
    assert p_eff < 1e-6 < 0.01 < p_null
    assert np.isnan(I.wald_nw(rng.normal(size=50), np.zeros(50, int))[1])        # одна группа — теста нет


def test_terciles_use_only_past_and_whole_sample_mutation_caught():
    rng = np.random.default_rng(3)
    x = rng.normal(size=800)
    x2 = x.copy()
    x2[500:] += 40.0
    a, b = pit.past_bucket(x, [1 / 3, 2 / 3]), pit.past_bucket(x2, [1 / 3, 2 / 3])
    assert np.array_equal(a[:500], b[:500], equal_nan=True)
    q = np.quantile(x[300:], [1 / 3, 2 / 3])                                       # мутация: пороги по всей выборке
    whole, whole2 = (x > q[1]).astype(int), (x2 > np.quantile(x2[300:], [1 / 3, 2 / 3])[1]).astype(int)
    assert not np.array_equal(whole[300:500], whole2[300:500])


def test_holm_with_n_final_stricter_than_number_of_tests():
    d, td, rv, har, g = synth()
    df = pd.DataFrame({"model": ["m"] * 3, "test": list("ABC"), "period": ["all"] * 3, "p_raw": [0.001, 0.02, 0.04]})
    a, b = V.add_holm(df, 477)["p_holm"].to_numpy(), V.add_holm(df, 3)["p_holm"].to_numpy()
    assert (a > b).all() and a[0] == pytest.approx(0.477)                          # мутация «N = число тестов» даёт 0,003
    with pytest.raises(ValueError):
        V.add_holm(df, 2)


def _tests_frame(delta_by_period: dict, p_raw=0.01, hi=-0.001):
    rows = []
    for t in "ABC":
        for per in V.SLICES:
            rows.append({"model": "M", "test": t, "period": per, "delta": delta_by_period.get(per, -0.01), "p_raw": p_raw, "p_holm": 0.05, "boot_hi95": hi})
    return pd.DataFrame(rows)


def test_levels_l2_and_concentrated_2017_2018_blocked():
    crit, q = {"HAR": 0.25, "XJ": 0.23, "M": 0.20}, {"HAR": 0.23, "M": 0.20}
    ok = V.levels(_tests_frame({}), "M", crit, q, "XJ")
    assert ok["L1"] and ok["L2"] and ok["verdict"] == "L2"
    conc = _tests_frame({"G_2019_2021": 0.001, "P3_2023_2025": 0.001, "late_2019_2026": 0.001})        # выигрыш только в P1 (2017–2018) и 2022
    r = V.levels(conc, "M", crit, q, "XJ")
    assert r["L1"] and not r["L2_d"] and not r["L2"] and r["verdict"] == "L1"
    raw_only = _tests_frame({})
    raw_only["p_holm"] = 0.5
    assert V.levels(raw_only, "M", crit, q, "XJ")["verdict"] == "L1 только сырой"


def test_trial_limit_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "ALL", tmp_path / "ALL.csv")
    monkeypatch.setattr(L, "CHECK", tmp_path / "CHECK.json")
    monkeypatch.setattr(L, "REP", tmp_path)
    for i in range(L.LIMITS["regime"]):
        L.register("regime", {"i": i})
    L.register("regime", {"i": 0})                                                 # повтор того же cfg — не новый trial
    assert L.used(L.load())["regime"] == L.LIMITS["regime"]
    with pytest.raises(ValueError):
        L.register("regime", {"i": 99})
