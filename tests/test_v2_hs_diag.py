"""Ж-212, этап A: модули диагностики (hs_diag, hs_pred, hs_diag8) на синтетике с известным ответом + подложенные ошибки (знак dq, шум вместо сигнала)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s8 import hs_diag, hs_diag8, hs_pit, hs_pred, hs_table


def make(n=900, plant=None, gamma_h=0.0, seed=11):
    """Синтетический o: истина f = XJ·exp(gamma_h·δ); plant — имя признака, который смещает ln HAR (HAR хуже при z>0)."""
    rng = np.random.default_rng(seed)
    d = (np.datetime64("2017-01-02", "D") + 4 * np.arange(n)).astype("datetime64[ns]")      # 2017…2026: периоды P1/P2/P3 непусты
    xj = np.exp(rng.normal(size=n) * 0.3)
    z = rng.normal(size=n)
    dl = rng.normal(size=n) * 0.3
    har = xj * np.exp(dl)
    f = xj * np.exp(gamma_h * dl)
    rv = f * rng.normal(size=n) ** 2
    if plant:
        rv = rv * np.exp(0.5 * z * (z > 0)) * 0 + rv                     # цель не трогаем: преимущество задаётся ниже
    df = pd.DataFrame({"d": d, "target_date": d + np.timedelta64(1, "D"), "rv": rv, "HAR": har, "XJ": xj, "LF": xj * np.exp(rng.normal(size=n) * 0.02), "st_g4": rng.random(n) < 0.2, "st_rv": False})
    o = hs_table.base_columns(df)
    if plant:                                                          # преимущество бустинга растёт с z: dq += 0.3·z
        o["dq_HAR_XJ"] = o["dq_HAR_XJ"] + 0.3 * z
    for c in ("ln_rv_d", "ln_rv_w", "ln_rv_m", "ln1p_J", "rs_asym"):
        o[c] = rng.normal(size=n)
    o["F_noise1"], o["F_f3_ret1"] = rng.normal(size=n), rng.normal(size=n)
    o["F_signal"] = z
    for c in ("mkt_rv", "mkt_rv_w", "mkt_rv_m"):
        o[c] = np.exp(rng.normal(size=n))
    o["ter_rv_w"], o["mkt_ter_w"] = rng.integers(0, 3, n), rng.integers(0, 3, n)
    o["rvi_regime"] = np.where(rng.random(n) < 0.5, "low", "high")
    o["past_dqX_m22"] = hs_pit.past_mean(o["dq_HAR_XJ"].to_numpy(), d, o["target_date"].to_numpy("datetime64[ns]"), 22)
    return o


def test_tnw_and_slice_row_known_answer():
    o = make()
    r = hs_diag.slice_row(o, np.ones(len(o), bool), "t", "all")
    assert r["n"] == len(o) and r["share_HAR_better"] == pytest.approx((o["dq_HAR_XJ"] < 0).mean())
    assert r["mean_r_XJ"] == pytest.approx((o["rv"] / o["XJ"]).mean()) and r["mean_dq_HAR_XJ"] == pytest.approx(o["q_HAR"].mean() - o["q_XJ"].mean())
    assert np.isnan(hs_diag.tnw(np.ones(10))) and np.isnan(hs_diag.tnw([]))


def test_when_table_partitions_and_mutation_dq_sign_caught():
    o = make()
    t = hs_diag.when_table(o)
    f = t[t["group"] == "factual"]
    assert f["n"].sum() == len(o) and f["share"].sum() == pytest.approx(1.0)
    flipped = o.copy()
    flipped["dq_HAR_XJ"] = -flipped["dq_HAR_XJ"]                         # подложенная ошибка: перепутан знак dq
    tf = hs_diag.when_table(flipped)
    assert not np.allclose(t["share_HAR_better"].to_numpy(float), tf["share_HAR_better"].to_numpy(float))


def test_slice_table_has_all_groups():
    t = hs_diag.slice_table(make())
    assert {"period", "year", "ter_rv_w_SBER", "ter_rv_w_IMOEX", "rvi_regime", "stress_g4", "ret_d", "rolling252"} <= set(t["group"])


def test_predictors_find_planted_signal_not_noise_and_sign_mutation_caught():
    o = make(plant=True)
    t = hs_pred.table(o).set_index("feature")
    assert t.loc["F_signal", "t_nw"] > 6 and t.loc["F_signal", "p_holm"] < 0.01 and t.loc["F_signal", "sign_same_in_2of3"]
    assert abs(t.loc["F_noise1", "t_nw"]) < 3.5 and not t.loc["F_noise1", "robust"]
    o2 = o.copy()
    o2["dq_HAR_XJ"] = -o2["dq_HAR_XJ"]
    assert hs_pred.table(o2).set_index("feature").loc["F_signal", "b"] < 0 < t.loc["F_signal", "b"]


def test_q7_slope_known_and_beta_profile_minimum_at_truth():
    o = make(gamma_h=0.0)
    q = hs_diag8.q7_table(o)
    assert (q["ml"] == "persistence").sum() == 1 and abs(q[(q.ml == "XJ") & (q.period == "all")]["b_eML_on_delta"].iloc[0]) < 0.6
    p = hs_diag8.beta_profile(o).query("ml == 'XJ'").set_index("beta")["qlike"]
    assert p.idxmin() in (0.0, 0.1) and p.loc[1.0] > p.loc[0.0]            # δ полностью HAR (β=1) хуже опоры, когда HAR не несёт информации


def test_q8_detects_planted_information_in_sample_and_oos():
    o = make(n=900, gamma_h=1.0)                                          # истина f = HAR: HAR несёт всю информацию сверх XJ
    t, y = hs_diag8.q8_tables(o)
    r = t[(t.base == "XJ") & (t.spec == "const+delta")].iloc[0]
    assert r["p_joint_in_sample"] < 1e-6 and r["delta_all"] < 0 and r["p_all"] < 0.05
    assert {"reverse: const+(-delta)"} <= set(t.spec) and len(set(y["year"])) >= 9
    t0, _ = hs_diag8.q8_tables(make(n=900, gamma_h=0.0))
    r0 = t0[(t0.base == "XJ") & (t0.spec == "const+delta")].iloc[0]
    assert r0["delta_all"] > -0.005                                       # без информации OOS-выигрыша нет
