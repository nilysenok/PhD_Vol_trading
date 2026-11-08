"""V5 лист 1a, G11 (1), (2), (7), (8), (9), (10): тождества F = K8 / F = H, положительность, аналитический градиент против численного, общие пары, сдвиг K8 на +1 день, SHA замороженных входов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s10 import hp_stats_sel as S
from models_v2.s11 import v4_audit as AU
from models_v2.s12 import v5_0_ctrl as CT, v5_gs_core as CO, v5_gs_fit as FI, v5_gs_panel as PA

TICKS = ["AFLT", "ALRS", "HYDR", "IRAO", "LKOH"]


def synth(n: int = 700, seed: int = 1) -> pd.DataFrame:
    """Синтетическая лента: общий и собственный множители, прогнозы K1–K3 с шумом; d = предыдущий торговый день, τ = следующий."""
    rng = np.random.default_rng(seed)
    ax = pd.bdate_range("2017-01-02", periods=n)
    d0 = pd.bdate_range(end=ax[0], periods=2)[0]
    f = np.cumsum(rng.normal(0, 0.05, n)) * 0.3
    rows = []
    for t in TICKS:
        lr = f + np.cumsum(rng.normal(0, 0.03, n)) * 0.3 + rng.normal(0, 0.4, n)
        rv = np.exp(lr)
        k1, k2, k3 = (np.exp(pd.Series(lr).rolling(5, min_periods=1).mean().shift(1).bfill().to_numpy() + rng.normal(0, s, n)) for s in (0.25, 0.2, 0.2))
        dd = pd.DatetimeIndex([d0]).append(ax[:-1])
        rows.append(pd.DataFrame({"ticker": t, "date": dd, "t_plus": ax, "group": "M16", "rv_tplus": rv, "k1": k1, "k2": k2, "k3": k3}))
    return pd.concat(rows, ignore_index=True)


def test_identity_k8_when_alpha_beta_zero_and_h_when_g0_zero():
    P = PA.build(synth(), TICKS)
    for th in ([0.9, 0.0, 0.0], [0.3, 0.0, 0.0]):
        U = CO.run(P.rv, P.k8, P.pres, th)[0]
        assert np.array_equal((P.k8 * np.exp(U))[P.pres], P.k8[P.pres])                      # F = K8 побитно
        assert np.array_equal((P.h * np.exp(U))[P.pres], P.h[P.pres])                        # g0 = 0: основа H, F = H побитно
    old = FI.MIN_DAYS
    FI.MIN_DAYS = 10 ** 9                                                                    # θ = θ₀ всегда
    try:
        U = FI.run_config(P, P.k8, "B", False)["U"]
        assert np.array_equal((P.k8 * np.exp(U))[P.pres], P.k8[P.pres])
    finally:
        FI.MIN_DAYS = old


def test_forecasts_positive_finite_all_variants():
    P = PA.build(synth(300), TICKS)
    for var, clip in (("A", False), ("A", True), ("B", True), ("C", False), ("D", True)):
        U = FI.run_config(P, P.k8, var, clip)["U"]
        F = (P.k8 * np.exp(U))[P.pres]
        assert np.isfinite(F).all() and (F > 0).all()


def _num_grad(P, th, W, c, clip, n):
    g = np.zeros(3)
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6
        g[j] = (CO.run(P.rv, P.k8, P.pres, th + e, W, c, clip, n_use=n)[3][0] - CO.run(P.rv, P.k8, P.pres, th - e, W, c, clip, n_use=n)[3][0]) / 2e-6
    return g


def test_analytic_gradient_matches_numeric_incl_clip_and_graph_weights():
    P = PA.build(synth(300), TICKS)
    m = len(TICKS)
    W = np.abs(np.random.default_rng(3).normal(size=(m, m))) * (1 - np.eye(m))
    c = np.linspace(0.8, 1.2, m)
    th = np.array([0.8, 0.12, 0.07])
    for clip in (False, True):
        a = CO.run(P.rv, P.k8, P.pres, th, W, c, clip, n_use=250)[3][2:5]
        assert np.max(np.abs(a - _num_grad(P, th, W, c, clip, 250)) / np.abs(a)) <= 1e-6


def test_score_is_minus_gradient_of_qlike_in_ln_f():
    rv, f, e = 2.3, 1.7, 1e-6
    q = lambda lf: rv * np.exp(-lf) + lf - 1.0 - np.log(rv)                  # QLIKE как r − ln r − 1 при r = RV/F
    s = rv / f - 1.0
    assert abs(-(q(np.log(f) + e) - q(np.log(f) - e)) / (2 * e) - s) < 1e-8


def test_comparisons_only_on_common_pairs():
    k = synth(60)
    k["A"], k["B"] = k["k1"], k["k2"]
    k.loc[3, "A"] = np.nan
    k.loc[7, "B"] = -1.0
    c = S.common(k.assign(t_plus=k["t_plus"] + pd.Timedelta(days=4000)), ["A", "B"])
    assert len(c) == len(k) - 2 and c["A"].notna().all() and (c["B"] > 0).all()


def test_mutation_forecast_shifted_plus_one_day_is_caught():
    k = synth(900)
    g = k[k["ticker"] == "LKOH"].reset_index(drop=True)
    rv = pd.Series(g["rv_tplus"].to_numpy(), index=g["t_plus"])
    f = pd.Series(np.exp(np.log(g["rv_tplus"]).shift(1).bfill().to_numpy() + np.random.default_rng(5).normal(0, 0.2, len(g))), index=g["t_plus"])      # честный прогноз: последний известный ln RV + шум
    good = AU.shift_ok(rv, f)
    bad = AU.shift_ok(rv.iloc[:-1], f.shift(-1).iloc[:-1])
    assert good and not bad


def test_frozen_inputs_sha_and_k8_bitwise_unchanged():
    assert bool(CT.sha_rows()["ok"].all())
    from models_v2.s11 import v4_fin_data as FD
    import models_v2.s12.v5_0_io as IO
    k = IO.frame()
    same, mx, nb, nout = CT.k8_rows(k)
    assert same and mx == 0.0 and nb == len(k) and nout == 0 and FD.OUT.exists()
