"""V5 лист 1a: ответы с известным результатом — связи W (три соседа) и λ (низкий ранг), оценка θ, охват w. Мутации: соседи не по корреляции, знак λ, w не в 0,4."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_gs_enc as EN, v5_gs_fit as FI, v5_gs_graph as GR, v5_gs_panel as PA


def _x(n=400, m=6, seed=4):
    rng = np.random.default_rng(seed)
    f, g = rng.normal(size=n), rng.normal(size=n)
    X = np.column_stack([f + 0.3 * rng.normal(size=n) for _ in range(3)] + [g + 0.3 * rng.normal(size=n) for _ in range(2)] + [rng.normal(size=n)])
    return X


def test_graph_c_three_neighbors_by_positive_correlation_weights_sum_one():
    X = _x()
    W, c, n_eq, _ = GR.graph_c(X, None)
    assert n_eq == 0
    for i in range(3):                                            # первая группа: соседи — две другие из группы (положительная корр.) и не более трёх
        nz = np.flatnonzero(W[i])
        assert set(nz) >= {j for j in range(3) if j != i} and len(nz) <= 3 and abs(W[i].sum() - 1.0) < 1e-12 and W[i, i] == 0


def test_graph_c_short_history_gives_equal_rows_and_loto_excludes():
    W, _, n_eq, _ = GR.graph_c(_x(50), None)
    assert n_eq == 6 and np.allclose(W, 1 - np.eye(6))
    W2, c2, _, _ = GR.graph_c(_x(), 1)
    assert (W2[:, 1] == 0).all() and (W2[1, [0, 2, 3, 4, 5]] == 1).all() and c2[1] == 1.0                 # столбец исключённой бумаги нулевой, её строка — равные веса


def test_graph_d_low_rank_known_answer_and_sign():
    X = _x()
    W, c, n_eq, spread = GR.graph_d(X, None)
    assert n_eq == 0 and spread >= 1.0 and (W >= 0).all() and abs(c.mean() - 1.0) < 1e-9                  # среднее c = 1: λ̃ = ½λ + ½λ̄ сохраняет среднее
    assert (c > 0).all() and c[:3].min() > c[5]                                                            # три бумаги общего фактора несут больший вес, чем независимая


def test_graph_d_loto_column_zero_and_own_row_equal_weights():
    W, c, _, _ = GR.graph_d(_x(), 5)
    assert (W[:, 5] == 0).all() and c[5] == 1.0 and (W[5, :5] == 1).all()


def test_theta_fit_recovers_positive_alpha_on_persistent_error_and_stays_near_prior_on_noise():
    rng = np.random.default_rng(0)
    n, m = 900, 4
    ax = pd.bdate_range("2017-01-02", periods=n)
    out = {}
    for name, ar in (("persist", 0.9), ("noise", 0.0)):
        e = np.zeros((n, m))
        for t in range(1, n):
            e[t] = ar * e[t - 1] + rng.normal(0, 0.15, m)
        rv = np.exp(e + rng.normal(0, 0.2, (n, m)))
        rows = [pd.DataFrame({"ticker": f"T{j}", "date": pd.DatetimeIndex([ax[0] - pd.offsets.BDay(1)]).append(ax[:-1]), "t_plus": ax, "group": "M16", "rv_tplus": rv[:, j], "k1": 1.0, "k2": 1.0, "k3": 1.0}) for j in range(m)]
        P = PA.build(pd.concat(rows, ignore_index=True), [f"T{j}" for j in range(m)])
        par = FI.run_config(P, P.k8, "B", False)["params"]
        out[name] = par.iloc[-1]
    assert out["persist"]["alpha"] > 0.05 and out["persist"]["alpha"] > 3 * out["noise"]["alpha"]
    assert out["noise"]["alpha"] < 0.05 and out["persist"]["fitted"]


def test_encompass_weight_known_answer_and_mutation_swapped_roles_caught():
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 0.3, 5000)
    lr = np.exp(0.4 * delta)                                       # RV = M^(0.6)·F^(0.4)
    assert abs(EN.w_hat(lr, delta) - 0.4) < 1e-6
    assert abs(EN.w_hat(lr, -delta) - 0.4) > 0.1                   # перепутанный знак δ даёт другой вес
