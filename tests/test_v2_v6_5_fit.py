"""V6-5, G41: робастные цели F1 и F2 — градиент, совпадение с LIN при одной среде, сдвиг к худшей среде, слияние малых сред, PIT сред через боевой путь (мутация «среда из будущих пар»)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_1_fit as FT, v6_1_wf as WF, v6_5_fit as R, v6_5_wf as W5


def _data(seed=0, n=(120, 80, 60), k=4):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(sum(n), k))
    lab = np.repeat(np.arange(len(n)), n)
    e = 0.3 * Z[:, 0] * (1 + lab) + 0.2 * rng.normal(size=len(lab)) * (1 + lab)
    return np.column_stack([np.ones(len(Z)), Z]), e, lab


def _num_grad(f, th, h=1e-6):
    out = np.zeros_like(th)
    for i in range(len(th)):
        d = np.zeros_like(th)
        d[i] = h
        out[i] = (f(th + d) - f(th - d)) / (2 * h)
    return out


@pytest.mark.parametrize("kind", ["F1", "F2"])
def test_analytic_gradient_equals_numeric(kind):
    A, e, lab = _data()
    th = np.random.default_rng(1).normal(scale=0.3, size=A.shape[1])
    val, gr = R.robust_obj(th, A, e, lab, 0.05, kind)
    num = _num_grad(lambda t: R.robust_obj(t, A, e, lab, 0.05, kind)[0], th)
    assert np.abs(gr - num).max() <= 1e-6


@pytest.mark.parametrize("kind", ["F1", "F2"])
def test_one_env_equals_lin_objective_and_solution(kind):
    A, e, _ = _data(n=(200,))
    lab = np.zeros(len(e), int)
    th = np.random.default_rng(2).normal(scale=0.3, size=A.shape[1])
    val, gr = R.robust_obj(th, A, e, lab, 0.1, kind)
    assert abs(val - FT._obj(A, e, th, 0.1)) <= 1e-10
    lin = R.robust_obj(th, A, e, lab, 0.1, "LIN")
    assert np.abs(gr - lin[1]).max() <= 1e-10
    th_r, info = R.fit_robust_fixed(A[:, 1:], e, lab, 0.1, kind)
    assert np.abs(th_r - FT.fit_lin_fixed(A[:, 1:], e, 0.1)).max() <= 1e-6 and info["success"]


def test_f2_shifts_to_worse_env_and_lin_does_not():
    rng = np.random.default_rng(3)
    n = (900, 100)
    z = rng.normal(size=(sum(n), 1))
    lab = np.repeat([0, 1], n)
    e = np.where(lab == 0, 0.4 * z[:, 0], -0.4 * z[:, 0] + 0.3) + 0.1 * rng.normal(size=len(lab))
    A = np.column_stack([np.ones(len(z)), z])
    th_lin = FT.fit_lin_fixed(z, e, 1e-3)
    th_f2, _ = R.fit_robust_fixed(z, e, lab, 1e-3, "F2")
    q_lin, q_f2 = R.env_q(A, e, th_lin, lab)[0], R.env_q(A, e, th_f2, lab)[0]
    assert q_lin[1] > q_lin[0]                                  # среда 1 — худшая
    assert q_f2[1] < q_lin[1] - 1e-3 and np.abs(th_f2 - th_lin).max() > 1e-3
    assert (q_lin * np.array(n)).sum() < (q_f2 * np.array(n)).sum()   # LIN оптимален по среднему по парам: он среду не выравнивает


def test_small_env_merges_with_previous_and_first_with_next():
    y = lambda d: pd.to_datetime(d)
    tp = np.r_[np.repeat(y("2015-08-01"), 100), np.repeat(y("2016-03-01"), 10), np.repeat(y("2017-03-01"), 150)]
    assert env_labels_list(tp) == [0, 1]
    tp = np.r_[np.repeat(y("2015-08-01"), 5), np.repeat(y("2016-03-01"), 60), np.repeat(y("2017-03-01"), 150)]
    assert env_labels_list(tp) == [0, 1]                         # 2015 (5 пар) — со следующей (2016)
    assert (R.env_labels(tp)[:65] == 0).all() and (R.env_labels(tp)[65:] == 1).all()
    assert (R.env_labels(np.repeat(y("2017-01-03"), 8)) == 0).all()       # всё мало — одна среда


def env_labels_list(tp):
    return sorted(set(R.env_labels(tp).tolist()))


def _panel(n_days=1100, seed=4):
    ix = pd.bdate_range("2015-07-07", periods=n_days)
    rng = np.random.default_rng(seed)
    tab = pd.DataFrame({"t_plus": ix.to_series().shift(-1).bfill().to_numpy(), "rv_next": np.exp(rng.normal(size=n_days) * 0.3), "h": np.ones(n_days)}, index=ix)
    return tab, pd.DataFrame(rng.normal(size=(n_days, 3)), index=ix, columns=["a", "b", "c"])


def _leaks(tab, S):
    seen = []

    class Dummy:
        def g(self, X):
            return np.zeros(len(X))

    def fit(X, e, tp):
        seen.append(pd.DatetimeIndex(tp).max())
        return Dummy()

    run = W5.walk_forward_env(tab, S, fit, first="2016-01-01")
    return any(s > f["d0"] for s, f in zip(seen, run["fits"]))


def test_envs_use_only_closed_pairs_and_future_mutation_caught(monkeypatch):
    tab, S = _panel()
    assert not _leaks(tab, S)
    monkeypatch.setattr(WF, "closed_rows", lambda t_plus, d0: np.flatnonzero(np.asarray(t_plus, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(400, "D")))
    assert _leaks(tab, S)


@pytest.mark.parametrize("kind", ["F1", "F2"])
def test_walk_forward_env_real_fit_runs(kind):
    tab, S = _panel(n_days=900)
    run = W5.walk_forward_env(tab, S, lambda X, e, tp: R.fit_robust(X, e, tp, kind), first="2018-06-01")
    assert np.isfinite(run["g"]).all() and len(run["fits"]) == 7
    c = W5.conv_table(run, kind)
    assert (c["n_env"] >= 1).all() and np.isfinite(c["grad_norm"]).all() and set(c["config"]) == {kind}
