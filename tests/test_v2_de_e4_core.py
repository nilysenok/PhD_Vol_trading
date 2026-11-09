"""E4 (X4): окна ансамбля, PIT затвора (мутации через боевой путь `run`), κ = 0 ⇒ α ≡ 1, α ∈ (0; 1], выбор κ — известный ответ; цель GBM — прежняя V6 (градиент/гессиан = численные)."""
import numpy as np
import pandas as pd

from models_v2.s15 import de_e4_core as C

FIRST = "2017-01-01"


class Fake:
    """Обучатель-заглушка: g зависит от e обучающих строк (утечка видна) и от размера окна (разброс не нулевой); записывает строки обучения."""
    calls: list = []

    def __init__(self, X, e):
        self.val = float(np.clip(np.mean(e) + 0.01 * np.log(len(e)), -2, 2))
        Fake.calls.append(X[:, 0].astype(int))

    def g(self, X):
        return np.full(len(X), self.val)


def fake_fit(X, e):
    return Fake(X, e)


def synth(n=1900, seed=1):
    d = pd.bdate_range("2015-07-07", periods=n)
    rng = np.random.default_rng(seed)
    tp = d[1:].append(pd.DatetimeIndex([d[-1] + pd.offsets.BDay(1)])).to_numpy("datetime64[ns]")
    H = np.exp(0.2 * rng.standard_normal(n))
    e = 0.3 * rng.standard_normal(n)
    X = np.column_stack([np.arange(n), rng.standard_normal(n)])
    return d, tp, H, H * np.exp(e), e, X


def test_windows_are_exactly_750_1000_1250_1500_last_closed_pairs():
    d, tp, H, rv, e, X = synth()
    Fake.calls = []
    w = C.walk_forward(X, e, tp, d, fake_fit)
    last = w["rows"][-1]
    a = int(np.flatnonzero(d == last[0])[0])
    assert last[1] == [a, 750, 1000, 1250, 1500]                         # расширяющееся — все закрытые (a пар), остальные — ровно N
    rows = Fake.calls[-5:]
    assert np.array_equal(rows[0], np.arange(a)) and all(np.array_equal(r, np.arange(a - n, a)) for r, n in zip(rows[1:], (750, 1000, 1250, 1500)))
    early = w["rows"][0]
    assert early[1][1] == early[1][0] < 750                              # закрытых меньше N — все закрытые


def _train_violates(calls, d, tp, w):
    """Есть ли обучающая строка с t⁺ > d0 (пара из будущего) на какой-либо переоценке."""
    return any(tp[rows].max() > np.datetime64(d0) for rows, (d0, _) in zip(calls[::5], w["rows"]))


def test_pair_with_tplus_after_d_is_caught_and_honest_path_clean(monkeypatch):
    d, tp, H, rv, e, X = synth()
    Fake.calls = []
    w = C.walk_forward(X, e, tp, d, fake_fit)
    assert not _train_violates(Fake.calls, d, tp, w)
    monkeypatch.setattr(C, "closed_rows", lambda t, d0: np.flatnonzero(np.asarray(t, "datetime64[ns]") <= np.datetime64(d0) + np.timedelta64(1, "D") * 1))
    Fake.calls = []
    w2 = C.walk_forward(X, e, tp, d, fake_fit)
    assert _train_violates(Fake.calls, d, tp, w2)                        # мутант «закрыто + 1» ловится


def _outputs(d, tp, H, rv, e, X):
    return C.run(X, e, rv, H, tp, d, fake_fit)


def _perturb_future(d, tp, H, rv, e, X, k):
    e2, X2, rv2 = e.copy(), X.copy(), rv.copy()
    e2[k:] += 0.7
    X2[k:, 1] += 3.0
    rv2[k:] = H[k:] * np.exp(e2[k:])
    return d, tp, H, rv2, e2, X2


def test_future_data_do_not_change_past_forecasts_and_s_bar_future_mutation_is_caught(monkeypatch):
    base = synth()
    d = base[0]
    k = int(np.flatnonzero(d >= pd.Timestamp("2022-06-01"))[0])
    a = _outputs(*base)
    b = _outputs(*_perturb_future(*base, k))
    cut = k - 30                                                          # раньше начала месяца переоценки, где меняется обучение
    for name in ("F", "alpha", "sbar", "kappa", "gbar"):
        assert np.array_equal(a[name][:cut], b[name][:cut], equal_nan=True), name
    assert np.isfinite(a["sbar"][cut - 1]) and a["act"][cut - 1]
    monkeypatch.setattr(C, "sbar_series", lambda sd, act: np.full(len(sd), float(np.median(sd[act]))))    # мутант: s̄ по всем датам, включая будущие
    a2, b2 = _outputs(*base), _outputs(*_perturb_future(*base, k))
    assert not np.array_equal(a2["alpha"][:cut], b2["alpha"][:cut])


def test_sbar_uses_only_strictly_earlier_active_days():
    sd = np.arange(1, 61, dtype=float)
    act = np.ones(60, bool)
    s = C.sbar_series(sd, act)
    assert np.isnan(s[:20]).all() and s[20] == np.median(sd[:20]) and s[-1] == np.median(sd[:59])
    act2 = act.copy()
    act2[:10] = False                                                     # дни без act в историю не входят
    assert np.isnan(C.sbar_series(sd, act2)[29]) and C.sbar_series(sd, act2)[30] == np.median(sd[10:30])
    sd2 = sd.copy()
    sd2[-1] = 1e6
    assert np.array_equal(C.sbar_series(sd2, act)[:-1], s[:-1], equal_nan=True) and C.sbar_series(sd2, act)[-1] == s[-1]


def test_kappa_zero_gives_alpha_one_and_F_equals_ensemble_bitwise():
    d, tp, H, rv, e, X = synth()
    r = C.run(X, e, rv, H, tp, d, fake_fit, kappas=(0.0,), default=0.0)
    assert (r["alpha"] == 1.0).all() and np.array_equal(r["F"], r["F_ens"])
    assert (C.alpha(np.array([0.1, 2.0]), np.array([0.5, 0.5]), np.array([True, True]), 0.0) == 1.0).all()


def test_alpha_in_unit_interval_positive_and_one_without_history():
    d, tp, H, rv, e, X = synth()
    r = _outputs(d, tp, H, rv, e, X)
    assert ((r["alpha"] > 0) & (r["alpha"] <= 1)).all() and (r["alpha"][~r["act"]] == 1).all()
    assert (r["alpha"][np.isnan(r["sbar"])] == 1).all() and (r["alpha"] < 1).any()
    assert np.isfinite(r["F"]).all() and (r["F"] > 0).all()
    assert (r["G"][~r["fitted"]] == 0).all() and (r["F"][~r["fitted"]] == H[~r["fitted"]]).all()      # до первой переоценки — F = H


def test_select_kappa_known_answer_and_only_closed_pairs():
    n = 600
    d = pd.bdate_range("2017-01-02", periods=n)
    tp = d[1:].append(pd.DatetimeIndex([d[-1] + pd.offsets.BDay(1)])).to_numpy("datetime64[ns]")
    sd = np.where(np.arange(n) % 2 == 0, 0.1, 1.0)
    act, fitted, gbar, H = np.ones(n, bool), np.ones(n, bool), np.full(n, 1.0), np.ones(n)
    rv = np.where(sd < 0.5, np.exp(1.0), np.exp(-2.0))                   # на дни малого разброса поправка верна, на дни большого — сильно не та
    sbar = np.where(np.arange(n) < 30, np.nan, 0.55)                     # s̄ задана явно (медиана двух значений нестабильна)
    kap, nval = C.select_kappa(rv, H, gbar, sd, sbar, act, tp, d, fitted)
    Q = {k: C.qlike(rv, H * np.exp(C.alpha(sd, sbar, act, k) * gbar)) for k in C.KAPPAS}
    j = np.flatnonzero((tp <= np.datetime64(d[400])) & np.isfinite(sbar))[-C.VAL:]
    assert kap[400] == min(C.KAPPAS, key=lambda k: Q[k][j].mean()) and nval[400] == C.VAL and kap[400] == 2.0
    assert (kap[:30] == 1.0).all() and (nval[:30] == 0).all()           # нет истории — κ = 1


def test_gbm_objective_is_v6_and_gradient_hessian_equal_numeric():
    FT = __import__("importlib").import_module("models_v2.s13.v6_1_fit")
    e, g, h = np.array([0.3, -0.5, 1.0]), np.array([0.1, 0.2, -0.4]), 1e-6
    gr, hs = FT.grad_hess(e, g)
    num_g = (FT.qlike_g(e, g + h) - FT.qlike_g(e, g - h)) / (2 * h)
    num_h = (FT.grad_hess(e, g + h)[0] - FT.grad_hess(e, g - h)[0]) / (2 * h)
    assert np.allclose(gr, num_g, atol=1e-7) and np.allclose(hs, num_h, atol=1e-6) and np.allclose(hs, np.exp(e - g))


def test_real_gbm_runs_on_short_synthetic_window():
    d, tp, H, rv, e, X = synth(n=700)
    F = C.walk_forward(X[:, 1:], e, tp, d, first="2018-02-01")
    assert F["fitted"].any() and np.isfinite(F["G"]).all() and (np.abs(F["G"]) <= 1).all()


def test_closed_rows_rejects_pair_closing_after_d():
    tp = np.array(["2020-01-02", "2020-01-03", "2020-01-06"], "datetime64[ns]")
    assert list(C.closed_rows(tp, "2020-01-03")) == [0, 1] and list(C.closed_rows(tp, "2020-01-01")) == []
