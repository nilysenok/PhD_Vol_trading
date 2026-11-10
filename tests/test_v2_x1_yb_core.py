"""Y-B: признаки, масштаб, градиент, отключение, мутации PIT (Y3/Y4). Синтетика + реальный склад M10 (только чтение)."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_yb_feat as FE, x1_yb_fit as FT, x1_yb_walk as W
from models_v2.s16 import x1_ya_walk as YW


def synth(nd=40, seed=0, drop=True):
    rng = np.random.default_rng(seed)
    pr = np.ones((nd, FE.SLOTS), bool)
    if drop:
        pr[rng.random(pr.shape) < 0.1] = False
    r = np.where(pr, rng.normal(0, 1, pr.shape), 0.0)
    vol = np.where(pr, rng.integers(100, 10000, pr.shape), 1).astype(float)
    return dict(days=pd.bdate_range("2020-01-01", periods=nd), r=r, r2=r * r, lvol=np.where(pr, np.log(vol), 0.0), present=pr)


def pairs(st, d_idx):
    d = st["days"][d_idx]
    n = len(d)
    rng = np.random.default_rng(5)
    return FE.pair_data(st, d, rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n))


def test_w_one_gives_rv_bitwise_and_mean_one():
    st = synth()
    D, _ = pairs(st, np.arange(25, 38))
    w = FT.weights(np.zeros(4), D["Phi"], D["M"], D["n"])
    assert np.array_equal(w, D["M"].astype(float))
    assert np.array_equal((w * D["R2"]).sum(1), D["R2"].sum(1))
    assert np.array_equal(FT.h_y3(np.zeros(4), D), np.zeros(len(D["n"])))
    w = FT.weights(np.array([.3, -.5, .2, .4]), D["Phi"], D["M"], D["n"])
    assert np.allclose(w.sum(1) / D["n"], 1.0, atol=1e-12) and (w[~D["M"]] == 0).all()


def test_theta_zero_or_gamma_zero_gives_x1_bitwise():
    st = synth()
    D, _ = pairs(st, np.arange(25, 38))
    x1 = D["u"] * 0 + 1.3
    for x in (np.array([0.0, .4, -.2, .1, .3]), np.array([0.7, 0, 0, 0, 0])):
        g = x[0] * FT.h_of(x, D, "Y3")
        assert np.array_equal(x1 * np.exp(g), x1)


def test_analytic_gradient_equals_numeric_and_dead_point():
    st = synth()
    D, _ = pairs(st, np.arange(10, 38))
    x = np.array([0.4, .3, -.2, .5, -.4])
    for lam in (0.0, 0.1):
        L, g = FT.loss_grad3(x, D, lam)
        num = approx_fprime(x, lambda z: FT.loss_grad3(z, D, lam)[0], 1e-7)
        assert np.allclose(g, num, atol=1e-6)
    assert np.allclose(FT.loss_grad3(np.zeros(5), D, 0.0)[1], 0.0, atol=1e-12)          # мёртвая точка
    assert np.abs(FT.loss_grad3(FT.start("Y3", 0), D, 0.0)[1]).min() > 0                # старт плана оживает
    L4, g4 = FT.loss_grad4(np.array([0.3]), D, 0.05)
    assert abs(g4[0] - approx_fprime(np.array([0.3]), lambda z: FT.loss_grad4(z, D, 0.05)[0], 1e-7)[0]) < 1e-6


def test_slot_median_excludes_current_day_and_mutation_caught():
    st = synth(60)
    a, _ = FE.phi_all(st)
    st2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    j = 45
    st2["r"][j:], st2["r2"][j:], st2["lvol"][j:] = st["r"][j:] * 3, st["r2"][j:] * 9, st["lvol"][j:] + 1
    b, _ = FE.phi_all(st2)
    assert np.array_equal(a[:j], b[:j])
    # φ2/φ4 дня j зависят от его собственных баров, но не от баров дня j+1 и позже
    st3 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st3["r"][j + 1:] *= 5
    c, _ = FE.phi_all(st3)
    assert np.array_equal(a[: j + 1], c[: j + 1])
    # мутация lag=0 (с текущим днём): φ2 слота в дне j зависит от его же значения как от истории
    m0 = FE.slot_roll(np.log(np.maximum(np.abs(st["r"]), 1e-8)), st["present"], 20, np.median, lag=0)
    m1 = FE.slot_roll(np.log(np.maximum(np.abs(st["r"]), 1e-8)), st["present"], 20, np.median, lag=1)
    assert not np.array_equal(np.nan_to_num(m0), np.nan_to_num(m1))
    st4 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st4["r"][j] *= 7
    d1 = FE.slot_roll(np.log(np.maximum(np.abs(st4["r"]), 1e-8)), st4["present"], 20, np.median, lag=1)
    assert np.array_equal(np.nan_to_num(d1[j]), np.nan_to_num(m1[j])) and not np.array_equal(np.nan_to_num(d1[j + 1:j + 2]), np.nan_to_num(m1[j + 1:j + 2]))


def test_median_slot_with_day_d_mutation_caught_in_pair_data():
    st = synth(60)
    ph = FE.phi_all(st)
    D, _ = pairs(st, np.array([50]))
    mut = FE.phi_all(st, lag=0)
    assert not np.array_equal(ph[0][50], mut[0][50])


def test_bar_of_t_plus_in_B_mutation_caught(monkeypatch):
    st = synth(60)
    D0, _ = pairs(st, np.arange(30, 40))
    st2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st2["r"][41:], st2["r2"][41:] = 9.0, 81.0
    D1, _ = pairs(st2, np.arange(30, 40))
    assert all(np.array_equal(D0[k], D1[k]) for k in D0 if k != "u")
    monkeypatch.setattr(FE, "feat_row", lambda i: i + 1)           # мутация: бары дня t⁺
    D2, _ = pairs(st, np.arange(30, 40))
    D3, _ = pairs(st2, np.arange(30, 40))
    assert not np.array_equal(D2["R2"][-1], D3["R2"][-1]) or not np.array_equal(D2["lrv"], D0["lrv"])


def test_profile_y4_pit_and_whole_sample_mutation_caught():
    st = synth(300)
    p = FE.profile_all(st)
    st2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
    st2["r2"][200:] *= 4
    assert np.array_equal(p[:201], FE.profile_all(st2)[:201])                # профиль дня d не видит d и позже
    rv = st["r2"].sum(1)
    share = st["r2"] / rv[:, None]
    whole = np.where(st["present"], share, 0).sum(0) / st["present"].sum(0)          # мутация: по всей выборке
    assert not np.allclose(p[100], whole)
    assert (p[0] == 0).all()
    # окно ровно 250 прошлых: на дне 299 профиль слота = среднее долей дней 49…298 с баром
    k = 3
    idx = np.flatnonzero(st["present"][:299, k])[-250:]
    assert np.isclose(p[299, k], share[idx, k].mean())


def test_y4_weights_mean_one_and_fallback():
    pb = np.array([[.2, .3, .5], [0, 0, 0]])
    m = np.array([[1, 1, 0], [1, 1, 1]], bool)
    w, bad = FE.weights_fixed(pb, m, m.sum(1).astype(float))
    assert np.allclose(w[0], [2 * .4, 2 * .6, 0]) and np.array_equal(w[1], [1, 1, 1]) and bad == 1


def test_softmax_over_absent_bars_mutation_breaks_scale():
    st = synth()
    D, _ = pairs(st, np.arange(25, 38))
    gam = np.array([.3, -.5, .2, .4])
    z = D["Phi"] @ gam                                              # мутация: softmax и по отсутствующим барам
    e = np.exp(z - z.max(1, keepdims=True))
    w_bad = D["n"][:, None] * e / e.sum(1, keepdims=True)
    assert not np.allclose(w_bad.sum(1) / D["n"], 1.0, atol=1e-6) or (w_bad[~D["M"]] != 0).any()
    w = FT.weights(gam, D["Phi"], D["M"], D["n"])
    assert np.allclose(w.sum(1), D["n"], atol=1e-10)


def fake_x1(nd):
    d = pd.bdate_range("2018-01-03", periods=nd)
    rng = np.random.default_rng(2)
    return pd.DataFrame(dict(d=d, t_plus=d + pd.offsets.BDay(1), rv=rng.uniform(.5, 2, nd), F=rng.uniform(.5, 2, nd)))


def _walk_setup(monkeypatch, nd=200):
    st = synth(nd, drop=False)
    x1 = fake_x1(nd)
    x1["d"], x1["t_plus"] = st["days"], st["days"] + pd.offsets.BDay(1)
    D, _ = FE.pair_data(st, x1["d"], x1["rv"].to_numpy(), x1["F"].to_numpy())
    ms = [p for p in YW.month_starts(x1["d"]) if p >= 100]
    a, b = ms[0], ms[1]
    for m in (YW, W):
        monkeypatch.setattr(m, "ACTIVATION", pd.Timestamp(x1["d"][a]))
        monkeypatch.setattr(m, "MIN_PAIRS", 50)
    return x1, D, a, b


def test_walk_off_before_activation_bitwise(monkeypatch):
    x1, D, a, b = _walk_setup(monkeypatch)
    f = W.walk(x1, D, "Y3", seeds=(0, 1))
    off = f["active"] == 0
    assert off.sum() == a and np.array_equal(f.loc[off, "F"].to_numpy(), x1.loc[off, "F"].to_numpy())
    assert (f.loc[~off, "active"] == 1).all() and (f.loc[~off, "lam"] >= 0).all()


def test_closed_plus_one_mutation_caught(monkeypatch):
    x1, D, a, b = _walk_setup(monkeypatch)
    D2 = {k: v.copy() for k, v in D.items()}
    D2["u"][a] *= 3.0                       # пара a: t⁺ = d[a+1] > d[a], к дню переоценки НЕ закрыта
    for mode, kw in (("Y3", dict(seeds=(0, 1))), ("Y4", {})):
        honest = W.walk(x1, D, mode, **kw)["g"].to_numpy()[a:b]
        assert np.array_equal(honest, W.walk(x1, D2, mode, **kw)["g"].to_numpy()[a:b])        # честный путь не видит
    monkeypatch.setattr(YW, "closed_date", lambda tp, d: np.asarray(tp, "datetime64[ns]") - np.timedelta64(3, "D"))    # мутация: «закрыто + 1»
    assert not np.array_equal(W.walk(x1, D, "Y4")["g"].to_numpy()[a:b], W.walk(x1, D2, "Y4")["g"].to_numpy()[a:b])
