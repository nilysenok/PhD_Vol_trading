"""S-Z-IMPL-1 (Ж-258, разд. 2, 3.4, 8): `z_model` — известные ответы на синтетике (статистики, обрезка, форма ln C, вложение θ_Z = 0 побитно, потеря, голова, прогноз). Без сети; боевые выходы не пишутся."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import math

import numpy as np
import pytest

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import z_model as ZM
from models_v2.s18.r_loss_syn import synth_data

N = 400


def world(n=N, seed=0, sig_z=0.0):
    """(V, z̃, z): π дня и β·g различны по парам — день пары важен; RV = F⁰·Z·exp(sig_z·z̃₀)."""
    rng = np.random.default_rng(seed)
    D, lv = synth_data(n, seed, np.zeros(6))
    z = rng.normal(2.0, 3.0, size=(n, 4))
    zt = ZM.ztil(z, *ZM.std_stats(z, np.arange(n)))
    lv = RL.Lv(rng.uniform(0.3, 1.0, n), lv.lnb, 0.1 * rng.normal(size=n), lv.has)
    D.rv = D.rv * np.exp(sig_z * zt[:, 0])
    return RL.view(D, lv, np.arange(n)), zt, z


def rho_np(u, k):
    lk = math.log(k)
    return np.where(u <= lk, np.exp(np.minimum(u, lk)) - u - 1.0, (k - 1.0 - lk) + (k - 1.0) * (u - lk))


def theta_ck(rng, k=0):
    b = np.array(CM.bounds("CK"))
    return b[:, 0] + (b[:, 1] - b[:, 0]) * rng.random(6)


def test_constants_and_api_shapes():
    assert ZM.CLIP == 3.0 and ZM.NP == {"CKZ": 10, "Z": 4} and ZM.ZB == (-3.0, 3.0)
    assert len(ZM.starts("CKZ")) == len(RL.SEEDS) and len(ZM.starts("Z")) == 1
    assert len(ZM.bounds("CKZ")) == 10 and ZM.bounds("CKZ")[6:] == [(-3.0, 3.0)] * 4 and ZM.bounds("Z") == [(-3.0, 3.0)] * 4
    for x in ZM.starts("CKZ"):
        assert (x[6:] == 0).all() and len(x) == 10
    assert (ZM.starts("Z")[0] == 0).all()


def test_std_stats_known_answer_only_rows_of_T():
    z = np.array([[1.0, 10, 0, 5], [3.0, 10, 2, 5.5], [5.0, 10, 4, 6], [1e9, -1e9, np.nan, 1e5]])
    with pytest.raises(ValueError, match="σ ≤ 0"):
        ZM.std_stats(z, [0, 1, 2])                                    # колонка 1 постоянна: подстановок нет
    z[:, 1] = [1.0, 2.0, 6.0, 1e9]
    mu, sd = ZM.std_stats(z, np.array([True, True, True, False]))
    assert np.allclose(mu, [3.0, 3.0, 2.0, 5.5]) and np.allclose(sd, [np.sqrt(8 / 3), np.sqrt(14 / 3), np.sqrt(8 / 3), np.sqrt(1 / 6)])   # ddof = 0
    assert np.array_equal(ZM.std_stats(z, [0, 1, 2])[1], sd)


def test_std_stats_refuses_nonfinite_in_T_without_substitution():
    z = np.arange(20.0).reshape(5, 4) + np.eye(5, 4) * 7
    z[4, 2] = np.nan
    ZM.std_stats(z, [0, 1, 2, 3])                                     # NaN вне T — не мешает
    for bad in (np.nan, np.inf, -np.inf):
        z[1, 0] = bad
        with pytest.raises(ValueError, match="нефинитный"):
            ZM.std_stats(z, [0, 1, 2])


def test_ztil_clip_after_standardization_known_answer():
    z, mu, sd = np.array([[11.0, 13.0, -1.0, np.nan], [5.0, 5.0, 5.0, 5.0]]), np.full(4, 5.0), np.full(4, 2.0)
    raw = ZM.ztil_raw(z, mu, sd)
    assert np.array_equal(raw, np.array([[3.0, 4.0, -3.0, np.nan], [0.0, 0.0, 0.0, 0.0]]), equal_nan=True)
    assert np.array_equal(ZM.ztil(z, mu, sd), np.array([[3.0, 3.0, -3.0, np.nan], [0.0, 0.0, 0.0, 0.0]]), equal_nan=True)
    assert np.array_equal(ZM.ztil([[-100.0] * 4], mu, sd), -3.0 * np.ones((1, 4)))


def test_eta_and_head_forms_zero_theta_nests_bitwise():
    V, zt, _ = world()
    rng = np.random.default_rng(1)
    th = theta_ck(rng)
    ref = torch.log(V.c8 * CM.forward(torch.as_tensor(th), V, "CK")["M"]).numpy()
    assert np.array_equal(ZM.lnc_head(np.concatenate([th, np.zeros(4)]), V, zt, "CKZ").numpy(), ref)           # Н1
    z0 = ZM.lnc_head(torch.zeros(4, dtype=torch.float64), V, zt, "Z").numpy()
    assert np.array_equal(z0, RL.lnc_of(torch.zeros(6, dtype=torch.float64), V, "CK").numpy()) and np.array_equal(z0, torch.log(V.c8).numpy())
    thz = 0.3 * rng.normal(size=4)
    assert np.allclose(ZM.eta_of(zt, torch.as_tensor(thz)).numpy(), zt @ thz, rtol=1e-12, atol=0)
    lz = ZM.lnc_head(np.concatenate([th, thz]), V, zt, "CKZ").numpy()
    assert np.array_equal(lz, ref + ZM.eta_of(zt, torch.as_tensor(thz)).numpy()) and np.allclose(lz, ref + zt @ thz, rtol=1e-12, atol=0)
    assert np.allclose(ZM.lnc_head(thz, V, zt, "Z").numpy(), torch.log(V.c8).numpy() + zt @ thz, rtol=1e-12, atol=0)


def test_lnc_head_rejects_wrong_length_and_var():
    V, zt, _ = world(50)
    for v, var in ((np.zeros(4), "CKZ"), (np.zeros(10), "Z"), (np.zeros(10), "X")):
        with pytest.raises(ValueError):
            ZM.lnc_head(v, V, zt, var)


def test_srz_theta_zero_lnf_equals_level_baseline():
    """S-RZ при θ_Z = 0: ln F = (1 − π)·ln B + π·log c8 + β·g — тот же путь, что `lnf0`."""
    V, zt, _ = world()
    want = (1 - V.l_pi) * V.l_lnb + V.l_pi * torch.log(V.c8) + V.l_bg
    assert np.array_equal(RL.ln_f(ZM.lnc_head(np.zeros(4), V, zt, "Z"), V).numpy(), want.numpy())


@pytest.mark.parametrize("var", ["CKZ", "Z"])
@pytest.mark.parametrize("mode", ["R", "plain"])
def test_loss_z_equals_independent_numpy_with_day_of_pair_pi(var, mode):
    V, zt, _ = world(sig_z=0.3)
    rng = np.random.default_rng(3)
    v = np.concatenate([theta_ck(rng), 0.3 * rng.normal(size=4)]) if var == "CKZ" else 0.3 * rng.normal(size=4)
    c, m, lam = 2.2, 0.9, 0.01
    lnc = ZM.lnc_head(v, V, zt, var).numpy()
    pi, lnb, bg = V.l_pi.numpy(), V.l_lnb.numpy(), V.l_bg.numpy()
    u = np.log(V.rv.numpy()) - ((1 - pi) * lnb + pi * lnc + bg)
    r = rho_np(u - math.log(m), c / m) if mode == "R" else np.exp(u) - u - 1.0
    got = float(ZM.loss_z(v, V, zt, var, lam, c, m, mode))
    assert abs(got - (r.mean() + float(CM.penalty(torch.as_tensor(v), lam)))) < 1e-12
    assert float(ZM.loss_z(v, V, zt, var, lam, c, m, mode)) - float(ZM.loss_z(v, V, zt, var, 0.0, c, m, mode)) > 0     # штраф есть
    with pytest.raises(ValueError):
        ZM.loss_z(v, V, zt, var, lam, c, m, "bad")


def test_penalty_covers_whole_vector_including_theta_z():
    V, zt, _ = world(60)
    a, b = np.zeros(4), np.array([0.5, 0, 0, 0.0])
    d = lambda v: float(ZM.loss_z(v, V, zt, "Z", 1.0, 2.0, 1.0)) - float(ZM.loss_z(v, V, zt, "Z", 0.0, 2.0, 1.0))
    assert d(a) == 0.0 and d(b) > 0.0 and d(b) == pytest.approx(float(CM.penalty(torch.as_tensor(b), 1.0)), rel=1e-12)


def test_fit_head_recovers_sign_and_stays_in_bounds():
    est = []
    for s in range(6):
        V, zt, _ = world(600, s, sig_z=0.3)
        th = ZM.fit_head(V, zt, "Z", 1e-3, *RL.clip_stats(RL.x0_of(V, "CK"), np.arange(600)), "R")
        assert th.shape == (4,) and (np.abs(th) <= 3.0).all()
        est.append(th[0])
    assert np.median(est) > 0.1 and (np.array(est) > 0).sum() >= 5


def test_fit_head_ckz_starts_at_nested_and_improves_or_equals_loss():
    V, zt, _ = world(300, 4, sig_z=0.3)
    c, m = RL.clip_stats(RL.x0_of(V, "CK"), np.arange(300))
    th = ZM.fit_head(V, zt, "CKZ", 1e-2, c, m, "R", seeds=(0, 1), maxiter=60)
    f0 = min(float(ZM.loss_z(x, V, zt, "CKZ", 1e-2, c, m)) for x in ZM.starts("CKZ", (0, 1)))
    assert th.shape == (10,) and float(ZM.loss_z(th, V, zt, "CKZ", 1e-2, c, m)) <= f0 + 1e-12


def test_predict_lnc_head0_plus_mean_deviation_known_answers():
    V, zt, z = world(200, 2)
    rng = np.random.default_rng(5)
    zts = [zt] + [ZM.ztil(z + 0.2 * rng.normal(size=z.shape), *ZM.std_stats(z, np.arange(200))) for _ in range(4)]
    heads = [np.concatenate([theta_ck(rng), 0.3 * rng.normal(size=4)]) for _ in range(5)]
    ls = [ZM.lnc_head(h, V, q, "CKZ").numpy() for h, q in zip(heads, zts)]
    assert np.allclose(ZM.predict_lnc(heads, V, zts), np.mean(ls, axis=0), rtol=1e-12, atol=1e-12)
    assert np.array_equal(ZM.predict_lnc(heads[:1], V, zts[:1]), ls[0])
    h0 = np.concatenate([theta_ck(rng), np.zeros(4)])
    base = ZM.lnc_head(h0, V, zt, "CKZ").numpy()
    assert np.array_equal(ZM.predict_lnc([h0] * 5, V, [zt] * 5), base) and np.array_equal(ZM.predict_lnc([h0] * 5, V, zts), base)    # θ_Z = 0: головы различаются лишь z̃
    hz = [np.zeros(4)] * 2
    assert ZM.predict_lnc(hz, V, zts[:2]).shape == (200,)                                                                       # var по длине θ
    with pytest.raises(ValueError):
        ZM.predict_lnc(heads, V, zts[:3])
    with pytest.raises(ValueError):
        ZM.predict_lnc([], V, [])
