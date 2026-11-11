"""S-Z-IMPL-1 (Ж-258, разд. 8): мутации `z_model`. Схема каждой: проверка `ok_*()` проходит на правке и падает на мутанте, подложенном через `monkeypatch` (атрибуты модулей; изоляция — класс № 43 / 110)."""
import torch  # noqa: F401  (первым: конфликт OpenMP, класс № 152)

import math

import numpy as np
import pytest

from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import z_model as ZM
from models_v2.s18.r_loss_syn import synth_data

ORIG = dict(std=ZM.std_stats, ztil=ZM.ztil, head=ZM.lnc_head, pred=ZM.predict_lnc, ln_f=RL.ln_f, rho=RL.rho)


def world(n=300, seed=0, sig_z=0.0):
    rng = np.random.default_rng(seed)
    D, lv = synth_data(n, seed, np.zeros(6))
    z = rng.normal(2.0, 3.0, size=(n, 4))
    zt = ORIG["ztil"](z, *ORIG["std"](z, np.arange(n)))
    lv = RL.Lv(rng.uniform(0.3, 1.0, n), lv.lnb, 0.1 * rng.normal(size=n), lv.has)
    D.rv = D.rv * np.exp(sig_z * zt[:, 0])
    V = RL.view(D, lv, np.arange(n))
    V.dl = (D, lv)
    return V, zt, z


def th_ck(rng):
    b = np.array(CM.bounds("CK"))
    return b[:, 0] + (b[:, 1] - b[:, 0]) * rng.random(6)


# ---------- μ, σ ----------
def ok_stats_rows() -> bool:
    z = np.random.default_rng(0).normal(size=(120, 4))
    T = np.arange(60)
    outs = [np.arange(60, 120), np.arange(100, 120), np.arange(60, 80)]
    for rows in outs:
        z2 = z.copy()
        z2[rows] = 7.0 * z2[rows] + 1000.0
        a, b = ZM.std_stats(z, T), ZM.std_stats(z2, T)
        if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
            return False
    return True


def test_stats_use_only_training_rows_honest():
    assert ok_stats_rows()
    mu, sd = ZM.std_stats(np.random.default_rng(0).normal(size=(120, 4)), np.arange(60))
    assert mu.shape == (4,) and (sd > 0).all()


def test_mutation_mu_sigma_whole_sample_caught(monkeypatch):
    monkeypatch.setattr(ZM, "std_stats", lambda z, T: ORIG["std"](z, np.arange(len(z))))
    assert not ok_stats_rows()


def test_mutation_mu_sigma_with_vb_or_forecast_days_caught(monkeypatch):
    """Блок V_b (60…79) и дни прогноза (80…99) в статистиках."""
    monkeypatch.setattr(ZM, "std_stats", lambda z, T: ORIG["std"](z, np.union1d(T, np.arange(60, 100))))
    assert not ok_stats_rows()


def ok_loss_month() -> bool:
    """Боевая цепочка месяца: μ, σ, z̃, c, m, L_R по строкам T; порча z и RV вне T (будущее, V_b, дни прогноза) ничего не меняет."""
    V, _, z = world(300, 1, 0.3)
    D, lv = V.dl
    T, v = np.arange(200), np.array([0.1, -0.2, 0.3, 0.1])
    for rows in (np.arange(260, 300), np.arange(200, 240), np.arange(240, 260)):
        z2, rv2 = z.copy(), np.array(D.rv, float)
        z2[rows], rv2[rows] = 7.0 * z2[rows] + 1000.0, rv2[rows] * 7.0
        res = []
        for zz, rr in ((z, np.array(D.rv, float)), (z2, rv2)):
            D2 = type(D).__new__(type(D))
            D2.__dict__.update(D.__dict__)
            D2.rv = rr
            VT = RL.view(D2, lv, T)
            mu, sd = ZM.std_stats(zz, T)
            res.append((mu, sd, RL.clip_stats(RL.x0_of(RL.view(D2, lv, np.arange(300)), "CK"), T), float(ZM.loss_z(v, VT, ZM.ztil(zz, mu, sd)[T], "Z", 1e-2, *RL.clip_stats(RL.x0_of(RL.view(D2, lv, np.arange(300)), "CK"), T)))))
        if not (np.array_equal(res[0][0], res[1][0]) and np.array_equal(res[0][1], res[1][1]) and res[0][2] == res[1][2] and res[0][3] == res[1][3]):
            return False
    return True


def test_loss_month_invariant_to_rows_outside_T_and_mutations_caught(monkeypatch):
    assert ok_loss_month()
    monkeypatch.setattr(ZM, "std_stats", lambda z, T: ORIG["std"](z, np.arange(len(z))))
    assert not ok_loss_month()


# ---------- обрезка ----------
def ok_ztil() -> bool:
    z, mu, sd = np.array([[11.0, 13.0, -1.0, -100.0]]), np.full(4, 5.0), np.full(4, 2.0)
    return bool(np.array_equal(ZM.ztil(z, mu, sd), np.array([[3.0, 3.0, -3.0, -3.0]])))


def shift_ratio() -> float:
    V, zt, z = world(300, 7, 0.3)
    n, mu, sd = 300, *ORIG["std"](world(300, 7, 0.3)[2], np.arange(300))
    cs = RL.clip_stats(RL.x0_of(V, "CK"), np.arange(n))
    z30 = z.copy()
    z30[0, 0] = mu[0] + 30.0 * sd[0]
    fit = lambda q: ZM.fit_head(V, q, "Z", 1e-3, *cs, "R")
    base, clip, raw = fit(ZM.ztil(z, mu, sd)), fit(ZM.ztil(z30, mu, sd)), fit(ZM.ztil_raw(z30, mu, sd))
    return float(np.linalg.norm(clip - base) / np.linalg.norm(raw - base))


def test_clip_known_answer_and_shift_ratio_honest():
    assert ok_ztil() and shift_ratio() <= 0.2


def test_mutation_no_clip_caught(monkeypatch):
    monkeypatch.setattr(ZM, "CLIP", None)
    assert not ok_ztil() and shift_ratio() > 0.2


def test_mutation_clip_before_standardization_caught(monkeypatch):
    monkeypatch.setattr(ZM, "ztil", lambda z, mu, sd: ZM.ztil_raw(np.clip(np.asarray(z, float), -ZM.CLIP, ZM.CLIP), mu, sd))
    assert not ok_ztil()


# ---------- прогноз по головам ----------
def heads_world():
    V, zt, z = world(300, 2)
    rng = np.random.default_rng(5)
    st = ORIG["std"](z, np.arange(300))
    zts = [zt] + [ORIG["ztil"](z + 0.3 * rng.normal(size=z.shape), *st) for _ in range(4)]
    return V, zts, rng


def ok_predict() -> bool:
    V, zts, rng = heads_world()
    heads = [np.concatenate([th_ck(rng), 0.3 * rng.normal(size=4)]) for _ in range(5)]
    ls = [ORIG["head"](h, V, q, "CKZ").numpy() for h, q in zip(heads, zts)]
    return bool(np.allclose(ZM.predict_lnc(heads, V, zts), np.mean(ls, axis=0), rtol=1e-12, atol=1e-12))


def ok_predict_bitwise() -> bool:
    """θ_Z = 0: головы различаются лишь z̃, добавка = 0 ⇒ ln C = голова 0 побитно."""
    V, zts, rng = heads_world()
    h0 = np.concatenate([th_ck(rng), np.zeros(4)])
    return bool(np.array_equal(ZM.predict_lnc([h0] * 5, V, zts), ORIG["head"](h0, V, zts[0], "CKZ").numpy()))


def test_predict_honest():
    assert ok_predict() and ok_predict_bitwise()


def test_mutation_mean_z_over_seeds_instead_of_heads_caught(monkeypatch):
    monkeypatch.setattr(ZM, "predict_lnc", lambda heads, V, zts, var=None: ORIG["head"](heads[0], V, np.mean(zts, axis=0), "CKZ").numpy())
    assert not ok_predict()


def test_mutation_np_mean_instead_of_head0_plus_deviations_caught(monkeypatch):
    monkeypatch.setattr(ZM, "predict_lnc", lambda heads, V, zts, var=None: np.mean([ORIG["head"](h, V, q, "CKZ").numpy() for h, q in zip(heads, zts)], axis=0))
    assert ok_predict() and not ok_predict_bitwise()      # значение то же с точностью до округления, но вложение θ_Z = 0 уже не побитно


# ---------- η после логарифма ----------
def ok_form() -> bool:
    V, zt, _ = world(300, 3)
    rng = np.random.default_rng(1)
    th, thz = th_ck(rng), 0.3 * rng.normal(size=4)
    ref = torch.log(V.c8 * CM.forward(torch.as_tensor(th), V, "CK")["M"]).numpy() + ZM.eta_of(zt, torch.as_tensor(thz)).numpy()
    return bool(np.array_equal(ZM.lnc_head(np.concatenate([th, thz]), V, zt, "CKZ").numpy(), ref))


def test_mutation_eta_inside_log_caught_only_when_theta_z_nonzero(monkeypatch):
    assert ok_form()
    V, zt, _ = world(300, 3)
    th = th_ck(np.random.default_rng(1))

    def inside(v, V_, zt_, var):
        v = torch.as_tensor(v, dtype=torch.float64)
        return torch.log(V_.c8 * CM.forward(v[:6], V_, "CK")["M"] * torch.exp(ORIG["head"].__globals__["eta_of"](zt_, v[6:])))
    monkeypatch.setattr(ZM, "lnc_head", inside)
    assert not ok_form()
    assert np.array_equal(ZM.lnc_head(np.concatenate([th, np.zeros(4)]), V, zt, "CKZ").numpy(), ORIG["head"](np.concatenate([th, np.zeros(4)]), V, zt, "CKZ").numpy())   # при θ_Z = 0 вложение цело — мутант виден только при θ_Z ≠ 0

