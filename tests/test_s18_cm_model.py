"""S-C-IMPL-2 (тестировщик): вложение (нулевые параметры ⇒ 𝓜 = 1 побитно; γ = 1, θ = 0 ⇒ C_K = K7-класс), тождество суммы с uncovered,
неотрицательность компонент, μ = 0,5 только у C-KA, штраф на все 6 параметров, градиенты, общий информационный набор C-D и C-K, итог по минимальному перезапуску.
Мутации — подменой функции через monkeypatch (№ 43); реальные файлы охраняет autouse."""
import numpy as np
import pytest
import torch

from models_v2.s18 import cm_model as M
from s18_cm_guard import guard_real_files  # noqa: F401  (autouse)
from s18_cm_synth import make_data, params

D = make_data(80, seed=1)
V = D.view(np.arange(D.n))
Z = torch.zeros(M.NP, dtype=torch.float64)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


def test_constants_match_spec():
    assert M.NP == 6 and M.LAMS == (1e-4, 1e-3, 1e-2) and tuple(M.SEEDS) == (0, 1, 2, 3, 4) and M.SEED_SD == 0.05
    assert M.MU == {"CD": 0, "CK": 0, "CKA": 0.5, "CKL": 0} and M.VARIANTS == ("CD", "CK", "CKA", "CKL", "C7")
    assert M.bounds("CK")[:2] == [(0.0, 1.0)] * 2 and M.bounds("CK")[2:] == [(-3.0, 3.0)] * 4
    assert M.bounds("CD")[0] == (-3.0, 3.0) and M.bounds("CD")[1:3] == [(0.0, 1.0)] * 2 and M.bounds("CD")[3:] == [(-3.0, 3.0)] * 3


def test_zero_parameters_give_M_exactly_one_and_C_equals_cfg8_bitwise():
    for var in M.TRAINED:
        m = M.forward(Z, V, var)["M"].numpy()
        assert (m == 1.0).all(), var
        assert ((V.c8 * M.forward(Z, V, var)["M"]).numpy() == V.c8.numpy()).all(), var


def test_mutation_exp_instead_of_expm1_form_breaks_bitwise_nesting(monkeypatch):
    def forward_exp(v, Vv, var):
        ec, ep = v[0] * Vv.xc + v[2] + v[4] * Vv.zc, v[1] * Vv.xp + v[3] + v[5] * Vv.zp
        return dict(M=Vv.sc * torch.exp(ec) + Vv.sp * torch.exp(ep) + Vv.su, Fc=None, Fp=None, Fu=None)
    assert (M.forward(Z, V, "CK")["M"].numpy() == 1.0).all()
    monkeypatch.setattr(M, "forward", forward_exp)
    assert not (M.forward(Z, V, "CK")["M"].numpy() == 1.0).all()           # тест вложения выше падает на этом мутанте


def test_gamma_one_theta_zero_gives_k7_class_value():
    v = torch.zeros(M.NP, dtype=torch.float64)
    v[:2] = 1.0
    ck, c7 = V.c8 * M.forward(v, V, "CK")["M"], V.c8 * M.forward(Z, V, "C7")["M"]
    assert rel(ck.numpy(), c7.numpy()) <= 1e-13
    assert rel(c7.numpy(), (V.Hc + V.Hp + V.su * V.c8).numpy()) <= 1e-13


def sum_identity(fwd, v):
    f = fwd(v, V, "CK")
    return rel((f["Fc"] + f["Fp"] + f["Fu"]).numpy(), (V.c8 * f["M"]).numpy())


def test_components_sum_with_uncovered_equals_C_and_mutation_without_uncovered_caught(monkeypatch):
    for s in range(5):
        assert sum_identity(M.forward, params(s)) <= 1e-13
    assert (V.su > 0).all()
    orig = M.forward
    monkeypatch.setattr(M, "forward", lambda v, Vv, var: {**orig(v, Vv, var), "Fu": orig(v, Vv, var)["Fu"] * 0})
    assert sum_identity(M.forward, params(0)) > 1e-3
    monkeypatch.undo()
    f7 = M.forward(Z, V, "C7")                                         # C7: сумма с u; мутант без u расходится
    assert rel((V.Hc + V.Hp + V.su * V.c8).numpy(), (V.c8 * f7["M"]).numpy()) <= 1e-13
    assert rel((V.Hc + V.Hp).numpy(), (V.c8 * f7["M"]).numpy()) > 1e-2


def test_label_is_full_rv_not_c_plus_p_in_synthetic_with_uncovered():
    """Синтетика (в): ŝ_u = 0,3; метка суммы = rv целиком. Мутант «rv = c + p» меняет потерю."""
    d = make_data(200, seed=2, theta=[0.4, 0.6, 0.1, -0.1, -0.3, 0.2], su_range=(0.3, 0.3), sc_range=(0.2, 0.4))
    Vd = d.view(np.arange(d.n))
    full = float(M.qlike_sum(Z + 0.1, Vd, "CK"))
    mut = M.qlike(Vd.rv * (Vd.sc + Vd.sp), Vd.c8 * M.forward(Z + 0.1, Vd, "CK")["M"]).mean()
    assert abs(full - float(mut)) > 1e-2


def test_components_nonnegative_at_extreme_bounds_and_linear_mutant_caught():
    for sgn in (-1.0, 1.0):
        v = torch.as_tensor([1.0, 1.0, 3 * sgn, 3 * sgn, 3 * sgn, 3 * sgn], dtype=torch.float64)
        f = M.forward(v, V, "CK")
        assert (f["Fc"] > 0).all() and (f["Fp"] > 0).all() and (f["Fu"] > 0).all() and (f["M"] > 0).all()
    v = torch.as_tensor([0.0, 0.0, -3.0, -3.0, 0.0, 0.0], dtype=torch.float64)
    linear_Fc = V.sc * V.c8 * (1.0 + (v[2]))                         # мутант: параметризация без exp
    assert (linear_Fc < 0).any() is torch.tensor(True) or bool((linear_Fc < 0).any())


def test_mu_is_half_only_in_cka_and_loss_known_answer():
    lam = 1e-3
    v = params(3, 0.3)
    base = float(M.qlike_sum(v, V, "CK") + M.penalty(v, lam))
    assert float(M.loss(v, V, "CK", lam)) == base                       # μ = 0 в S-CK (мутант: μ ≠ 0 протекла)
    f = M.forward(v, V, "CKA")
    comp = 0.5 * 0.5 * (M.qlike(V.labc, f["Fc"]) + M.qlike(V.labp, f["Fp"])).mean()
    assert abs(float(M.loss(v, V, "CKA", lam)) - (float(M.qlike_sum(v, V, "CKA")) + float(comp) + lam * float((v ** 2).sum()))) < 1e-14
    assert float(M.loss(v, V, "CKA", lam)) != float(M.loss(v, V, "CKA", lam, mu=0.0))
    assert float(M.loss(v, V, "CKL", lam)) == float(M.qlike_sum(v, V, "CKL") + M.penalty(v, lam))


def test_penalty_covers_all_six_parameters_including_gamma_and_theta0():
    for i in range(M.NP):
        e = torch.zeros(M.NP, dtype=torch.float64)
        e[i] = 0.7
        assert abs(float(M.penalty(e, 1e-2)) - 1e-2 * 0.49) < 1e-15, M.NAMES["CK"][i]


def num_grad(var, v, h=1e-6):
    g = np.zeros(M.NP)
    for i in range(M.NP):
        e = np.zeros(M.NP)
        e[i] = h
        g[i] = (float(M.loss(torch.as_tensor(v + e), V, var, 1e-3)) - float(M.loss(torch.as_tensor(v - e), V, var, 1e-3))) / (2 * h)
    return g


def auto_grad(var, v):
    t = torch.as_tensor(v).clone().requires_grad_(True)
    M.loss(t, V, var, 1e-3).backward()
    return t.grad.numpy()


def test_gradients_equal_central_differences_all_variants():
    for var in M.TRAINED:
        for s in range(2):
            v = params(10 + s, 0.4).numpy()
            a, n = auto_grad(var, v), num_grad(var, v)
            assert np.abs(a - n).max() / max(np.abs(n).max(), 1e-12) <= 1e-6, var


def test_gradient_at_zero_finite_and_nonzero():
    d = make_data(200, seed=6, theta=[0.4, 0.6, 0.1, -0.1, -0.3, 0.2])       # rv ≠ cfg8: (1 − RV/C) ≠ 0
    t = torch.zeros(M.NP, dtype=torch.float64, requires_grad=True)
    for var in ("CD", "CK", "CKA"):
        t.grad = None
        M.loss(t, d.view(np.arange(d.n)), var, 1e-3).backward()
        g = t.grad.numpy()
        assert np.isfinite(g).all() and np.abs(g).max() > 0 and (np.abs(g) > 0).sum() >= 4, var


def test_mutation_detach_in_gamma_term_caught_by_gradient_check(monkeypatch):
    v = params(11, 0.4).numpy()
    assert np.abs(auto_grad("CK", v) - num_grad("CK", v)).max() < 1e-6
    orig = M.forward

    def detached(vv, Vv, var):
        return orig(torch.cat([vv[:1].detach(), vv[1:]]), Vv, var)
    monkeypatch.setattr(M, "forward", detached)
    assert np.abs(auto_grad("CK", v) - num_grad("CK", v)).max() > 1e-4


class Rec:
    """Обёртка View: записывает, какие поля читает forward."""
    def __init__(self, V):
        object.__setattr__(self, "_V", V)
        object.__setattr__(self, "seen", set())

    def __getattr__(self, k):
        self.seen.add(k)
        return getattr(self._V, k)


SOURCE = {"xc": {"Hc", "sc", "c8"}, "xp": {"Hp", "sp", "c8"}, "logit": {"sc"}, "zc": {"zc"}, "zp": {"zp"}, "sc": {"sc"}, "sp": {"sp"}, "su": {"sc", "sp"}, "c8": {"c8"}, "labc": {"labc"}, "labp": {"labp"}}   # ŝ_u = 1 − ŝ_c − ŝ_p: не новая информация


def info_set(var, fwd=None):
    r = Rec(V)
    (fwd or M.forward)(params(5), r, var)
    unknown = r.seen - set(SOURCE) - {"rv"}                              # rv — метка, не вход
    assert not unknown, unknown
    return set().union(*(SOURCE[k] for k in r.seen if k in SOURCE))


def test_cd_and_ck_use_the_same_information_set_and_mutation_extra_input_caught(monkeypatch):
    assert info_set("CD") == info_set("CK") == {"Hc", "Hp", "sc", "sp", "zc", "zp", "c8"}
    assert info_set("CKA") == info_set("CKL") == info_set("CK")
    orig = M.forward

    def cd_with_su(v, Vv, var):
        out = orig(v, Vv, var)
        return {**out, "M": out["M"] * (1 + 0 * Vv.labc)} if var == "CD" else out
    assert info_set("CD", cd_with_su) != info_set("CK")


def synth_fit(var, seed=0, **kw):
    d = make_data(1200, seed=seed, theta=[0.4, 0.6, 0.1, -0.1, -0.3, 0.2], shape=200.0)
    return M.fit(d.view(np.arange(d.n)), var, 1e-6, **kw)


def test_synthetic_known_answer_recovers_theta_and_beats_start():
    r = synth_fit("CK")
    assert np.abs(r["theta"] - np.array([0.4, 0.6, 0.1, -0.1, -0.3, 0.2])).max() <= 0.1
    d = make_data(1200, seed=0, theta=[0.4, 0.6, 0.1, -0.1, -0.3, 0.2], shape=200.0)
    assert r["loss"] < float(M.loss(Z, d.view(np.arange(d.n)), "CK", 1e-6))


def test_null_signal_theta_stays_near_zero():
    d = make_data(1200, seed=3, theta=np.zeros(6), shape=200.0)
    r = M.fit(d.view(np.arange(d.n)), "CK", 1e-2)
    assert np.abs(r["theta"]).max() < 0.05
    assert abs(r["loss"] - float(M.loss(Z, d.view(np.arange(d.n)), "CK", 1e-2))) < 1e-3


def test_fit_returns_minimum_restart_not_the_last_and_mutation_caught(monkeypatch):
    d = make_data(300, seed=4, theta=[0.4, 0.6, 0.1, -0.1, -0.3, 0.2])
    Vd = d.view(np.arange(d.n))
    r = M.fit(Vd, "CK", 1e-3, maxiter=2)
    assert r["loss"] == float(np.min(r["losses"])) and r["seed"] == int(np.argmin(r["losses"]))
    assert len(set(np.round(r["losses"], 12))) > 1                       # перезапуски различаются: тест различающий
    monkeypatch.setattr(M.np, "argmin", lambda a: len(a) - 1)
    assert M.fit(Vd, "CK", 1e-3, maxiter=2)["loss"] > r["loss"]           # мутант «последний» хуже минимального


def test_seed_zero_is_exact_zero_and_other_seeds_perturbed_within_bounds():
    assert (M._x0("CK", 0) == 0).all()
    for var in M.TRAINED:
        for s in (1, 2, 3, 4):
            x = M._x0(var, s)
            assert (x != 0).any() and all(lo <= xi <= hi for xi, (lo, hi) in zip(x, M.bounds(var)))
