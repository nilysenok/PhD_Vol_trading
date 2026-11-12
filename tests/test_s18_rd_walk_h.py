"""S-RCD-IMPL-2 (Ж-259), тестировщик: синтетический контекст и детекторы прохода `rd_walk`. Детектор — список замечаний: на честном коде пуст, на мутанте — нет. Без сети, боевых данных и записи на диск."""
import contextlib
import math

import numpy as np
import pandas as pd
import torch

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as CM
from models_v2.s18 import cm_walk as CW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_model_chk as MC
from models_v2.s18 import rd_walk as RD
from models_v2.s18 import rd_walk_pit as RP
import test_s18_rd_model_h as H

INF = float("inf")
FORM = {"rcd": "CD", "cdf": "CD", "ckf": "CK"}
STRONG = np.array([0.1, 0.3, 0.2, 0.3, 0.3, 0.2])


@contextlib.contextmanager
def patched(obj, **kw):
    old = {k: getattr(obj, k) for k in kw}
    for k, v in kw.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def ctx_of(D, lv, train_c: bool = False):
    n = len(lv.pi)
    d = pd.bdate_range("2018-01-01", periods=n + 1)
    x = pd.DataFrame(dict(d=d[:-1], t_plus=d[1:], C=D.c8, rv=D.rv))
    lt = RL.Lv(np.ones(n), np.zeros(n), np.zeros(n), np.ones(n)) if train_c else None
    return W.Ctx("CK", D, lv, x, None, lv_train=lt)


def ctx_syn(n: int = 1000, seed: int = 0, th=H.ZERO, train_c: bool = False, **kw):
    return ctx_of(*H.syn(n, seed, th=th, **kw), train_c)


def month_at(ctx, ts: str) -> tuple:
    xd = pd.DatetimeIndex(ctx.x["d"])
    return next((a, b) for a, b in W.month_bounds(ctx) if xd[a] >= pd.Timestamp(ts))


def spy_problems() -> list:
    """Пул, переданный в обучение месяца, = пары t⁺ ≤ D (№ 164); форма = форме ряда (rcd, cdf — CD, ckf — CK)."""
    ctx, seen, bad = ctx_syn(900, 1), {}, []
    rec = lambda tag: (lambda c, form, P, *a, **k: seen.__setitem__(tag, (form, np.asarray(P))) or {})
    for var in RD.VARS:
        a, b = month_at(ctx, "2018-09-01" if var != "rcd" else "2018-09-01")
        with patched(RD, fit_month_r=rec("r"), fit_month_sc=rec("s")):
            RD.month(var, ctx, a, b)
        form, P = seen["r" if RD.VARS[var][1] == "R" else "s"]
        want = np.flatnonzero(ctx.x["t_plus"].to_numpy() <= ctx.x["d"].to_numpy()[a])
        bad += ["%s: пул ≠ t⁺ ≤ D (%d вместо %d)" % (var, len(P), len(want))] * (not np.array_equal(P, want))
        bad += ["%s: форма %s вместо %s" % (var, form, FORM[var])] * (form != FORM[var])
    return bad


def stub_env(ctx, d_of):
    """Подмена RM.fit / predict_lnc: ln F(θ̂_λ) = ln F(0) + d_of(λ, V). Возвращает (пары, записи обучения)."""
    rec = dict(calls=[], overlap=[])
    fit = lambda V, form, lam, c, m, mode="R", **k: rec["calls"].append((form, mode, c, m, np.asarray(V.sel))) or [np.array([lam])]

    def predict(th, VV, form):
        lam, pi = float(th[0][0]), VV.l_pi.numpy()
        rec["overlap"].append(bool(rec["calls"][-1][4].max() >= VV.sel.min()))
        return H.lnc_ref(torch.zeros(6, dtype=torch.float64), VV, form).numpy() + d_of(lam, VV) / pi
    return fit, predict, rec


def ell_ref(rv, lnf):
    return rv * np.exp(-lnf) + lnf - np.log(rv) - 1.0


D_WIN, D_LOSE = math.log(2.0), -0.5
CASES_R = [
    ({0.1: [1, 0, 0], 0.01: [0, 0, 0], 0.001: [0, 1, 0]}, 0.0), ({0.1: [1, 1, 0], 0.01: [1, 1, 1], 0.001: [1, 1, 1]}, 0.01), ({0.1: [0, 0, 0], 0.01: [0, 0, 0], 0.001: [0, 0, 0]}, 0.0),
    ({0.1: [1, 1, 1], 0.01: [0, 0, 0], 0.001: [0, 0, 0]}, 0.1), ({0.1: [0, 1, 1], 0.01: [1, 0, 1], 0.001: [1, 1, 0]}, 0.1), ({0.1: [1, 0, 0], 0.01: [1, 0, 0], 0.001: [1, 0, 0]}, 0.0)]


def rule_problems() -> list:
    """choose_r на заданных G: правило «≥ 2 из 3», ничья — больший λ, нет допущенных — θ = 0; T_b не пересекает V_b; G — обычная QLIKE (в V₃ выброс ×20)."""
    ctx, P, bad = ctx_syn(1000, 3), np.arange(800), []
    F0 = np.exp(ctx.lnf0)
    ctx.rv = 2.0 * F0
    ctx.rv[700] *= 20.0
    blk = [P[len(P) - 126 * (3 - b): len(P) - 126 * (2 - b)] for b in range(3)]
    for i, (wins, want) in enumerate(CASES_R):
        d_of = lambda lam, VV: D_WIN if wins[lam][next(k for k, v in enumerate(blk) if VV.sel[0] == v[0])] else D_LOSE
        fit, predict, rec = stub_env(ctx, d_of)
        with patched(RD.RM, fit=fit, predict_lnc=predict):
            lam, w, G = RD.choose_r(ctx, "CD", P)
        bad += ["случай %d: λ* %s вместо %s" % (i, lam, want)] * (lam != want)
        bad += ["случай %d: T_b пересекает V_b" % i] * any(rec["overlap"])
        bad += ["случай %d: форма/режим обучения %s" % (i, {c[:2] for c in rec["calls"]})] * ({c[:2] for c in rec["calls"]} != {("CD", "R")})
        for lam_k in G:
            for b, V in enumerate(blk):
                d = D_WIN if wins[lam_k][b] else D_LOSE
                x = ctx.rv[V] / F0[V]
                want_g = float(np.mean(x * (math.exp(-d) - 1.0) + d))
                bad += ["случай %d: G[%g][%d] = %.6e вместо %.6e" % (i, lam_k, b, G[lam_k][b], want_g)] * (abs(G[lam_k][b] - want_g) > 1e-10)
    return bad


def zero_problems() -> list:
    """Нулевой сигнал: θ = 0 в сетке ⇒ λ* = 0, θ = 0, C = cfg8 побитно (прогноз = Ctx.lnf0, G = 0 точно)."""
    ctx = ctx_syn(1000, 4, sig=0.0, jump=0.0)
    c0 = ctx.x["C"].to_numpy(float)[800:820]
    r = RD.fit_month_r(ctx, "CD", np.arange(800), np.arange(800, 820), c0)
    bad = ["λ* = %s вместо 0" % r["lam"]] * bool(r["lam"] != 0.0) + ["θ ≠ 0"] * bool(np.abs(r["theta"]).max() > 0)
    return bad + ["C ≠ cfg8 побитно"] * (not np.array_equal(r["C"], c0)) + ["не активен"] * (not r["active"])


def incl_problems() -> list:
    """S-RCD: |P| = 629 — не активен, 630 — активен; на 2018-07-02 (130 пар) не активен. S-CDF / S-CKF: до 2018-07-02 — нет, с него — да, |P| < 252 ⇒ λ = 1e-2 без валидации."""
    ctx, bad = ctx_syn(900, 5), []
    c0 = ctx.x["C"].to_numpy(float)[700:710]
    act = [RD.fit_month_r(ctx, "CD", np.arange(n), np.arange(700, 710), c0)["active"] for n in (629, 630)]
    bad += ["rcd |P| = 629 / 630: %s" % act] * (act != [False, True])
    a, b = month_at(ctx, "2018-07-02")
    pa, pb = month_at(ctx, "2018-06-01")
    try:
        bad += ["rcd активен на 2018-07-02"] * RD.month("rcd", ctx, a, b)["active"]
    except Exception as e:     # мутация включения (пул < 630 допущен) может упасть на пустом T — это тоже замечание
        bad.append("rcd на 2018-07-02: падение %r" % e)
    for var in ("cdf", "ckf"):
        r0, r1 = RD.month(var, ctx, pa, pb), RD.month(var, ctx, a, b)
        bad += ["%s: до 2018-07-02 активен, λ = %s" % (var, r0["lam"])] * bool(r0["active"] or not np.array_equal(r0["C"], ctx.x["C"].to_numpy(float)[pa:pb]))
        bad += ["%s: с 2018-07-02 не активен, или λ = %s ≠ 1e-2, или есть валидация" % (var, r1["lam"])] * bool(not r1["active"] or r1["lam"] != CW.LAM_DEF or not np.isnan(r1["val"]).all())
    return bad


SC_CASES = [((0.0, 0.0, 0.0), 1e-2), ((0.0, 0.3, 0.9), 1e-4), ((0.3, 0.0, 0.9), 1e-3), ((0.0, 0.0, 0.9), 1e-3), ((0.9, 0.3, 0.0), 1e-2), ((0.3, 0.3, 0.3), 1e-2)]


def sc_rule_problems() -> list:
    """choose_sc: λ = argmin QLIKE уровня F на последних 126 парах, ничья — больший λ, val — независимый расчёт обычной QLIKE; |P| < 252 ⇒ λ = 1e-2, обучения нет."""
    ctx, P, bad = ctx_syn(900, 6), np.arange(500), []
    for i, (ds, want) in enumerate(SC_CASES):
        fit, predict, rec = stub_env(ctx, lambda lam, VV: dict(zip((1e-4, 1e-3, 1e-2), ds)).get(lam, 0.9))
        with patched(RD.RM, fit=fit, predict_lnc=predict):
            lam, val = RD.choose_sc(ctx, "CD", P)
        Vv = P[-126:]
        lnf0 = H.lnf_ref(H.lnc_ref(torch.zeros(6, dtype=torch.float64), ctx.view(Vv), "CD").numpy(), ctx.lv, Vv)
        exp = [float(np.mean(ell_ref(ctx.D.rv[Vv], lnf0 + d))) for d in ds]
        bad += ["случай %d: λ = %g вместо %g" % (i, lam, want)] * (lam != want)
        bad += ["случай %d: val %s ≠ независимому %s" % (i, val, exp)] * (len(val) != 3 or np.abs(np.array(val) - exp).max() > 1e-12)
        bad += ["случай %d: режим обучения %s" % (i, {c[1:4] for c in rec["calls"]})] * ({c[1:4] for c in rec["calls"]} != {("plain", INF, 1.0)})
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("обучение при |P| < 252"))
    try:
        with patched(RD.RM, fit=boom):
            lam, val = RD.choose_sc(ctx, "CD", np.arange(200))
    except AssertionError as e:
        return bad + ["|P| = 200: %s" % e]
    return bad + ["|P| = 200: λ = %s, val %s" % (lam, val)] * bool(lam != 1e-2 or not np.isnan(val).all())
