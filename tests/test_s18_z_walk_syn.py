"""S-Z-IMPL-2 (Ж-258), тестировщик: проход `z_walk.fit_month` на синтетике с известным ответом — вложение Н2 (use_z=False ⇒ путь r_walk побитно), сигнал z, включение 630, пары без z (Д4),
дни прогноза берут z своего ряда, μ, σ — по ряду прохода (мутации: «сдвиг только пула», «μ, σ контроля по несдвинутому z», разное включение; spec разд. 8). Ничего не пишет."""
import copy

import numpy as np
import pytest

from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as RW
from models_v2.s18 import r_walk_syn as RS
from models_v2.s18 import z_model as ZM
from models_v2.s18 import z_walk as ZW

N, THZ = 1500, np.array([0.8, -0.6, 0.0, 0.0])


def zs_of(n=N, seed=0, scale=1.0, off=0.0):
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(n, 4))
    return z0, [off + scale * (z0 + 0.05 * rng.normal(size=(n, 4))) for _ in ZW.SEEDS]


def sig_ctx(seed=2, ths=None):
    """RV = F(θ_CK) · exp(z·θ_Z): у z есть сигнал; z̃ масштаб 1 (z0 нормальный)."""
    z0, zy = zs_of(seed=seed)
    base = RS.syn_ctx(N, seed, ths=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.3]) if ths is None else ths)
    D = copy.copy(base.D)
    D.rv = base.D.rv * np.exp(z0 @ THZ)
    return RW.Ctx("CK", D, base.lv, base.x, base.gap, base.dp), zy


def c8(ctx, a, b):
    return ctx.x["C"].to_numpy(float)[a:b]


def test_nest_use_z_false_equals_r_walk_bitwise():
    ctx, zy = sig_ctx()
    P, days = np.arange(900), np.arange(900, 921)
    r0, r1 = RW.fit_month(ctx, P, days, c8(ctx, 900, 921), "R"), ZW.fit_month(ctx, zy, P, days, c8(ctx, 900, 921), "CKZ", False)
    assert r0["lam"] > 0.0, "предусловие: у синтетики есть сигнал C-K"
    assert r1["active"] and r1["lam"] == r0["lam"] and r1["w"] == r0["w"] and r1["c"] == r0["c"] and r1["m"] == r0["m"]
    assert np.array_equal(r1["G"], r0["G"]) and np.array_equal(r1["theta"], r0["theta"]) and np.array_equal(r1["C"], r0["C"])
    assert (r1["thz"] == 0).all() and (r1["thsd"] == 0).all()


def test_nest_rz_use_z_false_is_cfg8_copy():
    ctx, zy = sig_ctx()
    r = ZW.fit_month(ctx, zy, np.arange(900), np.arange(900, 921), c8(ctx, 900, 921), "Z", False)
    assert r["active"] and r["lam"] == 0.0 and np.array_equal(r["C"], c8(ctx, 900, 921)) and (r["G"] == 0).all()


def test_mut_nest_changed_lambda_is_caught(monkeypatch):
    ctx, zy = sig_ctx()
    P, days = np.arange(900), np.arange(900, 921)
    r0, orig = RW.fit_month(ctx, P, days, c8(ctx, 900, 921), "R"), ZW.fit_heads
    monkeypatch.setattr(ZW, "fit_heads", lambda V, zts, var, use_z, lam, c, m: orig(V, zts, var, use_z, lam * 3.0, c, m))
    r1 = ZW.fit_month(ctx, zy, P, days, c8(ctx, 900, 921), "CKZ", False)
    assert not (np.array_equal(r1["G"], r0["G"]) and np.array_equal(r1["C"], r0["C"]))


def test_signal_in_z_is_found_with_right_sign():
    ctx, zy = sig_ctx()
    r = ZW.fit_month(ctx, zy, np.arange(900), np.arange(900, 921), c8(ctx, 900, 921), "Z")
    assert r["lam"] > 0.0 and r["thz"][0] > 0.3 and r["thz"][1] < -0.2 and abs(r["thz"][2]) < 0.2
    assert (r["thsd"][:6] == 0).all() and (r["thsd"][6:] > 0).all() and np.isfinite(r["n_clip"]) and r["z_ok"].all()


def test_ztilde_zero_gives_zero_thz_and_ck_path():
    ctx, zy = sig_ctx()
    P, days = np.arange(900), np.arange(900, 921)
    r0 = ZW.fit_month(ctx, zy, P, days, c8(ctx, 900, 921), "CKZ", False)
    r1 = ZW.fit_month(ctx, zy, P, days, c8(ctx, 900, 921), "CKZ", True, None, True)
    assert (r1["thz"] == 0).all() and np.isnan(r1["n_clip"]) and np.max(np.abs(r1["C"] / r0["C"] - 1)) < 1e-9


@pytest.mark.parametrize("var,nan_rows", [("Z", 0), ("CKZ", 0), ("CKZ", 5)])
def test_inclusion_630_for_all_three_series(var, nan_rows):
    """rz, rckz, rckzs: одно включение |P| ≥ 630 (после исключения пар без z): 629 — θ = 0 и cfg8 побитно, 630 — активна."""
    ctx, zy = sig_ctx(seed=5)
    zy = [z.copy() for z in zy]
    for z in zy:
        z[:nan_rows] = np.nan
    c0 = c8(ctx, 700, 710)
    lo = ZW.fit_month(ctx, zy, np.arange(629 + nan_rows), np.arange(700, 710), c0, var)
    hi = ZW.fit_month(ctx, zy, np.arange(630 + nan_rows), np.arange(700, 710), c0, var)
    assert not lo["active"] and np.array_equal(lo["C"], c0) and lo["n_excl"] == nan_rows and lo["n_train"] == 629
    assert hi["active"] and hi["n_excl"] == nan_rows and hi["n_train"] == 630


def _incl_ok(ctx, zy):
    return not ZW.fit_month(ctx, zy, np.arange(629), np.arange(700, 710), c8(ctx, 700, 710), "Z")["active"]


def test_mut_different_inclusion_is_caught(monkeypatch):
    ctx, zy = sig_ctx(seed=5)
    assert _incl_ok(ctx, zy)
    monkeypatch.setattr(RW, "MIN_POOL", 500)
    assert not _incl_ok(ctx, zy)


def _days_follow(ctx, zy, var="Z"):
    """прогноз дней зависит от z дней (строки days ряда прохода), а λ*, θ — нет."""
    P, days = np.arange(900), np.arange(900, 921)
    zb = [z.copy() for z in zy]
    for z in zb:
        z[days] += 2.0
    a, b = (ZW.fit_month(ctx, z, P, days, c8(ctx, 900, 921), var) for z in (zy, zb))
    return a["lam"] == b["lam"] and np.array_equal(a["thz"], b["thz"]) and not np.allclose(a["C"], b["C"])


def test_forecast_days_use_z_of_the_walk_series():
    ctx, zy = sig_ctx()
    assert _days_follow(ctx, zy)


def test_mut_shift_of_pool_only_days_from_other_series_is_caught(monkeypatch):
    ctx, zy = sig_ctx()
    orig = ZW.head_lnc

    def mut(ths, V, zts, var, use_z):     # дни прогноза (21 строка) берут z не из ряда прохода
        return orig(ths, V, [np.zeros_like(t) for t in zts] if len(zts[0]) < 100 else zts, var, use_z)
    assert _days_follow(ctx, zy)
    monkeypatch.setattr(ZW, "head_lnc", mut)
    assert not _days_follow(ctx, zy)


def test_nan_z_on_forecast_day_falls_back_to_reference_c():
    ctx, zy = sig_ctx()
    P, days = np.arange(900), np.arange(900, 921)
    zb, ref = [z.copy() for z in zy], c8(ctx, 900, 921) * 1.05
    zb[2][905, 1] = np.nan
    r = ZW.fit_month(ctx, zb, P, days, c8(ctx, 900, 921), "CKZ", True, ref)
    assert not r["z_ok"][5] and r["z_ok"].sum() == 20 and r["C"][5] == ref[5] and np.isfinite(r["C"]).all()


def _captured(monkeypatch, zs):
    seen, orig = [], ZM.fit_head
    monkeypatch.setattr(ZW.ZM, "fit_head", lambda V, zt, *a, **k: (seen.append(np.asarray(zt)), orig(V, zt, *a, **k))[1])
    ctx, _ = sig_ctx()     # RV сигнальный по z0; ряд прохода zs — со сдвигом и масштабом
    ZW.fit_month(ctx, zs, np.arange(900), np.arange(900, 921), c8(ctx, 900, 921), "Z")
    return seen


def test_mu_sigma_by_walk_series_and_mutation_unshifted_stats_is_caught(monkeypatch):
    z0, zs = zs_of(scale=2.0, off=5.0)
    ok = lambda seen: all(np.abs(s.mean(0)).max() < 0.05 and np.abs(s.std(0) - 1).max() < 0.1 for s in seen)
    assert ok(_captured(monkeypatch, zs))
    monkeypatch.setattr(ZW, "_zts", lambda zy, T, zero=False: [ZM.ztil(z, *ZM.std_stats(z0, T)) for z in zy])
    assert not ok(_captured(monkeypatch, zs))
