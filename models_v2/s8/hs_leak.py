"""Ж-212 (G14): главный тест обрезки. Прогноз на d не меняется (побитно), если ВСЕ данные после 18:50 d испорчены: строки F > d (RV, признаки, ML-ряд) домножены на случайные множители,
PIT HAR-J и блоки признаков пересчитаны из испорченного мира. Одна дата = одно переобучение на парах t+1 <= d."""
from __future__ import annotations

import dataclasses

import numpy as np

from models_v2.s4 import rows
from models_v2.s5.b1 import designs
from models_v2.s6.lgbm import job
from models_v2.s6.xgb import c_calc
from models_v2.s8 import hs_bfit, hs_ctx, hs_feat


def corrupt(F, ml: dict, d: int, seed: int = 7) -> tuple:
    """(F', ml'): строки > d умножены на lognormal-множители (RV, X, ML); до d (включительно) — без изменений."""
    rng = np.random.default_rng(seed)
    n = len(F.rv)
    m = np.ones(n)
    m[d + 1:] = np.exp(rng.normal(0, 1.0, n - d - 1))
    X = F.X * m[:, None]
    X[:d + 1] = F.X[:d + 1]
    rv = F.rv * m
    Fc = dataclasses.replace(F, X=X, rv=rv) if dataclasses.is_dataclass(F) else _clone(F, X, rv)
    return Fc, {k: v * m for k, v in ml.items()}


def _clone(F, X, rv):
    import copy
    c = copy.copy(F)
    c.X, c.rv, c.lnrv = X, rv, np.log(rv)
    return c


def forecast_xgb(w: dict, extra, d: int, seed: int = 42) -> float:
    sp = hs_bfit.xgb_spec(w, extra)
    c_calc.init(w["F"], sp, hs_bfit.xgb_hpy(w), w["eps"], seed, "l2")
    _, r = c_calc.block_job(d)
    return float(designs.sigma2(sp, np.array([d]), np.array([r["f"]]), np.array([r["s2"]]), w["eps"])[0][0])


def forecast_lgbm(w: dict, extra, d: int, hp_y: dict | None = None) -> float:
    sp, F = hs_bfit.lgbm_spec(w, extra), w["F"]
    hp = (hp_y or hs_bfit.lgbm_inh(w))[rows.year_of(F, d)]
    r = job.day_fc(F, sp, hp, d)
    return float(job.fitl.sigma2(sp.kind, np.array([d]), np.array([r["f"]]), np.array([r["s2"]]), sp.off, np.exp(sp.off), w["eps"])[0][0])


def g_day(F, lb, gate, X, eps, d: int, lib: str, mu: float, mono) -> float:
    """g = c0 + g_tree на дату d (C): одно переобучение на парах t+1 <= d."""
    from models_v2.s4 import fit
    from models_v2.s8 import hs_cfit
    rvn = np.append(F.rv[1:], np.nan)
    y = np.where(np.isfinite(lb), np.maximum(rvn, eps), np.nan)
    idx = fit.pair_index(F.seg, gate, y, (0, 1, 2, 3), d)
    mono = tuple(mono) if len(mono) else (0,) * X.shape[1]
    gt, c0 = hs_cfit.fit_day(X, y, lb, idx, d, lib, mu, mono)
    return gt + c0


def trunc_c(ctx: dict, build_X, lib: str, mu: float, mono, d: int) -> float:
    """Обрезка для C: build_X(F, lb) -> матрица признаков корректора (n×k); max |g(d) исходный − g(d) из испорченного мира|. Ожидание — 0."""
    w, F, gate = ctx["w"], ctx["w"]["F"], hs_bfit.xgb_spec(ctx["w"], None).X
    a = g_day(F, ctx["lb"], gate, build_X(F, ctx["lb"]), w["eps"], d, lib, mu, mono)
    Fc, _ = corrupt(F, ctx["ml"], d)
    lbc, _ = hs_feat.har_pit(Fc)
    return abs(a - g_day(Fc, lbc, gate, build_X(Fc, lbc), w["eps"], d, lib, mu, mono))


def trunc_d(ctx: dict, d: int, cols=None, k=None, depth: int = 1, xs_fn=None) -> float:
    """Обрезка для D: ln σ̂²(d) линейной модели с компонентами cols или формы по φ_k (+ xs_fn(F) — X_t для Dx) из исходного и испорченного мира. Ожидание — 0."""
    from models_v2.s4 import fit
    from models_v2.s8 import hs_dfit as D

    def one(F):
        Phi, y = D.phi_matrix(F), D.target(F, ctx["w"]["eps"])
        if k is None:
            gam = D.lin_fit(Phi, y, fit.pair_index(F.seg, Phi, y, tuple(range(Phi.shape[1])), d), cols)
            return float(gam[0] + (gam[1:] * Phi[d, list(cols)]).sum())
        return D.nl_day(Phi, y, F.seg, xs_fn(F) if xs_fn else Phi[:, [k]], k, depth, d, np.zeros(3))[0]
    F = ctx["w"]["F"]
    return abs(one(F) - one(corrupt(F, ctx["ml"], d)[0]))


def trunc_diff(ctx: dict, fam: str, blocks: tuple, d: int, hp_y: dict | None = None) -> float:
    """max |σ̂²(d) исходный − σ̂²(d) из испорченного мира| / σ̂² для блока признаков `blocks` модели `fam`. Ожидание — 0."""
    w, F = ctx["w"], ctx["w"]["F"]
    fc = forecast_xgb if fam == "XGB" else forecast_lgbm
    kw = {} if fam == "XGB" else {"hp_y": hp_y}
    a = fc(w, hs_ctx.extra_for(F, ctx["lb"], ctx["beta"], ctx["ml"], fam, blocks)[0], d, **kw)
    Fc, mlc = corrupt(F, ctx["ml"], d)
    lbc, bc = hs_feat.har_pit(Fc)
    wc = dict(w, F=Fc)
    b = fc(wc, hs_ctx.extra_for(Fc, lbc, bc, mlc, fam, blocks)[0], d, **kw)
    return abs(a - b) / a
