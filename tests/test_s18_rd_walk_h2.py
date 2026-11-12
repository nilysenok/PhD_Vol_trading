"""S-RCD-IMPL-2 (Ж-259), тестировщик: детекторы второй группы — аргументы обучения, форма C в проходе, уровень потери и валидации, PIT месяца. Только расчёт на синтетике."""
import numpy as np
import pandas as pd
import torch

from models_v2.s18 import cm_model as CM
from models_v2.s18 import rd_model_chk as MC
from models_v2.s18 import rd_walk as RD
from models_v2.s18 import rd_walk_pit as RP
import test_s18_rd_model_h as H
import test_s18_rd_walk_h as K
from test_s18_rd_walk_h import FORM, INF, STRONG, ctx_of, ctx_syn, month_at, patched


class Rec:
    """Прокси `rd_model`: пишет аргументы обучения (после мутации, если она внешняя); alter(form, lam, c, m, mode) — подложенная мутация аргументов."""

    def __init__(self, real, calls, alter=None):
        self.real, self.calls, self.alter = real, calls, alter

    def _go(self, name, V, form, lam, c, m, mode, a, k):
        form, lam, c, m, mode = (form, lam, c, m, mode) if self.alter is None else self.alter(form, lam, c, m, mode)
        self.calls.append((form, lam, c, m, mode))
        return getattr(self.real, name)(V, form, lam, c, m, mode, *a, **k)

    def fit(self, V, form, lam, c, m, mode="R", *a, **k):
        return self._go("fit", V, form, lam, c, m, mode, a, k)

    def fit_full(self, V, form, lam, c, m, mode="R", *a, **k):
        return self._go("fit_full", V, form, lam, c, m, mode, a, k)

    def __getattr__(self, k):
        return getattr(self.real, k)


def args_problems(alter=None) -> list:
    """Реальный месяц: S-CDF / S-CKF обучаются ТОЛЬКО plain, c = ∞, m = 1, λ ∈ {1e-4; 1e-3; 1e-2} (4 подгонки: 3 валидационных + финал), нулевого λ и w, G нет; S-RCD — режим R,
    c, m конечные; C = cfg8·𝓜_форма(θ) (SC — побитно, R — до 1e-12)."""
    ctx, bad = ctx_syn(900, 7, th=STRONG), []
    for var, ts in (("cdf", "2019-05-01"), ("ckf", "2019-05-01"), ("rcd", "2020-09-01")):
        a, b = month_at(ctx, ts)
        calls = []
        with patched(RD, RM=Rec(RD.RM, calls, alter)):
            r = RD.month(var, ctx, a, b)
        form, c8, c = FORM[var], ctx.x["C"].to_numpy(float)[a:b], r["C"]
        if RD.VARS[var][1] == "SC":
            bad += ["%s: аргументы обучения %s" % (var, sorted({(x[0], x[2], x[3], x[4]) for x in calls}))] * bool({(x[0], x[2], x[3], x[4]) for x in calls} != {(form, INF, 1.0, "plain")} or len(calls) != 4)
            bad += ["%s: λ %s, c %s, m %s, w %s" % (var, r["lam"], r["c"], r["m"], r["w"])] * bool(r["lam"] not in CM.LAMS or r["c"] != INF or r["m"] != 1.0 or not np.isnan(r["w"]).all() or not np.isnan(r["G"]).all())
            want = c8 * CM.forward(torch.as_tensor(r["theta"]), ctx.view(np.arange(a, b)), form)["M"].numpy()
            bad += ["%s: C ≠ cfg8·𝓜_%s(θ) побитно" % (var, form)] * (not np.array_equal(c, want))
        else:
            bad += ["rcd: режимы %s, формы %s" % ({x[4] for x in calls}, {x[0] for x in calls})] * bool({x[4] for x in calls} != {"R"} or {x[0] for x in calls} != {form} or not np.isfinite([x[2] for x in calls]).all())
            bad += ["rcd не активен или c, m не конечны: %s, %s, %s" % (r["active"], r["c"], r["m"])] * bool(not r["active"] or not np.isfinite([r["c"], r["m"]]).all())
            want = np.exp(H.RM.predict_lnc([r["theta"]], ctx.view(ctx.dp[a:b]), form)) if r["lam"] > 0 else c8
            bad += ["rcd: C ≠ cfg8·𝓜_CD(θ)"] * bool(np.abs(c / want - 1.0).max() > 1e-12)
    return bad


def level_val_problems() -> list:
    """Известный ответ (спецификация 8.3): на валидационных парах π = 0 ⇒ ln F не зависит от C ⇒ val равны при всех λ ⇒ ничья ⇒ λ = 1e-2. Валидация уровня C дала бы разные val."""
    D, lv = H.syn(900, 8, th=STRONG, pi=0.6, jit=0.0, lnb_shift=0.3)
    a, b = month_at(ctx_of(D, lv), "2019-09-02")
    lv.pi = lv.pi.copy()
    lv.pi[a - 126:a] = 0.0
    ctx, bad = ctx_of(D, lv, train_c=True), []
    for var in ("cdf", "ckf"):
        r = RD.month(var, ctx, a, b)
        bad += ["%s: val %s, λ = %s (при π = 0 на валидации все val равны, λ = 1e-2)" % (var, r["val"], r["lam"])] * bool(len(set(r["val"])) != 1 or r["lam"] != 1e-2)
    return bad


def level_train_problems() -> list:
    """S-CKF = контроль уровня F: проекционный градиент потери уровня F в сохранённом θ ≤ 1e-4, потери уровня C ≥ 1e-3 (W.G5 на синтетике, где уровни различаются)."""
    ctx, bad = ctx_syn(900, 9, th=STRONG, pi=0.5, lnb_shift=0.4, train_c=True), []
    a, b = month_at(ctx, "2019-09-02")
    P = RP.pool_of("ckf", ctx, a)
    r = RD.month("ckf", ctx, a, b)
    gf = RP.proj_grad(r["theta"], ctx.view(P), "CK", r["lam"], INF, 1.0, "plain")
    gc = RP.proj_grad(r["theta"], ctx.view_train(P), "CK", r["lam"], INF, 1.0, "plain")
    return ["ckf: градиент уровня F %.2e, уровня C %.2e" % (gf, gc)] * bool(gf > 1e-4 or gc < 1e-3)


def pit_problems() -> list:
    """Порча RV пар t⁺ > D, строк признаков после месяца и обоих не меняет λ, w, θ̂, C, val месяца (rcd, cdf, ckf)."""
    ctx, n, bad = ctx_syn(900, 10, th=STRONG), 900, []
    for var, ts in (("rcd", "2020-09-01"), ("cdf", "2019-09-02"), ("ckf", "2019-09-02")):
        a, b = month_at(ctx, ts)
        P, out, late = np.flatnonzero(ctx.x["t_plus"].to_numpy() <= ctx.x["d"].to_numpy()[a]), np.ones(n, bool), np.arange(n) >= b     # пул — независимо от YA (иначе мутация «закрыто + 1» прячет себя)
        out[P] = False
        Dc2, lc2 = MC.corrupt(ctx.D, ctx.lv, late)
        base = RD.month(var, ctx, a, b)
        r = [RD.month(var, RP.with_data(ctx, d_, l_), a, b) for d_, l_ in ((RP.spoil_rv(ctx.D, out), ctx.lv), (Dc2, lc2), (RP.spoil_rv(Dc2, out), lc2))]
        bad += ["%s: порча будущего изменила месяц %s: %s" % (var, ts, [RP.same(base, x) for x in r])] * (not all(RP.same(base, x) for x in r))
    return bad
