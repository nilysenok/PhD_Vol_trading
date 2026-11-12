"""S-RCD-IMPL-2 (Ж-259; `S_RCD_SPEC.md`, разд. 3, 4, 8.3): помесячный проход S-RCD (форма C-D под процедурой R), S-CDF и S-CKF (контроли без R, обычная QLIKE уровня F), слои X1 и W5-G над новым C,
запись `data/regen/harj_struct/RD/{rcd,cdf,ckf}.parquet`. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Гипотеза выбрана после просмотра S-R, S-D, S-L, S-Z; результат исследовательский, не OOS.
Протокол "R" — `r_walk.choose` / `fit_month` с `rd_model` (форма — аргумент): сетка Λ = {0; 1e-1; 1e-2; 1e-3}, 3 блока × 126, 2 из 3, c = Q₀,₉₉, m, включение |P| ≥ 630; при θ̂ = 0 C = cfg8 копией (F = W5-G).
Протокол "SC" — `cm_walk.choose_lambda` / `fit_month`, потеря и валидация на уровне F (замороженные π, β): λ ∈ {1e-4; 1e-3; 1e-2}, |P| < 252 — 1e-2, валидация 126 последних закрытых пар, ничья — больший λ, включение 2018-07-02; c = ∞, m = 1.
`level="C"` — только для сверки Н3 (π := 1, ln B := 0, β·g := 0 в обучении и валидации).
Публичное API: `VARS`, `IDS`, `OUTS`, `RD_DIR`; `choose_r(ctx, form, P) -> (λ*, w, G)`, `fit_month_r(ctx, form, P, days, c0)`; `choose_sc(ctx, form, P, level) -> (λ, val)`, `fit_month_sc(ctx, form, P, days, c0, level)`;
`walk(var, ctx, positions=None, work=None, level="F") -> [(a, b, dict)]`; `assemble(var, ctx, parts) -> DataFrame` (как `r_walk.assemble`, θ-колонки `CM.NAMES[form]`; у "SC": w_*, G_* = NaN, c = ∞, m = 1, `val_1e-4/-3/-2`; у всех — `spread`); `run(var, write=True, out=RD_DIR)`.
Запуск: `python3 -W ignore -m models_v2.s18.rd_walk --run rcd|cdf|ckf | --check-nest | --check-pit | --check-level | --check`."""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401  (torch раньше lightgbm / xgboost — класс № 152)

from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_model as CM
from models_v2.s18 import cm_walk as CW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18 import rd_model as RM
from models_v2.s18 import rd_reg as RG
from models_v2.s18.r_walk_ctx import make_ctx

VARS = {"rcd": ("CD", "R"), "cdf": ("CD", "SC"), "ckf": ("CK", "SC")}
IDS = {"rcd": "S-RCD", "cdf": "S-CDF", "ckf": "S-CKF"}
OUTS = {v: v + ".parquet" for v in IDS}
RD_DIR = RG.RD_DIR
VK = ("1e-4", "1e-3", "1e-2")
NAN, INF = float("nan"), float("inf")


def _view(ctx, sel, level: str):
    return ctx.view_train(sel) if level == "C" else ctx.view(sel)


def choose_r(ctx, form: str, P) -> tuple:
    G = {lam: [] for lam in W.LAMS}
    for T, Vb in W.blocks(P):
        c, m = W.stats_of(ctx.x0, T, "R")
        VT, VV = ctx.view(T), ctx.view(Vb)
        base = float(RL.ell(ctx.rv[Vb], ctx.lnf0[Vb]).mean())
        for lam in W.LAMS:
            th = RM.fit(VT, form, lam, c, m, "R")
            lnf = RL.ln_f(torch.as_tensor(RM.predict_lnc(th, VV, form)), VV).numpy()
            G[lam].append(float(RL.ell(ctx.rv[Vb], lnf).mean()) - base)
    lam, w = W.decide(G, "blocks")
    return lam, w, G


def fit_month_r(ctx, form: str, P, days, c0) -> dict:
    out = dict(active=False, lam=NAN, w=(NAN,) * 3, G=np.full((3, W.NB), NAN), c=NAN, m=NAN, n_train=len(P), theta=np.zeros(CM.NP), C=np.asarray(c0, float).copy(), spread=NAN, val=(NAN,) * 3)
    if len(P) < W.MIN_POOL:
        return out
    lam, w, G = choose_r(ctx, form, P)
    c, m = W.stats_of(ctx.x0, P, "R")
    out.update(active=True, lam=lam, w=tuple(w[k] for k in W.LAMS), c=c, m=m)
    for i, k in enumerate(W.LAMS):
        out["G"][i, W.NB - len(G[k]):] = G[k]
    if lam > 0.0:
        th, sp = RM.fit_full(ctx.view(P), form, lam, c, m, "R")
        out.update(theta=np.mean(th, axis=0), C=np.exp(RM.predict_lnc(th, ctx.view(days), form)), spread=sp)
    return out


def choose_sc(ctx, form: str, P, level: str = "F") -> tuple:
    if len(P) < CW.MIN_VALID:
        return CW.LAM_DEF, (NAN,) * 3
    Vt, Vv, best, val = _view(ctx, P[:-CW.VAL], level), _view(ctx, P[-CW.VAL:], level), None, []
    for lam in CM.LAMS:
        th = RM.fit(Vt, form, lam, INF, 1.0, "plain")
        q = float(RL.ell(Vv.rv.numpy(), RL.ln_f(torch.as_tensor(RM.predict_lnc(th, Vv, form)), Vv).numpy()).mean())
        val.append(q)
        if best is None or q <= best[0]:
            best = (q, lam)
    return best[1], tuple(val)


def fit_month_sc(ctx, form: str, P, days, c0, level: str = "F") -> dict:
    P = np.asarray(P)
    lam, val = choose_sc(ctx, form, P, level)
    th, sp = RM.fit_full(_view(ctx, P, level), form, lam, INF, 1.0, "plain")
    C = np.asarray(c0, float) * CM.forward(torch.as_tensor(th[0]), ctx.view(days), form)["M"].numpy()
    return dict(active=True, lam=lam, w=(NAN,) * 3, G=np.full((3, W.NB), NAN), c=INF, m=1.0, n_train=len(P), theta=th[0], C=C, spread=sp, val=val)


def zero_month_sc(P, c0) -> dict:
    return dict(active=False, lam=NAN, w=(NAN,) * 3, G=np.full((3, W.NB), NAN), c=INF, m=1.0, n_train=len(P), theta=np.zeros(CM.NP), C=np.asarray(c0, float).copy(), spread=NAN, val=(NAN,) * 3)


def code_key() -> str:
    h = hashlib.sha256()
    for n in ("r_loss", "r_walk", "rd_model", "rd_walk", "cm_model", "cm_walk"):
        h.update((Path(__file__).parent / (n + ".py")).read_bytes())
    return h.hexdigest()[:12]


def month(var: str, ctx, a: int, b: int, level: str = "F", spec=None) -> dict:
    form, proto = spec or VARS[var]
    c0, xd = ctx.x["C"].to_numpy(float)[a:b], pd.DatetimeIndex(ctx.x["d"])
    if proto == "R":
        return fit_month_r(ctx, form, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c0)
    P = YA.closed_pool(pd.DatetimeIndex(ctx.x["t_plus"]), xd, xd[a])
    return zero_month_sc(P, c0) if xd[a] < YA.ACTIVATION else fit_month_sc(ctx, form, P, np.arange(a, b), c0, level)


def walk(var: str, ctx, positions=None, work=None, level: str = "F", spec=None) -> list:
    if (spec or VARS[var])[1] == "SC":
        YA.check_activation(pd.DatetimeIndex(ctx.x["d"]), pd.DatetimeIndex(ctx.x["t_plus"]))
    out = []
    for a, b in W.month_bounds(ctx):
        if positions is not None and a not in positions:
            continue
        f = None if work is None else Path(work) / ("m%03d.pkl" % a)
        if f is not None and f.exists():
            out.append((a, b, pickle.loads(f.read_bytes())))
            continue
        t0, res = time.time(), month(var, ctx, a, b, level, spec)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(res))
        print("%s %s: пар %d, λ %s, разброс L %.2e, %.0f с" % (var, pd.Timestamp(ctx.x["d"].iloc[a]).date(), res["n_train"], res["lam"], res["spread"], time.time() - t0), flush=True)
        out.append((a, b, res))
    return out


def assemble(var: str, ctx, parts: list, spec=None) -> pd.DataFrame:
    form, proto = spec or VARS[var]
    df, n = W.assemble("rck", ctx, parts), len(ctx.x)
    sp, val = np.full(n, NAN), np.full((n, 3), NAN)
    for a, b, r in parts:
        sp[a:b], val[a:b] = r["spread"], r["val"]
    df["spread"] = sp
    if proto == "SC":
        for i, k in enumerate(VK):
            df["val_" + k] = val[:, i]
    return df.rename(columns=dict(zip(CM.NAMES["CK"], CM.NAMES[form])))


def run(var: str, write: bool = True, out: Path = RD_DIR) -> pd.DataFrame:
    RG.require(IDS[var])
    bad = RG.check()
    if bad:
        raise SystemExit("реестр S-RCD не готов: %s" % "; ".join(bad))
    ctx = make_ctx("CK")
    df = assemble(var, ctx, walk(var, ctx, work=out / "_part" / ("%s_%s" % (var, code_key()))))
    p = out / OUTS[var]
    if write:
        if p.exists() and not W._same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        tmp = out / (OUTS[var] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" in sys.argv:
        v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
        print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
    else:
        from models_v2.s18 import rd_walk_chk as K
        sys.exit(K.main(sys.argv[1:]))
