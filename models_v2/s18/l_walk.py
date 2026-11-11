"""S-L-IMPL-1/2 (Ж-256 п. 5; `S_L_SPEC.md`, разд. 2–3): помесячный проход S-RL1 (C = cfg8·e^a), контроль (а) S-RL1a (π := π̄ пула в потере обучения) и контроль (б) S-RL1b (π со сдвигом 252 в потере обучения).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; исследовательский результат, не OOS. R импортом: `r_walk.blocks`, `decide`, `LAMS`, `VAL`, `NB`, `MIN_POOL`, `month_bounds`, `make_ctx("CK")`, `r_loss.pool_of`, `cm_layers.layers` (через `r_walk.assemble`).
Публичное API: `VARS_L` = {"rl1": "pi", "rl1a": "pibar", "rl1b": "shift252"}; `IDS_L`; `OUTS_L`; `L_DIR`; `arrays(ctx)`; `choose_l(ctx, P, kind) -> (λ*, w, G)`; `fit_month_l(ctx, P, days, c0, kind) -> dict` (как `r_walk.fit_month` + `a`);
`walk_l(kind, ctx, positions=None)`; `assemble_l(ctx, parts)`: колонки R/rck.parquet без θ-колонок + `a`; `run(var, write=True, out=L_DIR)`: `l_reg.require` + `l_reg.check()` пуст → проход → запись через `.tmp` и `replace`; существующий иной выход — отказ.
Валидация и слои — всегда настоящий π; при a = 0 прогноз берётся тем же путём, что `ctx.lnf0` (G = 0 точно, как после ошибки S-R 11.11).
Запуск: `python3 -W ignore -m models_v2.s18.l_walk --run rl1|rl1a|rl1b|rl1b2`. Проверки — `l_walk_chk.py`."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import l_model as ML
from models_v2.s18 import l_reg as LR
from models_v2.s18 import l_reg2 as LR2
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W

VARS_L = {"rl1": "pi", "rl1a": "pibar", "rl1b": "shift252", "rl1b2": "shift252nr"}    # rl1b2 — Ж-257 п. 3: сдвиг без переноса, основной контроль (б)
IDS_L = {"rl1": "S-RL1", "rl1a": "S-RL1a", "rl1b": "S-RL1b", "rl1b2": "S-RL1b2"}
OUTS_L = {v: v + ".parquet" for v in IDS_L}
L_DIR = LR.L_DIR


def arrays(ctx: W.Ctx) -> dict:
    return dict(rv=ctx.rv, lnb=ctx.lv.lnb, pi=ctx.lv.pi, bg=ctx.lv.bg, lnc8=torch.log(torch.as_tensor(np.asarray(ctx.D.c8, float))).numpy())


def lnf_fr(a: float, A: dict, sel, lnf0) -> np.ndarray:
    """ln F^fr(a) с настоящим π на sel; a = 0 — путь `ctx.lnf0`."""
    sel = np.asarray(sel)
    return lnf0[sel] if a == 0.0 else ML.ln_ftrain(a, A["lnb"][sel], A["pi"][sel], A["bg"][sel], A["lnc8"][sel])


def choose_l(ctx: W.Ctx, P, kind: str) -> tuple:
    A, G = arrays(ctx), {lam: [] for lam in W.LAMS}
    for T, Vb in W.blocks(P):
        c, m = ML.stats_l(kind, A, T)
        base = float(RL.ell(ctx.rv[Vb], ctx.lnf0[Vb]).mean())
        for lam in W.LAMS:
            a = ML.fit_kind(kind, A, T, lam, c, m)
            G[lam].append(float(RL.ell(ctx.rv[Vb], lnf_fr(a, A, Vb, ctx.lnf0)).mean()) - base)
    lam, w = W.decide(G, "blocks")
    return lam, w, G


def fit_month_l(ctx: W.Ctx, P, days, c0, kind: str) -> dict:
    nan = float("nan")
    out = dict(active=False, lam=nan, w=(nan,) * 3, G=np.full((3, W.NB), nan), c=nan, m=nan, n_train=len(P), theta=np.zeros(RL.n_par("CK")), C=np.asarray(c0, float).copy(), a=0.0)
    if len(P) < W.MIN_POOL:
        return out
    A = arrays(ctx)
    lam, w, G = choose_l(ctx, P, kind)
    c, m = ML.stats_l(kind, A, P)
    out.update(active=True, lam=lam, w=tuple(w[k] for k in W.LAMS), c=c, m=m)
    for i, k in enumerate(W.LAMS):
        out["G"][i, W.NB - len(G[k]):] = G[k]
    if lam > 0.0:
        a = ML.fit_kind(kind, A, P, lam, c, m)
        out.update(a=a, C=np.asarray(c0, float) * np.exp(a))
    return out


def walk_l(kind: str, ctx: W.Ctx, positions=None) -> list:
    out, c8 = [], ctx.x["C"].to_numpy(float)
    for a, b in W.month_bounds(ctx):
        if positions is not None and a not in positions:
            continue
        res = fit_month_l(ctx, RL.pool_of("CK", ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], kind)
        print("%s %s: пар %d, λ* %s, a %s" % (kind, pd.Timestamp(ctx.x["d"].iloc[a]).date(), res["n_train"], res["lam"], res["a"]), flush=True)
        out.append((a, b, res))
    return out


def assemble_l(ctx: W.Ctx, parts: list) -> pd.DataFrame:
    df = W.assemble("rck", ctx, parts).drop(columns=W.names("CK"))
    a = np.zeros(len(ctx.x))
    for s, e, r in parts:
        a[s:e] = r["a"]
    df["a"] = a
    return df


def run(var: str, write: bool = True, out: Path = L_DIR) -> pd.DataFrame:
    reg = LR2 if var == "rl1b2" else LR
    reg.require(IDS_L[var])
    bad = reg.check()
    if bad:
        raise SystemExit("реестр S-L не готов: %s" % "; ".join(bad))
    ctx = W.make_ctx("CK")
    df = assemble_l(ctx, walk_l(VARS_L[var], ctx))
    p = out / OUTS_L[var]
    if write:
        if p.exists() and not W._same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / (OUTS_L[var] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" not in sys.argv:
        raise SystemExit("укажите --run rl1|rl1a|rl1b")
    v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
    print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
