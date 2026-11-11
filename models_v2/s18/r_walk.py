"""S-R-IMPL-2 (Ж-254; `S_R_SPEC.md`, разд. 2.4–2.5, 3, 6, 7): помесячный проход вариантов S-RA2 / S-RCK / S-RA2b / S-RCKb / S-RA2c / S-RCKc, слои X1 и W5-G над C_R, запись `data/regen/harj_struct/R/{ra2,rck,ra2b,rckb,ra2c,rckc}.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Процедура выбрана после S-C-WHY: результат исследовательский, не независимый OOS.
Протокол: переоценка в первый торговый день месяца D (106 месяцев); закрытый пул P (t⁺ ≤ D) по возрастанию t⁺; включение при |P| ≥ 630 (до него θ = 0, C_R = cfg8 побитно); сетка Λ = {θ = 0; 1e-1; 1e-2; 1e-3};
блоки V₃ = последние 126 пар P, V₂ — предыдущие 126, V₁ — ещё предыдущие; T_b = пары P раньше V_b; c, m — по T_b; G_{λ,b} = mean_{V_b}[ℓ(RV, F(θ̂_{λ,b})) − ℓ(RV, F(0))] (обычная QLIKE уровня F); w_λ = #{b: G < 0};
допуск w ≥ 2; нет допущенных — θ = 0; иначе λ* = argmax w, ничья — больший λ; финальная подгонка на P с c, m по P. Rb — потеря ℓ уровня F (c = ∞, m = 1); Rc — одна валидация V₃: минимум G, ничья — θ = 0, иначе больший λ.
Публичное API:
- `LAMS`, `VAL = 126`, `NB = 3`, `MIN_POOL = 630`, `MIN_W = 2`; `VARS` — id → (архитектура, потеря, правило); `IDS` — вариант → id реестра; `OUTS` — вариант → parquet; `R_DIR`; `names(arch)`.
- `Ctx(arch, D, lv, x, gap, dp)`: данные архитектуры + x0 = RV/F(0); `make_ctx(arch)`; `blocks(P, nb) -> [(T_b, V_b)]`; `decide(G, rule) -> (λ*, w)`; `choose(ctx, P, mode, rule) -> (λ*, w, G)`; `fit_month(ctx, P, days, c0, mode, rule) -> dict`.
- `walk_variant(var, ctx, positions=None, work=None) -> [(a, b, dict)]`; `assemble(var, ctx, parts) -> DataFrame`: d, t_plus, rv, cfg8, C, F_fr, X1_R, F, pi, beta_gap, refit_date, active, lam, w_*, G_*, c, m, n_train, параметры архитектуры;
  F_fr = B_mix^(1−π)·C_R^π·e^{βg} с сохранёнными π, β; refit_R — дата переоценки R (первый день месяца), refit_date — переоценка слоя W5; `run(var, write=True, out=R_DIR)`: отказ, если id нет в реестре; существующий выход с иным содержимым — отказ. Проверки — `r_walk_chk.py`.
Запуск: `python3 -W ignore -m models_v2.s18.r_walk --run ra2|rck|ra2b|rckb|ra2c|rckc | --check-nest | --check-pit | --check`."""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s18 import a_layers as AL
from models_v2.s18 import a_walk as AW
from models_v2.s18 import cm_layers as CL
from models_v2.s18 import cm_model as CM
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_reg as RG
from models_v2.s18.r_walk_ctx import Ctx, make_ctx  # noqa: F401  (API: Ctx, make_ctx)

torch.set_num_threads(1)
LAMS, VAL, NB, MIN_POOL, MIN_W = RL.GRID, 126, 3, 630, 2
VARS = {"ra2": ("A", "R", "blocks"), "rck": ("CK", "R", "blocks"), "ra2b": ("A", "plain", "blocks"), "rckb": ("CK", "plain", "blocks"), "ra2c": ("A", "R", "single"), "rckc": ("CK", "R", "single")}
IDS = {"ra2": "S-RA2", "rck": "S-RCK", "ra2b": "S-RA2b", "rckb": "S-RCKb", "ra2c": "S-RA2c", "rckc": "S-RCKc"}
OUTS = {v: v + ".parquet" for v in IDS}
R_DIR = RG.R_DIR
LK = ("1e-1", "1e-2", "1e-3")
assert LAMS == (1e-1, 1e-2, 1e-3)


def names(arch: str) -> list:
    return list(AW.NAMES) if arch == "A" else list(CM.NAMES["CK"])


def blocks(P, nb: int = NB) -> list:
    P = np.asarray(P)
    return [(P[: len(P) - VAL * (nb - b)], P[len(P) - VAL * (nb - b): len(P) - VAL * (nb - b - 1)]) for b in range(nb)]


def decide(G: dict, rule: str) -> tuple:
    """G: λ → [G по блокам]. Возвращает (λ*, w); λ* = 0.0 — θ = 0."""
    if rule == "blocks":
        w = {lam: sum(g < 0 for g in G[lam]) for lam in LAMS}
        ok = [lam for lam in LAMS if w[lam] >= MIN_W]
        return (max(l for l in ok if w[l] == max(w[k] for k in ok)) if ok else 0.0), w
    cand = [(0.0, 0.0)] + [(lam, G[lam][0]) for lam in LAMS]
    best = min(cand, key=lambda t: (t[1], 0 if t[0] == 0 else 1, -t[0]))
    return best[0], {lam: int(G[lam][0] < 0) for lam in LAMS}


LOSS_MODE = {"R": "R", "plain": "plain", "Rm1": "R"}    # режим потери для RL.fit_r; Rm1 — S-RCKm (Ж-256): m := 1, обрезка c сохранена


def stats_of(x0, T, mode: str) -> tuple:
    if mode == "Rm1":
        return RL.clip_stats(x0, T)[0], 1.0
    return RL.clip_stats(x0, T) if mode == "R" else (float("inf"), 1.0)


def choose(ctx: Ctx, P, mode: str, rule: str = "blocks") -> tuple:
    bl = blocks(P) if rule == "blocks" else blocks(P)[-1:]
    G = {lam: [] for lam in LAMS}
    for T, Vb in bl:     # обучение — на уровне ctx.view_train / x0_train; валидация — всегда уровень F (ctx.view, ctx.lnf0)
        c, m, VT, VV = *stats_of(ctx.x0_train, T, mode), ctx.view_train(T), ctx.view(Vb)
        base = float(RL.ell(ctx.rv[Vb], ctx.lnf0[Vb]).mean())
        for lam in LAMS:
            th = RL.fit_r(VT, ctx.arch, lam, c, m, LOSS_MODE[mode], corr=ctx.corr(T))
            lnf = RL.ln_f(torch.as_tensor(RL.predict_lnc(th, VV, ctx.arch)), VV).numpy()
            G[lam].append(float(RL.ell(ctx.rv[Vb], lnf).mean()) - base)
    lam, w = decide(G, rule)
    return lam, w, G


def fit_month(ctx: Ctx, P, days, c0, mode: str, rule: str = "blocks") -> dict:
    nan, np_ = float("nan"), RL.n_par(ctx.arch)
    out = dict(active=False, lam=nan, w=(nan,) * 3, G=np.full((3, NB), nan), c=nan, m=nan, n_train=len(P), theta=np.zeros(np_), C=np.asarray(c0, float).copy())
    if len(P) < MIN_POOL:
        return out
    lam, w, G = choose(ctx, P, mode, rule)
    out.update(active=True, lam=lam, w=tuple(w[k] for k in LAMS), c=stats_of(ctx.x0_train, P, mode)[0], m=stats_of(ctx.x0_train, P, mode)[1])
    for i, k in enumerate(LAMS):
        out["G"][i, NB - len(G[k]):] = G[k]
    if lam > 0.0:
        th = RL.fit_r(ctx.view_train(P), ctx.arch, lam, out["c"], out["m"], LOSS_MODE[mode], corr=ctx.corr(P))
        out.update(theta=np.mean(th, axis=0), C=np.exp(RL.predict_lnc(th, ctx.view(days), ctx.arch)))
    return out


def code_key() -> str:
    h = hashlib.sha256()
    for n in ("r_loss", "r_walk"):
        h.update((Path(__file__).parent / (n + ".py")).read_bytes())
    return h.hexdigest()[:12]


def month_bounds(ctx: Ctx) -> list:
    from models_v2.s16 import x1_ya_walk as YA
    xd = pd.DatetimeIndex(ctx.x["d"])
    pos = AW.refit_positions(xd, AW.FIRST) if ctx.arch == "A" else YA.month_starts(xd)
    return list(zip(pos, pos[1:] + [len(xd)]))


def walk_variant(var: str, ctx: Ctx, positions=None, work=None, spec=None) -> list:
    arch, mode, rule = spec or VARS[var]
    out, c8 = [], ctx.x["C"].to_numpy(float)
    for a, b in month_bounds(ctx):
        if positions is not None and a not in positions:
            continue
        f = None if work is None else Path(work) / ("m%03d.pkl" % a)
        if f is not None and f.exists():
            out.append((a, b, pickle.loads(f.read_bytes())))
            continue
        t0 = time.time()
        res = fit_month(ctx, RL.pool_of(arch, ctx.D, ctx.x, a), ctx.dp[a:b], c8[a:b], mode, rule)
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(pickle.dumps(res))
        print("%s %s: пар %d, λ* %s, %.0f с" % (var, pd.Timestamp(ctx.x["d"].iloc[a]).date(), res["n_train"], res["lam"], time.time() - t0), flush=True)
        out.append((a, b, res))
    return out


def assemble(var: str, ctx: Ctx, parts: list) -> pd.DataFrame:
    x, arch, n = ctx.x, ctx.arch, len(ctx.x)
    col = {k: np.full(n, np.nan) for k in ("lam", "c", "m")}
    w, G = np.full((n, 3), np.nan), np.full((n, 9), np.nan)
    C, ntr, act, th = x["C"].to_numpy(float).copy(), np.zeros(n, int), np.zeros(n, bool), np.zeros((n, RL.n_par(arch)))
    rr = np.full(n, np.datetime64("NaT"), "datetime64[ns]")
    for a, b, r in parts:
        C[a:b], ntr[a:b], act[a:b], th[a:b], rr[a:b] = r["C"], r["n_train"], r["active"], r["theta"], np.datetime64(x["d"].iloc[a])
        w[a:b], G[a:b] = np.asarray(r["w"], float), r["G"].ravel()
        for k in col:
            col[k][a:b] = r[k]
    lay = (AL if arch == "A" else CL).layers(x, C, ctx.gap)
    lvx = RL.level_arrays("CK", x=x)
    ffr = np.exp((1.0 - lvx.pi) * lvx.lnb + lvx.pi * np.log(C) + lvx.bg)
    df = pd.DataFrame(dict(d=x["d"].to_numpy(), t_plus=x["t_plus"].to_numpy(), rv=x["rv"].to_numpy(float), cfg8=x["C"].to_numpy(float), C=C, F_fr=ffr, X1_R=lay["X1_A" if arch == "A" else "X1_V"],
                           F=lay["F"], pi=lay["pi"], beta_gap=lay["beta_gap"], refit_date=lay["refit_date"], refit_R=rr, active=act, lam=col["lam"]))
    for i, k in enumerate(LK):
        df["w_" + k] = w[:, i]
    for i, k in enumerate(LK):
        for b in range(NB):
            df["G_%s_b%d" % (k, b + 1)] = G[:, i * NB + b]
    for k in ("c", "m"):
        df[k] = col[k]
    df["n_train"] = ntr
    return pd.concat([df, pd.DataFrame(th, columns=names(arch))], axis=1)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_exact=True)
        return True
    except AssertionError:
        return False


def run(var: str, write: bool = True, out: Path = R_DIR) -> pd.DataFrame:
    RG.require(IDS[var])
    bad = RG.check()
    if bad:
        raise SystemExit("реестр S-R не готов: %s" % "; ".join(bad))
    ctx = make_ctx(VARS[var][0])
    df = assemble(var, ctx, walk_variant(var, ctx, work=out / "_part" / ("%s_%s" % (var, code_key()))))
    p = out / OUTS[var]
    if write:
        if p.exists() and not _same(pd.read_parquet(p), df):
            raise SystemExit("%s: существует и отличается от пересчёта — перезапись отказана" % p)
        tmp = out / (OUTS[var] + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(p)
    return df


if __name__ == "__main__":
    if "--run" in sys.argv:
        v, t0 = sys.argv[sys.argv.index("--run") + 1], time.time()
        print("%s: записано %d строк, %.0f с" % (v, len(run(v)), time.time() - t0))
    elif any(k in sys.argv for k in ("--check", "--check-pit", "--check-nest")):
        from models_v2.s18 import r_walk_chk as K
        k = [a for a in sys.argv if a in ("--check", "--check-pit", "--check-nest")][0]
        b = {"--check": K.check, "--check-pit": K.check_pit, "--check-nest": K.check_nest}[k]()
        print("%s: %s" % (k, "ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        raise SystemExit("укажите --run VAR, --check, --check-pit или --check-nest")
