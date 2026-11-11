"""S-L, L3 (`S_D_SPEC.md` разд. 7.2; объявления О-7, О-8 — `<sp>/s_l/GATES.md`). Обучения нет: прямой проход `a_model.forward` / `r_loss.lnc_of` по сохранённым θ месяца.
Выигрыш в выборке месяца: gain_j = ℓ(θ = 0) − ℓ(θ̂) по закрытому пулу месяца; k = ⌈0,01·|P|⌉; «выигрыш на 99 %» ⇔ Σgain − Σ(k лучших) > 0.
Ряды: старый S-A2 (`A/a2`, пул `closed_rows(D.tp, D)`, основной уровень — C, цель обучения `a_model.loss`), S-RCK и S-RA2 (`R/*`, пул `r_loss.pool_of`, основной уровень — F при замороженных π, β·g, как `r_eval_par.insample`).
Пишет `<sp>/s_l/l3.csv`. Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_l/l3.py"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s13.v6_1_wf import closed_rows
from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import a_model as AM
from models_v2.s18 import a_walk as AW
from models_v2.s18 import r_loss as RL
from models_v2.s18 import r_walk as W
from models_v2.s18.r_walk_ctx import make_ctx

HERE = Path(__file__).parent
R = C.ROOT / "data/regen/harj_struct"
ROWS, MON = [], []
MEM = list(range(5, 26))         # ω, ψ, c — веса лагов и внутридневные (обучаемая память A2)


def put(ser, part, level, item, n, value, note=""):
    ROWS.append(dict(section="L3", series=ser, part=part, level=level, item=item, n=int(n), value=float(value), note=note))


def split(gain) -> tuple:
    k = int(math.ceil(0.01 * len(gain)))
    top = float(np.sort(gain)[-k:].sum())
    tot = float(gain.sum())
    return tot, tot - top, (top / tot if tot > 0 else float("nan"))


def ell(rv, lnf):
    return RL.ell(rv, lnf)


def lnc_a(th, V):
    return AM.forward(torch.as_tensor(th, dtype=torch.float64), V, "A2")[1].numpy()


def old_a2(ctx):
    """Старый S-A2: 106 месяцев, θ = сохранённое среднее seeds; уровень C (цель) и F (замороженные π, β·g)."""
    df = pd.read_parquet(R / "A/a2.parquet")
    mon = df.drop_duplicates("refit_date")
    mon = mon[mon["lam"].notna()]
    D, xd = ctx.D, pd.DatetimeIndex(ctx.x["d"])
    errs = []
    for _, r in mon.iterrows():
        th_full = r[list(AW.NAMES)].to_numpy(float)
        parts = {"full": th_full, "memory_only": np.where(np.isin(np.arange(AM.NP), MEM), th_full, 0.0),
                 "no_memory": np.where(np.isin(np.arange(AM.NP), MEM), 0.0, th_full)}
        a = int(np.flatnonzero(xd == pd.Timestamp(r["refit_date"]))[0])
        b = a + int((df["refit_date"] == r["refit_date"]).sum())
        lnC_days = lnc_a(th_full, D.view(ctx.dp[a:b]))
        errs.append(np.max(np.abs(np.exp(lnC_days) / df["C_A"].to_numpy(float)[a:b] - 1.0)))
        P = closed_rows(D.tp, pd.Timestamp(r["refit_date"]))
        Vc, Vf = D.view(P), ctx.view(P)
        rv = Vc.rv.numpy()
        l0c = ell(rv, lnc_a(np.zeros(AM.NP), Vc))
        l0f = ctx.lnf0[P]
        for p, th in parts.items():
            lc = lnc_a(th, Vc)
            gc = l0c - ell(rv, lc)
            gf = ell(rv, l0f) - ell(rv, RL.ln_f(torch.as_tensor(lc), Vf).numpy())
            for lev, g in (("C", gc), ("F", gf)):
                tot, rest, sh = split(g)
                MON.append(dict(series="S-A2 (старый)", part=p, level=lev, refit=pd.Timestamp(r["refit_date"]).date(), n_pool=len(P), gain_total=tot, gain_rest99=rest, share_top1=sh, approx_err=errs[-1]))
    put("S-A2 (старый)", "full", "C", "approx_max_rel_C_meantheta_vs_saved", len(errs), max(errs), "прямой проход по среднему θ seeds против сохранённого C_A (среднее ln C seeds)")
    put("S-A2 (старый)", "full", "C", "approx_median_rel_C_meantheta_vs_saved", len(errs), float(np.median(errs)))
    put("S-A2 (старый)", "full", "C", "approx_months_rel_gt_1e-3", len(errs), int((np.asarray(errs) > 1e-3).sum()))


def r_series(ser, key, ctx, arch):
    """S-RCK / S-RA2: месяцы θ ≠ 0; уровень F (цель R, `r_eval_par.insample`) и C (против θ = 0: cfg8 у CK, C_A(0) у A2)."""
    df = pd.read_parquet(R / ("R/%s.parquet" % key))
    mon = df[df["active"]].groupby("refit_R", sort=True).first().reset_index()
    xd = pd.DatetimeIndex(ctx.x["d"])
    for _, r in mon.iterrows():
        if r["lam"] == 0.0:
            continue
        a = int(np.flatnonzero(xd == pd.Timestamp(r["refit_R"]))[0])
        P = RL.pool_of(arch, ctx.D, ctx.x, a)
        V = ctx.view(P)
        th = torch.as_tensor(r[W.names(arch)].to_numpy(float))
        lc = RL.lnc_of(th, V, arch).numpy()
        lc0 = RL.lnc_of(torch.zeros(RL.n_par(arch), dtype=torch.float64), V, arch).numpy()
        rv = ctx.rv[P]
        gf = ell(rv, ctx.lnf0[P]) - ell(rv, RL.ln_f(torch.as_tensor(lc), V).numpy())
        gc = ell(rv, lc0) - ell(rv, lc)
        for lev, g in (("F", gf), ("C", gc)):
            tot, rest, sh = split(g)
            MON.append(dict(series=ser, part="full", level=lev, refit=pd.Timestamp(r["refit_R"]).date(), n_pool=len(P), gain_total=tot, gain_rest99=rest, share_top1=sh))


def summary():
    m = pd.DataFrame(MON)
    for (s, p, lev), g in m.groupby(["series", "part", "level"], sort=False):
        n = len(g)
        put(s, p, lev, "months", n, n)
        put(s, p, lev, "frac_rest99_better", n, (g["gain_rest99"] > 0).mean())
        put(s, p, lev, "frac_rest99_not_better", n, (g["gain_rest99"] <= 0).mean())
        put(s, p, lev, "frac_total_better", n, (g["gain_total"] > 0).mean())
        put(s, p, lev, "share_top1_median", n, float(np.nanmedian(g["share_top1"])))
        put(s, p, lev, "gain_rest99_median", n, float(g["gain_rest99"].median()))
        put(s, p, lev, "gain_total_median", n, float(g["gain_total"].median()))
    a2 = m[(m["series"] == "S-A2 (старый)") & (m["approx_err"] <= 1e-3)]
    for (p, lev), g in a2.groupby(["part", "level"], sort=False):
        put("S-A2 (старый)", p, lev, "frac_rest99_not_better|approx_err<=1e-3", len(g), (g["gain_rest99"] <= 0).mean(), "устойчивость: только месяцы, где среднее θ seeds воспроизводит C_A с отн. ошибкой ≤ 1e-3")
    for (s, p, lev), g in m.assign(yr=pd.to_datetime(m["refit"]).dt.year).groupby(["series", "part", "level"], sort=False):
        for y, gy in g.assign(yr=pd.to_datetime(g["refit"]).dt.year).groupby("yr"):
            put(s, p, lev, "frac_rest99_better|%d" % y, len(gy), (gy["gain_rest99"] > 0).mean())
    return m


def main():
    ca, ck = make_ctx("A"), make_ctx("CK")
    old_a2(ca)
    r_series("S-RCK", "rck", ck, "CK")
    r_series("S-RA2", "ra2", ca, "A")
    m = summary()
    pd.DataFrame(ROWS).to_csv(HERE / "l3.csv", index=False, float_format="%.10g")
    m.to_csv(HERE / "l3_months.csv", index=False, float_format="%.10g")
    pd.set_option("display.width", 250, "display.max_rows", 3000)
    print(pd.DataFrame(ROWS).to_string(index=False, float_format=lambda v: "%.6g" % v))


if __name__ == "__main__":
    main()
