"""Строки Ж-200: разрешение конфигурации на дату, план задач (с дедупликацией), сборка прогнозов и утечек."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s2 import boost
from models_v2.s4 import design, runner
from models_v2.s4.fit import HP


@dataclass
class Ctx:
    F: object
    grid: np.ndarray
    DZ: dict
    old: HP
    i2: dict                      # {год цели: HP}
    rows: list = field(default_factory=list)
    hard: dict = field(default_factory=dict)   # {год: HP} для «Б-1 жёсткий»


@dataclass
class Row:
    name: str
    kind: str                     # old_mono | old_free | i2 | i6
    dzk: str = ""
    w: bool = False
    pick: dict = field(default_factory=dict)   # i6: {год: (dzk, w)}


def year_of(F, d: int) -> int:
    return int(F.dates[d + 1].astype("datetime64[Y]").astype(int)) + 1970


def resolve(row: Row, d: int, ctx: Ctx) -> tuple:
    """(dz_key, wflag, hp, mono) строки на дату d (год — по дате цели)."""
    Y = year_of(ctx.F, d)
    if row.kind == "old_mono":
        return row.dzk, False, ctx.old, design.mono_vec(11)
    if row.kind == "old_free":
        return row.dzk, False, ctx.old, None
    if row.kind == "i2":
        return row.dzk, row.w, ctx.i2[Y], None
    if row.kind == "hard":
        return row.dzk, row.w, ctx.hard[Y], None
    dzk, w = row.pick[Y]
    return dzk, w, ctx.i2[Y], None


def monthly_blocks(ctx: Ctx) -> dict:
    return boost.blocks(ctx.F, ctx.grid, daily=False)


def plan(ctx: Ctx, rows: list, with_leak: bool = True) -> dict:
    """{ключ: задача}; ключ = (dzk, w, hp, mono, T, variant); дубли строк схлопываются."""
    jobs, bl = {}, monthly_blocks(ctx)
    n = len(ctx.F.dates)
    for r in rows:
        for d in ctx.grid:
            dzk, w, hp, mono = resolve(r, int(d), ctx)
            k = (dzk, w, hp, mono, int(d), "work")
            jobs[k] = (k, dzk, hp, int(d), [int(d)], w, mono, "work")
        if not with_leak:
            continue
        for T, ds in bl.items():
            dzk, w, hp, mono = resolve(r, ds[0], ctx)
            for v in ("work", "oracle", "perm"):
                k = (dzk, w, hp, mono, T, "m_" + v)
                jobs[k] = (k, dzk, hp, T, ds, w, mono, v)
        dzk, w, hp, mono = resolve(r, int(ctx.grid[-1]), ctx)
        k = (dzk, w, hp, mono, n - 1, "full")
        jobs[k] = (k, dzk, hp, n - 1, [int(x) for x in ctx.grid], w, mono, "full")
    return jobs


def run_plan(ctx: Ctx, jobs: dict, procs: int = 4) -> dict:
    return dict(runner.map_jobs(ctx.F, ctx.DZ, list(jobs.values()), "block", procs))


def _sig(ctx, dzk, d_arr, f, s2, kind):
    off = ctx.DZ[dzk].off[d_arr]
    return np.exp(off + f + (s2 / 2 if kind == "b" else 0.0))


def forecasts(ctx: Ctx, res: dict, row: Row) -> pd.DataFrame:
    F, out = ctx.F, []
    for kind in ("a", "b"):
        sig = []
        for d in ctx.grid:
            dzk, w, hp, mono = resolve(row, int(d), ctx)
            r = res[(dzk, w, hp, mono, int(d), "work")]
            sig.append(_sig(ctx, dzk, np.array([d]), r["f"], r["s2_es"], kind)[0])
        g = ctx.grid
        out.append(pd.DataFrame({"d": F.dates[g], "target_date": F.dates[g + 1], "model": "%s_%s" % (row.name, kind),
                                 "sigma2": sig, "rv": F.rv[g + 1]}))
    return pd.concat(out)


def overfit(ctx: Ctx, res: dict, row: Row) -> pd.DataFrame:
    """MSE ln: обучение (предварительная модель вне блока остановки), блок остановки, отношение; по годам и 'all'."""
    rec = []
    for d in ctx.grid:
        r = res[resolve(row, int(d), ctx)[:4] + (int(d), "work")]
        rec.append((year_of(ctx.F, int(d)), r["k"], r["s2_tr_prelim"], r["s2_es"]))
    df = pd.DataFrame(rec, columns=["year", "k", "tr", "es"])
    df["ratio"] = df["es"] / df["tr"]
    out = []
    for y, g in [("all", df)] + [(str(k), v) for k, v in df.groupby("year")]:
        out.append({"model": row.name, "year": y, "n_retrain": len(g), "trees_mean": g.k.mean(), "mse_train": g.tr.mean(),
                    "mse_es": g.es.mean(), "ratio_mean": g.ratio.mean(), "ratio_of_means": g.es.mean() / g.tr.mean()})
    return pd.DataFrame(out)


def leak_row(ctx: Ctx, res: dict, row: Row, daily_b: pd.DataFrame, rv_m_q: float) -> dict:
    """QLIKE (б): work (ежедневно), monthly work/oracle/perm, full; допуски (i), (ii), (iii)."""
    F, out = ctx.F, {"model": row.name}
    out["work_daily"] = float(metrics.qlike(daily_b["rv"], daily_b["sigma2"]).mean())
    for v in ("work", "oracle", "perm", "full"):
        sig, rv = [], []
        if v == "full":
            dzk, w, hp, mono = resolve(row, int(ctx.grid[-1]), ctx)
            items = [(res[(dzk, w, hp, mono, len(F.dates) - 1, "full")], np.array(ctx.grid))]
        else:
            items = []
            for T, ds in monthly_blocks(ctx).items():
                dzk, w, hp, mono = resolve(row, ds[0], ctx)
                items.append((res[(dzk, w, hp, mono, T, "m_" + v)], np.array(ds)))
        for r, ds in items:
            dzk = resolve(row, int(ds[0]), ctx)[0]
            sig.append(_sig(ctx, dzk, ds, r["f"], r["s2_es"], "b"))
            rv.append(F.rv[ds + 1])
        out[v] = float(metrics.qlike(np.concatenate(rv), np.concatenate(sig)).mean())
    out["qlike_rv_m"] = rv_m_q
    out["i_ok"] = out["oracle"] <= 0.5 * out["work"]
    out["ii_ok"] = out["full"] <= out["work_daily"] + 1e-12
    out["iii_near_rv_m"] = abs(out["perm"] / rv_m_q - 1) <= 0.30
    return out
