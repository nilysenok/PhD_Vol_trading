"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 8 п. 6, 8): параметры по месяцам и годам, число обученных вариантов и подгонок. Только чтение рядов `A/{a0,a1,a2,a2p}.parquet` (прогнозы и реестр не трогаются).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Колонки параметров — 28 имён после колонки `corr` (порядок `a_walk.NAMES`: d0, d_s, d_m, d_l, d_J, om_*, psi_*, c_*, dw, dk); θ в parquet — среднее по seeds.
Публичное API:
- `N_LAMS = 3`, `N_SEEDS = 5`: число значений λ и seeds (сверяются в `a_eval --check` с `a_walk.LAMS` / `a_walk.SEEDS`); `GRID` = τ 0; 0,25; 0,5; 0,75; 1.
- `params(frames) -> DataFrame` (frames: «S-A0»… → кадр; в колонке `variant` — A0, A1, A2, A2p): level `refit` (месяц D: n = n_train, lam, corr_share, 28 параметров), `year` (среднее по переоценкам года; n = число переоценок), `b_grid` (A2: b_k(τ) на сетке по среднему c года,
  b = exp(h(τ))/mean по 53 слотам exp(h) — показ, не фактический множитель сессии с пропусками), `fits` (число обученных вариантов и подгонок).
- `n_fits(var, n_refits)`: подгонок на переоценку — (число λ + 1)·seeds (A0 — один seed, ω, ψ, c не свободны)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s18 import a_feat as AF

N_LAMS, N_SEEDS = 3, 5
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
VARIANTS = ("A0", "A1", "A2", "A2p")


def names(df: pd.DataFrame) -> list:
    return list(df.columns[df.columns.get_loc("corr") + 1:])


def n_fits(var: str, n_refits: int) -> int:
    return n_refits * (N_LAMS + 1) * (1 if var == "A0" else N_SEEDS)


def b_grid(c: np.ndarray) -> np.ndarray:
    """c (4,) -> b(τ) на GRID: exp(B(τ)c)/mean_{слоты 53} exp(B(τ_j)c)."""
    e = np.exp(AF.spline_basis(np.asarray(GRID)) @ c)
    return e / np.exp(AF.spline_basis(AF.TAU) @ c).mean()


def params(frames: dict) -> pd.DataFrame:
    out = []
    for var, df in frames.items():
        var, nm = var.replace("S-", ""), names(df)
        r = df.drop_duplicates("refit_date").copy()
        r["year"] = pd.DatetimeIndex(r["refit_date"]).year
        fit = r[r["lam"].notna()]
        for x in r.itertuples():
            out.append(dict(variant=var, level="refit", key=str(pd.Timestamp(x.refit_date).date()), n=int(x.n_train), lam=x.lam, corr_share=float(x.corr), **{k: getattr(x, k) for k in nm}))
        for y, g in fit.groupby("year"):
            out.append(dict(variant=var, level="year", key=str(y), n=len(g), lam=float(g["lam"].mean()), corr_share=float(g["corr"].mean()), **{k: float(g[k].mean()) for k in nm}))
            if var == "A2":
                for k in "sml":
                    bg = b_grid(np.array([g["c_%s%d" % (k, i)].mean() for i in (1, 2, 3, 4)]))
                    out.append(dict(variant=var, level="b_grid", key="год %s, память %s" % (y, k), **{"b_t%d" % round(100 * t): float(v) for t, v in zip(GRID, bg)}))
        out.append(dict(variant=var, level="fits", key="переоценок с подгонкой %d из %d месяцев; подгонок на переоценку %d" % (len(fit), len(r), n_fits(var, 1)), n=n_fits(var, len(fit)), lam=np.nan))
    return pd.DataFrame(out)
