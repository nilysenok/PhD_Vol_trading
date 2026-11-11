"""S-C-EVAL (Ж-254; `S_C_SPEC.md`, разд. 2 «Учёт сложности», 4.5, 7 п. 7): параметры по месяцам и годам, число обученных вариантов и подгонок, распределение ln 𝓜. Только чтение рядов `CM/{cd,ck,cka,c7,ckl}.parquet` и `CM/feat.parquet`.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Параметры — колонки после `spread` (`cm_model.NAMES`: у C7 их нет).
Публичное API:
- `N_LAMS = 3`, `N_SEEDS = 5` (сверяются в `cm_eval --check` с `cm_model.LAMS` / `SEEDS`); `NZ_EPS = 1e-6` (‖θ‖₀: |θ| > 1e-6); `LN_EPS = 1e-3`; `MIN_VALID = 252` (до него без валидации: 5 подгонок на переоценку, иначе (3 λ + 1)·5 = 20).
- `params(frames) -> DataFrame` (frames: «S-CD»… → кадр): level `refit` (месяц D: n_train, lam, spread, seed_best, nz0 = ‖θ‖₀, параметры), `year` (среднее по переоценкам года), `fits` (число обученных вариантов и подгонок; у S-C7 — 0).
- `n_fits(n_trains)`: подгонок L-BFGS-B по списку n_train активных переоценок.
- `lnm(frames, feat_s) -> DataFrame`: kind `dist` (ln 𝓜 на D_M и на активных датах: доля |ln 𝓜| > 1e-3, среднее, std, квантили), `year` (по календарным годам), `share` (средние F_c/C, F_p/C по годам против ŝ_c, ŝ_p; feat_s — датафрейм ŝ_c, ŝ_p на датах d)."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_LAMS, N_SEEDS, NZ_EPS, LN_EPS, MIN_VALID = 3, 5, 1e-6, 1e-3, 252
QS = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def names(df: pd.DataFrame) -> list:
    return list(df.columns[df.columns.get_loc("spread") + 1:])


def n_fits(n_trains) -> int:
    return int(sum(N_SEEDS if n < MIN_VALID else (N_LAMS + 1) * N_SEEDS for n in n_trains))


def params(frames: dict) -> pd.DataFrame:
    out = []
    for var, df in frames.items():
        nm = names(df)
        r = df[df["active"]].drop_duplicates("refit_date").copy()
        r["year"] = pd.DatetimeIndex(r["refit_date"]).year
        nz = (r[nm].abs() > NZ_EPS).sum(axis=1) if nm else pd.Series(0, index=r.index)
        for x, z in zip(r.itertuples(), nz):
            out.append(dict(variant=var, level="refit", key=str(pd.Timestamp(x.refit_date).date()), n=int(x.n_train), lam=x.lam, spread=x.spread, seed_best=x.seed_best, nz0=int(z), **{k: getattr(x, k) for k in nm}))
        for y, g in r.assign(nz0=nz).groupby("year"):
            out.append(dict(variant=var, level="year", key=str(y), n=len(g), lam=float(g["lam"].mean()), spread=float(g["spread"].mean()), spread_max=float(g["spread"].max()), nz0=float(g["nz0"].mean()),
                            **{k: float(g[k].mean()) for k in nm}))
        nf = n_fits(r["n_train"]) if nm else 0
        out.append(dict(variant=var, level="fits", key="переоценок %d; параметров %d; подгонок %d (n_train < %d — %d подгонок, иначе %d)" % (len(r), len(nm), nf, MIN_VALID, N_SEEDS, (N_LAMS + 1) * N_SEEDS), n=nf, lam=np.nan))
    return pd.DataFrame(out)


def _dist(x: np.ndarray) -> dict:
    q = np.quantile(x, QS)
    return dict(n=len(x), share_abs_gt_1e3=float(np.mean(np.abs(x) > LN_EPS)), mean=float(x.mean()), std=float(x.std(ddof=1)), min=float(x.min()), **{"q%02d" % round(100 * k): float(v) for k, v in zip(QS, q)}, max=float(x.max()))


def lnm(frames: dict, feat_s: pd.DataFrame) -> pd.DataFrame:
    out = []
    for var, df in frames.items():
        ln, act, y = np.log(df["M"].to_numpy(float)), df["active"].to_numpy(bool), pd.DatetimeIndex(df["t_plus"]).year.to_numpy()
        out += [dict(kind="dist", variant=var, key=w, **_dist(ln[m])) for w, m in (("D_M", np.ones(len(ln), bool)), ("активные", act))]
        out += [dict(kind="year", variant=var, key=str(k), **_dist(ln[y == k])) for k in np.unique(y)]
        if var in ("S-CK", "S-CKA", "S-CKL"):
            fc, fp, C = (df[k].to_numpy(float) for k in ("F_c", "F_p", "C"))
            sc, sp = feat_s["sc"].to_numpy(float), feat_s["sp"].to_numpy(float)
            out += [dict(kind="share", variant=var, key=str(k), n=int((y == k).sum()), fc_over_c=float(np.mean(fc[y == k] / C[y == k])), fp_over_c=float(np.mean(fp[y == k] / C[y == k])),
                         s_c=float(sc[y == k].mean()), s_p=float(sp[y == k].mean())) for k in np.unique(y)]
    return pd.DataFrame(out)
