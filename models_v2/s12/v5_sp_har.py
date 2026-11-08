"""V5 лист 2, SP1 (вариант A, фиксированное разделение): F = K8·clip(R, 2/3, 3/2), R = Σ_b Ê_b / Ê_tot.
Ê_b — HAR в уровнях по лагам полосы b (E_b дня d, среднее за 5, за 22 строки бумаги), Ê_tot — HAR в уровнях по лагам RV; МНК по бумаге, переоценка ежемесячно по закрытым парам (цель τ < первого числа месяца);
меньше 250 закрытых пар у бумаги или Ê_tot ≤ 0 / нефинитно — R = 1 (F = K8 побитно). Цели полос — E_b дня τ (кэш листа 0), цель суммы — RV дня τ. Моделей по другим бумагам нет: LOTO совпадает с основным прогнозом."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s12 import v5_sp_feat as FE
from models_v2.s12.v5_gs_fit import closed_n

MIN_PAIRS = 250
LO, HI = 2.0 / 3.0, 3.0 / 2.0


def arrays(P, F: FE.Feat) -> tuple:
    """X (N, m, 4, 3) лаги дня d (цели 0–2 — полосы, 3 — сумма) и Y (N, m, 4) цели дня τ; NaN, где нет."""
    di, ok = FE.panel_index(P, F)
    jx = np.broadcast_to(np.arange(len(P.tick))[None, :], di.shape)
    ti = np.minimum(np.searchsorted(F.D.to_numpy(), P.axis.to_numpy()), len(F.D) - 1)
    tok = (F.D.to_numpy()[ti] == P.axis.to_numpy())[:, None]
    X = np.stack([np.stack([F.E[..., b], F.E5[..., b], F.E22[..., b]], axis=-1) for b in range(3)] + [np.stack([F.RV, F.R5, F.R22], axis=-1)], axis=2)[di, jx]
    Y = np.concatenate([F.E[ti], P.rv[:, :, None]], axis=-1)
    return np.where(ok[..., None, None], X, np.nan), np.where((tok & P.pres)[..., None], Y, np.nan)


def run(P, F: FE.Feat) -> dict:
    """-> dict(F (N, m), R (N, m), params DataFrame)."""
    N, m = P.rv.shape
    X, Y = arrays(P, F)
    good = P.pres & np.isfinite(X).all(axis=(2, 3))
    fit_ok = good & np.isfinite(Y).all(axis=2)
    R, rows = np.full((N, m), np.nan), []
    for mo in sorted(set(P.mon[P.pres].tolist())):
        m0 = pd.Timestamp(year=(mo - 1) // 12, month=(mo - 1) % 12 + 1, day=1)
        nu = closed_n(P.axis, m0)
        n_fit, clipped, cnt = 0, 0, 0
        for j in range(m):
            sel = np.flatnonzero((P.mon[:, j] == mo) & P.pres[:, j])
            if not len(sel):
                continue
            tr = np.flatnonzero(fit_ok[:nu, j])
            r = np.ones(len(sel))
            if len(tr) >= MIN_PAIRS:
                n_fit += 1
                pred = np.empty((len(sel), 4))
                for b in range(4):
                    A = np.c_[np.ones(len(tr)), X[tr, j, b, :]]
                    beta = np.linalg.lstsq(A, Y[tr, j, b], rcond=None)[0]
                    pred[:, b] = np.c_[np.ones(len(sel)), np.nan_to_num(X[sel, j, b, :])] @ beta
                tot = pred[:, 3]
                with np.errstate(all="ignore"):
                    raw = pred[:, :3].sum(axis=1) / tot
                use = good[sel, j] & np.isfinite(raw) & (tot > 0)
                r = np.where(use, np.clip(raw, LO, HI), 1.0)
                clipped += int((use & ((raw < LO) | (raw > HI))).sum())
                cnt += int(use.sum())
            R[sel, j] = r
        rows.append(dict(month=m0.strftime("%Y-%m"), n_closed_dates=nu, n_tickers_fitted=n_fit, mean_R=float(np.nanmean(R[(P.mon == mo) & P.pres])), share_clipped=clipped / cnt if cnt else 0.0))
    return dict(F=P.k8 * R, M=R, params=pd.DataFrame(rows))
