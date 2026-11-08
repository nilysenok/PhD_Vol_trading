"""V5 лист 1a, G14 (диагностика, в выбор не входит): охват и калибровка score лучшей GS (план п. 4, 6.3).
(а) QLIKE-оптимальный ex post вес w в M^(1−w)·F^w; (б) то же в K8^(1−w)·F^w; 90 % интервал — круговой блочный бутстрап по датам (блок 20, B = 500, seed 42; пул строк M16 окна). (в) корреляция ln(RV/F) и ln(RV/M).
Калибровка: среднее s = RV/F − 1 по квинтилям φ₁ (ln RV_d − среднее ln RV за прошлые 252 дня) и φ₄ (среднее s к K8 у остальных бумаг в день d); границы квинтилей года Y — по строкам с датой признака < 1 января Y."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.d import stats_d as SD
from models_v2.s12 import v5_0_io as IO, v5_gs_eval as EV

B_BOOT = 500
WINS = ("2017_2021", "2022_2025", "all")


def w_hat(lr: np.ndarray, delta: np.ndarray, it: int = 30) -> float:
    """argmin_w mean[RV/F_w + ln F_w], ln F_w = ln(основа) + w·δ, δ = ln(F/основа): Ньютон на выпуклой функции; lr = RV/основа."""
    w = 0.0
    for _ in range(it):
        r = lr * np.exp(-w * delta)
        g, h = np.mean((1.0 - r) * delta), np.mean(r * delta * delta)
        if h <= 1e-18:
            break
        w = float(np.clip(w - g / h, -3.0, 4.0))
    return w


def boot_w(lr, delta, dates, idx) -> np.ndarray:
    """w по бутстрап-выборкам дат: idx — (B, n_dates); строки даты берутся целиком."""
    order = np.argsort(dates, kind="stable")
    d_sorted = dates[order]
    ud, start = np.unique(d_sorted, return_index=True)
    end = np.append(start[1:], len(d_sorted))
    lr, delta = lr[order], delta[order]
    out = np.empty(len(idx))
    for b, row in enumerate(idx):
        sel = np.concatenate([np.arange(start[j], end[j]) for j in row])
        out[b] = w_hat(lr[sel], delta[sel])
    return out


def encompass(c: pd.DataFrame, best: str) -> pd.DataFrame:
    rows = []
    for w in WINS:
        m = c[(c["group"] == "M16") & EV.wmask(c, w)]
        dates = m["date"].to_numpy()
        idx = SD.block_idx(len(np.unique(dates)), B_BOOT)
        for name, base in (("M", "M"), ("K8", "K8")):
            lr = (m["rv_tplus"] / m[base]).to_numpy(float)
            delta = np.log(m[best] / m[base]).to_numpy(float)
            bw = boot_w(lr, delta, dates, idx)
            rows.append({"best": best, "base": name, "window": w, "w_hat": w_hat(lr, delta), "w_lo90": float(np.percentile(bw, 5)), "w_hi90": float(np.percentile(bw, 95)), "n_rows": len(m), "n_dates": len(np.unique(dates)),
                         "corr_ln_rv_over_F_ln_rv_over_M": float(np.corrcoef(np.log(m["rv_tplus"] / m[best]), np.log(m["rv_tplus"] / m["M"]))[0, 1]),
                         "same_sign_share": float(np.mean(np.sign(np.log(m["rv_tplus"] / m[best])) == np.sign(np.log(m["rv_tplus"] / m["M"]))))})
    return pd.DataFrame(rows)


def features(P) -> dict:
    """φ₁, φ₄ на оси τ (матрицы): известны после закрытия дня τ; используются для прогноза СЛЕДУЮЩЕЙ строки бумаги."""
    lr = pd.DataFrame(np.log(np.where(P.pres, P.rv, np.nan)))
    phi1 = (lr - lr.rolling(252, min_periods=126).mean().shift(1)).to_numpy()
    s8 = np.where(P.pres, P.rv / P.k8 - 1.0, np.nan)
    cnt = np.isfinite(s8).sum(axis=1, keepdims=True)
    others = (np.nansum(s8, axis=1, keepdims=True) - np.nan_to_num(s8)) / np.maximum(cnt - 1, 1)
    phi4 = np.where(np.isfinite(s8) & (cnt > 1), others, np.nan)
    return {"phi1": phi1, "phi4": phi4}


def calib(P, models: dict) -> pd.DataFrame:
    """models: {имя: матрица F (N, m)}. Строка — (бумага, следующая строка): признак на дне τ_prev, исход s строки. Квинтили года Y — по признакам с датой < 1 января Y; M16."""
    ft = features(P)
    pairs = []
    for j, t in enumerate(P.tick):
        i = np.flatnonzero(P.pres[:, j])
        pairs.append(pd.DataFrame({"ticker": t, "n_feat": i[:-1], "n_out": i[1:], "j": j}))
    pr = pd.concat(pairs, ignore_index=True)
    pr = pr[pr["ticker"].isin(IO.M16)].copy()
    pr["date_feat"] = P.axis.to_numpy()[pr["n_feat"].to_numpy()]
    pr["year_out"] = P.axis.year.to_numpy()[pr["n_out"].to_numpy()]
    rows = []
    for fname, mat in ft.items():
        pr[fname] = mat[pr["n_feat"].to_numpy(), pr["j"].to_numpy()]
        for y in sorted(pr["year_out"].unique()):
            past = pr.loc[pr["date_feat"] < pd.Timestamp(year=int(y), month=1, day=1), fname].dropna()
            if len(past) < 500:
                continue
            q = np.quantile(past, [0.2, 0.4, 0.6, 0.8])
            sub = pr[(pr["year_out"] == y) & pr[fname].notna()]
            bins = np.searchsorted(q, sub[fname].to_numpy()) + 1
            for mname, F in models.items():
                s = P.rv[sub["n_out"].to_numpy(), sub["j"].to_numpy()] / F[sub["n_out"].to_numpy(), sub["j"].to_numpy()] - 1.0
                rows.append(pd.DataFrame({"feature": fname, "model": mname, "year": y, "quintile": bins, "s": s}))
    a = pd.concat(rows, ignore_index=True)
    out = []
    for wname, (lo, hi) in {"2018_2021": (2018, 2021), "2022_2025": (2022, 2025)}.items():
        g = a[(a["year"] >= lo) & (a["year"] <= hi)].groupby(["feature", "model", "quintile"])["s"].agg(["mean", "std", "size"]).reset_index()
        g["window"] = wname
        g["se_naive"] = g["std"] / np.sqrt(g["size"])
        out.append(g.rename(columns={"mean": "mean_s", "size": "n"}).drop(columns=["std"]))
    return pd.concat(out, ignore_index=True)
