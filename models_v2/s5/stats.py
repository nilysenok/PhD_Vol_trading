"""Метрики диагностики (волна A): QLIKE через models_v2.metrics, MZ-регрессии, калибровка, крупные ошибки, доля «лучше»."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s5 import common

BIG_LN, BIG_Q = 1.0, 1.0


def ols1(x, y) -> tuple:
    """y = a + b x: (a, b, R²). Вырожденный x — NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx = ((x - x.mean()) ** 2).sum()
    if vx == 0:
        return float("nan"), float("nan"), float("nan")
    b = ((x - x.mean()) * (y - y.mean())).sum() / vx
    a = y.mean() - b * x.mean()
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ((y - a - b * x) ** 2).sum() / ss if ss > 0 else float("nan")
    return float(a), float(b), float(r2)


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def full_metrics(rv, s2) -> dict:
    rv, s2 = np.asarray(rv, float), np.asarray(s2, float)
    if len(rv) == 0:
        raise ValueError("пустой срез")
    e, el = np.log(rv) - np.log(s2), rv - s2
    q = metrics.qlike(rv, s2)
    a_l, b_l, r2_l = ols1(np.log(s2), np.log(rv))
    a_v, b_v, r2_v = ols1(s2, rv)
    return {"n": len(rv), "qlike": float(q.mean()), "rmse_ln": float(np.sqrt((e ** 2).mean())), "mae_ln": float(np.abs(e).mean()),
            "rmse_lvl": float(np.sqrt((el ** 2).mean())), "mae_lvl": float(np.abs(el).mean()),
            "r2_mz_ln": r2_l, "mz_alpha_ln": a_l, "mz_beta_ln": b_l, "r2_mz_lvl": r2_v,
            "bias_ln": float(e.mean()), "bias_lvl": float(el.mean()), "mean_r": float((rv / s2).mean()),
            "corr_ln": corr(np.log(rv), np.log(s2)), "corr_lvl": corr(rv, s2),
            "cal_alpha": a_v, "cal_beta": b_v,
            "big_ln_share": float((np.abs(e) > BIG_LN).mean()), "big_qlike_share": float((q > BIG_Q).mean())}


def win_share(loss_b2, loss_harj) -> float:
    """Доля наблюдений, где Б-2 лучше (меньше потеря); ничья считается за половину."""
    d = np.asarray(loss_b2, float) - np.asarray(loss_harj, float)
    if len(d) == 0:
        raise ValueError("пустой срез")
    return float(((d < 0).sum() + 0.5 * (d == 0).sum()) / len(d))


def compare(df: pd.DataFrame, mask) -> list:
    """Строки по двум моделям для среза mask датафрейма PREDICTIONS."""
    sub, out = df[mask], []
    ws = win_share(sub["qlike_b2"], sub["qlike_harj"])
    dq = float(sub["qlike_b2"].mean() - sub["qlike_harj"].mean())
    for name, col in (("b2", "s2_b2"), ("harj", "s2_harj")):
        out.append({"model": name, **full_metrics(sub["rv"], sub[col]), "delta_qlike_b2_minus_harj": dq, "win_share_b2": ws})
    return out


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, m in common.period_masks(df["target_date"]).items():
        rows += [{"scope": "period", "label": lab, **r} for r in compare(df, m)]
    y = df["target_date"].dt.year
    for k in sorted(y.unique()):
        rows += [{"scope": "year", "label": str(k), **r} for r in compare(df, (y == k).to_numpy())]
    q = df["target_date"].dt.to_period("Q").astype(str)
    for k in sorted(q.unique()):
        rows += [{"scope": "quarter", "label": k, **r} for r in compare(df, (q == k).to_numpy())]
    return pd.DataFrame(rows)


def rolling(df: pd.DataFrame, win: int = 252) -> pd.DataFrame:
    rows = []
    for i in range(win - 1, len(df)):
        s = df.iloc[i - win + 1:i + 1]
        rows.append({"end_target_date": s["target_date"].iloc[-1], "qlike_b2": s["qlike_b2"].mean(), "qlike_harj": s["qlike_harj"].mean(),
                     "win_share_b2": win_share(s["qlike_b2"], s["qlike_harj"]), "bias_ln_b2": s["err_ln_b2"].mean(),
                     "bias_ln_harj": s["err_ln_harj"].mean()})
    return pd.DataFrame(rows)


def annual(diag: pd.DataFrame) -> pd.DataFrame:
    y = diag[diag.scope == "year"]
    b, h = y[y.model == "b2"].set_index("label"), y[y.model == "harj"].set_index("label")
    cols = ["n", "qlike", "rmse_ln", "rmse_lvl", "mae_ln", "bias_ln", "mean_r", "r2_mz_ln", "r2_mz_lvl", "cal_beta", "big_ln_share", "big_qlike_share"]
    out = pd.DataFrame({"year": b.index})
    for c in cols:
        out[c + "_b2"] = b[c].to_numpy()
        out[c + "_harj"] = h[c].to_numpy()
    out["delta_qlike_b2_minus_harj"] = b["delta_qlike_b2_minus_harj"].to_numpy()
    out["win_share_b2"] = b["win_share_b2"].to_numpy()
    return out.drop(columns=["n_harj"]).rename(columns={"n_b2": "n"})
