"""V4-A, G6: комплементарность пар прогнозов (Ж-222 п. 2, пункты 1–10) — диагностика на готовых рядах. Ошибки: e = RV − F; относительная RV/F − 1; логарифмическая ln RV − ln F.
Режим «кризис» — день d, когда RV IMOEX(d) выше 90-го процентиля её прошлых 252 дней (только диагностика; в прогноз не идёт)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s11 import v4_io as IO

PAIRS = (("H", "K2"), ("H", "K3"), ("K2", "K3"))
COL = {"H": "h", "K2": "k2", "K3": "k3"}
MIN_N = 30


def crisis_flag(rv: pd.Series = None) -> pd.Series:
    """Индекс — дата d; True, если RV IMOEX в день d > квантиль 0,9 по 252 предыдущим дням (строго до d); нет истории — False."""
    rv = IO.load_daily("IMOEX")["rv"] if rv is None else rv
    thr = rv.shift(1).rolling(252, min_periods=252).quantile(0.9)
    return (rv > thr).fillna(False)


def _c(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


def top_share(dq: np.ndarray, frac: float) -> float:
    """Доля суммарной разности потерь, приходящаяся на frac крайних дней по |dq| (NaN, если сумма ≈ 0)."""
    tot = float(dq.sum())
    k = max(1, int(round(frac * len(dq))))
    top = dq[np.argsort(-np.abs(dq))[:k]]
    return float(top.sum() / tot) if abs(tot) > 1e-12 else float("nan")


def pair_stats(rv, a, b) -> dict:
    """Показатели пары (A, B) на одном срезе; dq = QLIKE(A) − QLIKE(B) (минус — A лучше)."""
    rv, a, b = (np.asarray(x, float) for x in (rv, a, b))
    n = len(rv)
    if n < MIN_N:
        return {"n": n}
    la, lb, lr = np.log(a), np.log(b), np.log(rv)
    qa, qb = metrics.qlike(rv, a), metrics.qlike(rv, b)
    dq, lrat = qa - qb, lb - la
    return {"n": n, "corr_err_rv": _c(rv - a, rv - b), "corr_err_rel": _c(rv / a - 1, rv / b - 1), "corr_err_log": _c(lr - la, lr - lb), "corr_q": _c(qa, qb), "corr_dq_logratio": _c(dq, lrat),
            "share_opposite": float(np.mean(np.sign(lr - la) != np.sign(lr - lb))), "share_both_under": float(np.mean((rv > a) & (rv > b))), "share_both_over": float(np.mean((rv < a) & (rv < b))),
            "mean_dq": float(dq.mean()), "gain_top1": top_share(dq, 0.01), "gain_top5": top_share(dq, 0.05), "mean_logratio": float(lrat.mean()), "std_logratio": float(lrat.std(ddof=1))}


def slices(a: pd.DataFrame, crisis: pd.Series) -> list:
    """(окно, режим, маска): 4 окна × {all, calm, crisis} и годы (Y<год>, all) для динамики log(M/H) по годам; окно — год цели t⁺."""
    yr = a["t_plus"].dt.year.to_numpy()
    cr = a["date"].map(crisis).fillna(False).to_numpy(bool)
    out = []
    for w, (lo, hi) in IO.WINDOWS.items():
        m = (yr >= lo) & (yr <= hi)
        out += [(w, "all", m), (w, "calm", m & ~cr), (w, "crisis", m & cr)]
    out += [("Y%d" % y, "all", yr == y) for y in range(2017, 2027)]
    return out


def compl_table(tickers=None) -> pd.DataFrame:
    crisis, rows = crisis_flag(), []
    for t in tickers or IO.T17:
        a = IO.aligned(t)
        for w, reg, m in slices(a, crisis):
            for x, y in PAIRS:
                s = a[m]
                rows.append({"ticker": t, "group": IO.P.group(t), "pair": "%s-%s" % (x, y), "window": w, "regime": reg, **pair_stats(s["rv"], s[COL[x]], s[COL[y]])})
    return pd.DataFrame(rows)
