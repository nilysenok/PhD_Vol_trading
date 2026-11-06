"""Внутренний критерий потока LGBM (объявлен до счёта): для года выбора Y и фолдов LOYO y ∈ [2015, Y−1] (цели < 01.01.Y):
pen_Y = mean_y q + 0,25 · (max_y q − mean_y q); crit_pen = среднее pen_Y по Y = 2017…2026. crit_mean — то же без штрафа (как `b2/loyo`)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s6.lgbm import grid


def frame(res: dict, name: str) -> pd.DataFrame:
    """{(name, Y, y): q} -> DataFrame Y, y, q."""
    return pd.DataFrame([{"Y": k[1], "y": k[2], "q": v} for k, v in res.items() if k[0] == name]).sort_values(["Y", "y"], ignore_index=True)


def pen_year(q: np.ndarray, pen: float = grid.PEN) -> float:
    q = np.asarray(q, float)
    q = q[np.isfinite(q)]
    if len(q) == 0:
        return float("nan")
    return float(q.mean() + pen * (q.max() - q.mean()))


def by_year(df: pd.DataFrame, pen: float = grid.PEN) -> pd.DataFrame:
    g = df.groupby("Y")["q"]
    out = pd.DataFrame({"mean": g.mean(), "worst": g.max(), "n_folds": g.count()})
    out["pen"] = [pen_year(df.loc[df.Y == Y, "q"].to_numpy(), pen) for Y in out.index]
    return out


def crit(df: pd.DataFrame, pen: float = grid.PEN) -> dict:
    a = by_year(df, pen)
    return {"crit_pen": float(a["pen"].mean()), "crit_mean": float(a["mean"].mean()), "worst_fold": float(a["worst"].max()),
            "by_year": {int(Y): float(v) for Y, v in a["pen"].items()}}


def select(frames: dict, points: tuple) -> tuple:
    """frames: {индекс точки: DataFrame фолдов} -> ({Y: Hp}, таблица Y×точка pen). Ничья |Δ| ≤ 1e-12 — grid.order_key."""
    pens = pd.DataFrame({i: by_year(f)["pen"] for i, f in frames.items()})
    chosen = {}
    for Y, row in pens.iterrows():
        ok = row[np.isfinite(row)]
        tied = [i for i, v in ok.items() if v - ok.min() <= 1e-12]
        chosen[int(Y)] = points[min(tied, key=lambda i: grid.order_key(points[i]))]
    return chosen, pens
