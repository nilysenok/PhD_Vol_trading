"""V6 лист Z3, диагностики 1–3: годы (WHY_YEARS), крайние дни (WHY_TAILS), режимы (WHY_REGIME). Для каждой пары (гибрид, база K3/K2).
Тождества: сумма вкладов лет = общей Δ; топ + остальные = общей Δ; сумма вкладов групп = общей Δ (ряд `total`: Δ считается напрямую, вклад — суммой строк; допуск 1e-12 проверяет `v6_why_check`).
Режимы PIT: порог терцилей на дату d — по уровню прогноза K1 на датах ≤ d (расширяющееся окно); первые 250 дат — группа `no_group`. Группы «по исходу» (RV(t⁺) > F_K1) — только описание."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_q as Q, v6_why_data as DT, v6_why_stat as ST

MIN_HIST = 250
FRACS = (0.01, 0.05)
TERC = ["low", "mid", "high", "no_group"]
OUTC = ["up", "down"]


def years_table(d: dict) -> pd.DataFrame:
    rv, f, rows = d["rv"], d["f"], []
    k1q, N = Q.ql(rv, f["K1"]), len(rv)
    k1_all = float(k1q.mean())
    masks = ST.year_masks(d["tp"])
    for h, b in DT.PAIRS:
        x, cs, ns = ST.dd(rv, f[h], f[b]), 0.0, 0
        for y, m in masks.items():
            if not m.any():
                continue
            c = float(x[m].sum()) / N
            cs, ns = cs + c, ns + int(m.sum())
            rows.append(dict(hybrid=h, base=b, year=y, n=int(m.sum()), weight=m.sum() / N, delta=float(x[m].mean()), delta_pct=ST.pct(x[m].mean(), k1q[m].mean()), contrib=c, contrib_pct=ST.pct(c, k1_all), sign=int(np.sign(x[m].mean()))))
        rows.append(dict(hybrid=h, base=b, year="total", n=ns, weight=ns / N, delta=float(x.mean()), delta_pct=ST.pct(x.mean(), k1_all), contrib=cs, contrib_pct=ST.pct(cs, k1_all), sign=int(np.sign(x.mean()))))
    return pd.DataFrame(rows)


def tails_table(d: dict) -> pd.DataFrame:
    rv, f, rows = d["rv"], d["f"], []
    k1q, N = Q.ql(rv, f["K1"]), len(rv)
    k1_all = float(k1q.mean())
    for h, b in DT.PAIRS:
        x = ST.dd(rv, f[h], f[b])
        order = np.argsort(-np.abs(x), kind="stable")
        for fr in FRACS:
            k = int(round(fr * N))
            keep = np.ones(N, bool)
            keep[order[:k]] = False
            top, delta = x[order[:k]], float(x.mean())
            ct, cr = float(top.sum()) / N, float(x[keep].sum()) / N
            rows.append(dict(hybrid=h, base=b, frac=fr, n_top=k, n_top_worse=int((top > 0).sum()), delta=delta, delta_pct=ST.pct(delta, k1_all), contrib_top=ct, contrib_rest=cr, contrib_top_pct=ST.pct(ct, k1_all),
                             share_top_pct=100 * ct / delta if abs(delta) > 1e-12 else np.nan, delta_without=float(x[keep].mean()), delta_without_pct=ST.pct(x[keep].mean(), k1q[keep].mean())))
    return pd.DataFrame(rows)


def thresholds(f_k1, min_hist: int = MIN_HIST) -> tuple:
    """Пороги терцилей (1/3, 2/3) на каждую дату: по f_k1[:i+1] — только даты ≤ d; для первых min_hist дат — NaN."""
    f_k1 = np.asarray(f_k1, float)
    q = np.full((len(f_k1), 2), np.nan)
    for i in range(min_hist, len(f_k1)):
        q[i] = np.quantile(f_k1[: i + 1], [1 / 3, 2 / 3])
    return q[:, 0], q[:, 1]


def tercile_groups(f_k1) -> np.ndarray:
    q1, q2 = thresholds(f_k1)
    f_k1 = np.asarray(f_k1, float)
    g = np.where(f_k1 <= q1, "low", np.where(f_k1 <= q2, "mid", "high"))
    return np.where(np.isnan(q1), "no_group", g)


def outcome_groups(rv, f_k1) -> np.ndarray:
    return np.where(np.asarray(rv, float) > np.asarray(f_k1, float), "up", "down")


def regime_table(d: dict) -> pd.DataFrame:
    rv, f, rows = d["rv"], d["f"], []
    k1q, N = Q.ql(rv, f["K1"]), len(rv)
    k1_all = float(k1q.mean())
    kinds = {"tercile": (tercile_groups(f["K1"]), TERC), "outcome": (outcome_groups(rv, f["K1"]), OUTC)}
    for h, b in DT.PAIRS:
        x = ST.dd(rv, f[h], f[b])
        for kind, (lab, names) in kinds.items():
            cs, ns = 0.0, 0
            for g in names:
                m = lab == g
                c = float(x[m].sum()) / N
                cs, ns = cs + c, ns + int(m.sum())
                rows.append(dict(hybrid=h, base=b, kind=kind, group=g, n=int(m.sum()), delta=float(x[m].mean()) if m.any() else np.nan, delta_pct=ST.pct(x[m].mean(), k1q[m].mean()) if m.any() else np.nan, contrib=c, contrib_pct=ST.pct(c, k1_all)))
            rows.append(dict(hybrid=h, base=b, kind=kind, group="total", n=ns, delta=float(x.mean()), delta_pct=ST.pct(x.mean(), k1_all), contrib=cs, contrib_pct=ST.pct(cs, k1_all)))
    return pd.DataFrame(rows)
