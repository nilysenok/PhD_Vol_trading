"""Волна D, поток LGBM: составные ряды (годовой выбор окна, переключение по режиму), crit_wf, метрики переходов. Выбор года Y — только по pen_Y/фолдам с целями < 01.01.Y."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6.lgbm import crit, grid

FIRST_WF = 2018
CAP_DELAY = 20


def fold_frame(res: dict) -> pd.DataFrame:
    """{(Y, y): (val, q)} -> DataFrame Y, y, row, q (по строкам проверки)."""
    parts = [pd.DataFrame({"Y": Y, "y": y, "row": v[0], "q": v[1]}) for (Y, y), v in sorted(res.items()) if v is not None]
    return pd.concat(parts, ignore_index=True)


def crit_of(df: pd.DataFrame, name: str) -> dict:
    """crit_pen/crit_mean/by_year по QLIKE-фолдам (среднее по строкам проверки фолда) — тот же формат, что `crit.crit`."""
    m = df.groupby(["Y", "y"])["q"].mean()
    return crit.crit(crit.frame({(name, int(Y), int(y)): float(v) for (Y, y), v in m.items()}, name))


def crit_wf(fc: pd.DataFrame, pen: float = grid.PEN, first: int = FIRST_WF) -> float:
    """Для Y = first…последний год: pen_Y по годовым QLIKE ряда за годы 2017…Y−1 (реализованные цели < 01.01.Y); crit_wf = среднее pen_Y."""
    q = np.asarray(metrics.qlike(fc["rv"], fc["sigma2"]), float)
    yr = pd.to_datetime(fc["target_date"]).dt.year.to_numpy()
    ym = pd.Series(q).groupby(yr).mean()
    ym = ym[ym.index >= 2017]
    pens = [crit.pen_year(ym[ym.index < Y].to_numpy(), pen) for Y in range(first, int(ym.index.max()) + 1) if (ym.index < Y).any()]
    return float(np.mean(pens))


def frequency_verdict(cw_d: float, cw_w: float, cw_m: float, thr: float = 0.0005) -> str:
    """RD2: ежедневное — только если crit_wf(d) < crit_wf(w) − thr; еженедельное — если crit_wf(w) < crit_wf(m) − thr; иначе ежемесячное."""
    if cw_d < cw_w - thr:
        return "daily"
    return "weekly" if cw_w < cw_m - thr else "monthly"


def pick_by_year(pens: pd.DataFrame) -> dict:
    """pens: Y × кандидат (pen_Y по целям < 01.01.Y). Год -> кандидат с min pen_Y; ничья — раньше по порядку столбцов."""
    return {int(Y): pens.columns[int(np.argmin(pens.loc[Y].to_numpy(float)))] for Y in pens.index}


def compose_by_year(fcs: dict, pick: dict) -> pd.DataFrame:
    """Ряд кандидата pick[Y] на датах целей года Y (по году даты цели, как годовой выбор C)."""
    base = next(iter(fcs.values())).copy()
    yr = pd.to_datetime(base["target_date"]).dt.year.to_numpy()
    for Y, n in pick.items():
        m = yr == Y
        base.loc[m, "sigma2"] = fcs[n].loc[m, "sigma2"].to_numpy()
    return base


def regime_scores(dfs: dict, st_rows: np.ndarray) -> dict:
    """{(Y, r): Series кандидат -> среднее по фолдам (Y, y) QLIKE на днях проверки режима r}; r = 1 стресс, 0 спокойно (NaN режима — спокойно)."""
    out = {}
    for r in (0, 1):
        for n, df in dfs.items():
            flag = np.nan_to_num(st_rows[df["row"].to_numpy()], nan=0.0) == 1
            g = df[flag == bool(r)].groupby(["Y", "y"])["q"].mean().groupby(level=0).mean()
            for Y, v in g.items():
                out.setdefault((int(Y), r), {})[n] = float(v)
    return {k: pd.Series(v) for k, v in out.items()}


def pick_by_regime(scores: dict, names: list, years) -> tuple:
    """-> (calm {Y: имя}, stress {Y: имя}); нет данных режима в фолдах года — первое имя списка (расширяющееся)."""
    res = ({}, {})
    for r in (0, 1):
        for Y in years:
            s = scores.get((Y, r))
            res[r][Y] = names[0] if s is None or not s.notna().any() else s.reindex(names).idxmin()
    return res


def switch_series(fcs: dict, calm: dict, stress: dict, st_grid: np.ndarray, years: np.ndarray) -> tuple:
    """Составной ряд: на дату d (строка сетки) с флагом режима st_grid[d] и годом years[d] — ряд calm[Y] или stress[Y]; NaN режима — спокойно.
    -> (DataFrame, имя активной конфигурации по дням)."""
    base = next(iter(fcs.values())).copy()
    act = []
    for i, (s, Y) in enumerate(zip(np.nan_to_num(st_grid, nan=0.0), years)):
        n = (stress if s == 1 else calm)[int(Y)]
        base.iloc[i, base.columns.get_loc("sigma2")] = fcs[n]["sigma2"].iloc[i]
        act.append(n)
    return base, np.array(act)


def switch_folds(dfs: dict, calm: dict, stress: dict, st_rows: np.ndarray) -> pd.DataFrame:
    """Фолдовые QLIKE составной конфигурации: для года выбора Y и строки проверки — ряд calm[Y] / stress[Y] по флагу режима строки."""
    parts = []
    for Y in sorted(calm):
        a, b = dfs[calm[Y]], dfs[stress[Y]]
        m = a[a.Y == Y].merge(b[b.Y == Y], on=["Y", "y", "row"], suffixes=("_c", "_s"))
        s = np.nan_to_num(st_rows[m["row"].to_numpy()], nan=0.0) == 1
        parts.append(pd.DataFrame({"Y": m.Y, "y": m.y, "row": m.row, "q": np.where(s, m.q_s, m.q_c)}))
    return pd.concat(parts, ignore_index=True)


def transitions(fc: pd.DataFrame, st_grid: np.ndarray) -> dict:
    """Начала стресса (0 -> 1) и выходы (1 -> 0) по дате d: число, медианная задержка реакции (дней до σ̂²/RV ≥ 0,5, кэп 20),
    QLIKE и доля занижения (σ̂² < RV) в первые 5 дней после входа и после выхода."""
    s = np.asarray(st_grid, float)
    sig, rv = fc["sigma2"].to_numpy(float), fc["rv"].to_numpy(float)
    q = np.asarray(metrics.qlike(rv, sig), float)
    on = [i for i in range(1, len(s)) if s[i] == 1 and s[i - 1] == 0]
    off = [i for i in range(1, len(s)) if s[i] == 0 and s[i - 1] == 1]

    def delay(i):
        ok = np.flatnonzero(sig[i:i + CAP_DELAY + 1] / rv[i:i + CAP_DELAY + 1] >= 0.5)
        return float(ok[0]) if len(ok) else float(CAP_DELAY)
    w = lambda ix: np.concatenate([np.arange(i, min(i + 5, len(s))) for i in ix]) if ix else np.array([], int)
    out = {"n_onsets": len(on), "n_exits": len(off), "delay_median": float(np.median([delay(i) for i in on])) if on else float("nan")}
    for lab, ix in (("on5", w(on)), ("off5", w(off))):
        out["q_" + lab] = float(q[ix].mean()) if len(ix) else float("nan")
        out["under_" + lab] = float((sig[ix] < rv[ix]).mean()) if len(ix) else float("nan")
    return out
