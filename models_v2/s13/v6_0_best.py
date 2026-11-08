"""V6-0, G3: BEST_SBER — точный актуальный лучший результат SBER на общих датах. Общие даты — пересечение дат всех PIT-сопоставимых рядов (контроли + класс «PIT-сопоставим»); метрики по окнам на них;
лучший ряд по окну среди PIT-сопоставимых и среди всех рядов (потолки помечены). Ряд-потолок с неполным покрытием общих дат в сравнение «все ряды» не входит (строка `partial` с его числом дат). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_class as CL, v6_0_q as Q

CTRL = ["K1", "K2", "K3", "M", "K8", "D2"]
SETS = ("PIT-сопоставимые", "все ряды (потолки помечены)")


def split_sets(prior: pd.DataFrame) -> tuple:
    """-> (pit, ceil): списки рядов; контроли K1…D2 входят в PIT (замороженные, оговорка K2/K3)."""
    p = prior[prior["class"].isin([CL.PIT, CL.CEIL])]
    return CTRL + [s for s, c in zip(p["series"], p["class"]) if c == CL.PIT], [s for s, c in zip(p["series"], p["class"]) if c == CL.CEIL]


def common_mask(w: pd.DataFrame, cols: list) -> np.ndarray:
    x = w[cols].to_numpy(float)
    return (np.isfinite(x) & (x > 0)).all(axis=1) & np.isfinite(w["rv"].to_numpy()) & (w["rv"].to_numpy() > 0)


def common_rows(w: pd.DataFrame, pit: list, full: list, partial: list) -> pd.DataFrame:
    rows = []
    for name, cols in ((SETS[0], pit), (SETS[1], pit + full)):
        m = common_mask(w, cols)
        d = w.index[m]
        rows.append(dict(section="common", set=name, n_series=len(cols), n_dates=int(m.sum()), first_d=str(d.min().date()), last_d=str(d.max().date()), first_t_plus=str(w["t_plus"][m].min().date()), last_t_plus=str(w["t_plus"][m].max().date()),
                         note="" if name == SETS[0] else "исключено рядов с неполным покрытием: %d" % len(partial)))
    return pd.DataFrame(rows)


def metrics_table(w: pd.DataFrame, series: list, cls: dict, mask: np.ndarray) -> pd.DataFrame:
    rows, tp = [], w["t_plus"].to_numpy()
    for win in Q.WIN:
        m = mask & Q.wmask(tp, win)
        if m.any():
            rows += [dict(section="metrics", series=s, class_=cls[s], window=win, **Q.metr_row(w["rv"].to_numpy()[m], w[s].to_numpy()[m])) for s in series]
    return pd.DataFrame(rows)


def partial_rows(w: pd.DataFrame, partial: list, cls: dict) -> pd.DataFrame:
    rows = []
    for s in partial:
        m = common_mask(w, [s])
        rows.append(dict(section="partial", series=s, class_=cls[s], window="all", n=int(m.sum()), qlike=float(Q.ql(w["rv"].to_numpy()[m], w[s].to_numpy()[m]).mean()) if m.any() else np.nan, note="неполное покрытие дат, в лучший ряд не входит"))
    return pd.DataFrame(rows)


def best_rows(met: pd.DataFrame, pit: list, full: list) -> pd.DataFrame:
    rows = []
    for win, g in met.groupby("window", sort=False):
        for name, cols in ((SETS[0], pit), (SETS[1], pit + full)):
            x = g[g["series"].isin(cols)].sort_values("qlike")
            b, s2 = x.iloc[0], (x.iloc[1] if len(x) > 1 else None)
            rows.append(dict(section="best", set=name, window=win, series=b["series"], class_=b["class_"], qlike=b["qlike"], n=b["n"], second=s2["series"] if s2 is not None else "", second_qlike=s2["qlike"] if s2 is not None else np.nan))
    return pd.DataFrame(rows)


def best_table(w: pd.DataFrame, prior: pd.DataFrame) -> tuple:
    pit, ceil = split_sets(prior)
    cls = {s: c for s, c in zip(prior["series"], prior["class"])} | {s: "контроль" for s in CTRL}
    m_pit = common_mask(w, pit)
    full = [s for s in ceil if common_mask(w, pit + [s]).sum() == m_pit.sum()]
    partial = [s for s in ceil if s not in full]
    met = metrics_table(w, pit + full, cls, m_pit)
    best = best_rows(met, pit, full)
    com = common_rows(w, pit, full, partial)
    out = pd.concat([com, best, partial_rows(w, partial, cls), met], ignore_index=True).rename(columns={"class_": "class"})
    return out, dict(met=met, pit=pit, ceil=ceil, full=full, partial=partial, mask_pit=m_pit, best=best, common=com)
