"""P0, G4 (R3): K1 = HAR-J(б) Ж-214 V1 без изменений, для каждой бумаги и IMOEX:
  ln RV_{t⁺} = b0 + b_d ln RV_d + b_w ln RV_w + b_m ln RV_m + b_J ln(1+J_d),  σ̂² = exp(f + s²/2),  OLS на парах (s, s⁺), s⁺ <= d, один сегмент, расширяющееся окно, переоценка КАЖДЫЙ день.
Код — models_v2/har.py (pairs, ols, predict), как у A0 (s9/hv2_a0). t⁺ — СЛЕДУЮЩИЙ СОБСТВЕННЫЙ торговый день бумаги (строка ряда +1), не календарный t+1 (R2).
Строки сетки: те же, что у A0 (`har.eligible`), плюс строки через остановку 2022 (последний день сегмента 0 → первый день сегмента 1; пара gap, R2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import har as H, metrics
from models_v2.s10 import hp_daily as HD, hp_paths as P

BCOLS = ["b0", "b_d", "b_w", "b_m", "b_J"]


def fit_at(p: H.Panel, i: int):
    """(beta, s2, n) по парам s⁺ <= i этой бумаги. МУТАНТ «по полной выборке» подменяет эту функцию в тесте (класс № 104)."""
    Xt, yt = H.pairs(p, i, H.COLS_HARJ)
    return H.ols(Xt, yt)


def n_pairs(p: H.Panel) -> np.ndarray:
    """Число обучающих пар (s, s+1), s+1 <= i, для каждой строки i (как H.pairs)."""
    n = len(p.rv)
    ok = np.zeros(n, bool)
    ok[:-1] = (p.seg[:-1] == p.seg[1:]) & np.isfinite(p.X[:-1][:, list(H.COLS_HARJ)]).all(axis=1)
    return np.concatenate([[0], np.cumsum(ok)[:-1]])


def grid(p: H.Panel, first_target: str = P.FIRST_TARGET, min_pairs: int = P.MIN_PAIRS) -> np.ndarray:
    """Строки d с целью в следующей строке: цель >= first_target, признаки конечны, пар >= min_pairs. Включает пары через разрыв сегментов."""
    n = len(p.rv)
    i = np.arange(n - 1)
    ok = np.isfinite(p.X[i]).all(axis=1) & (p.dates[i + 1] >= np.datetime64(first_target)) & (n_pairs(p)[i] >= min_pairs)
    return i[ok]


def target_of(p: H.Panel, rows: np.ndarray) -> tuple:
    """(t⁺, RV_{t⁺}): СЛЕДУЮЩАЯ строка собственного ряда бумаги, не календарный t+1. МУТАНТ «календарный t+1» подменяет эту функцию в тесте."""
    return p.dates[rows + 1], p.rv[rows + 1]


def series(d: pd.DataFrame, rows: np.ndarray | None = None) -> pd.DataFrame:
    """Дневной ряд (daily.build_daily) -> таблица K1: d, t⁺, RV_{t⁺}, σ̂², коэффициенты, gap."""
    p = H.make_panel(d)
    rows = grid(p) if rows is None else rows
    out = []
    for i in rows:
        b, s2, n = fit_at(p, int(i))
        sa, sb = H.predict(b, p.X[i, list(H.COLS_HARJ)], s2)
        out.append([p.dates[i], sb, sa, n, s2, *b])
    df = pd.DataFrame(out, columns=["d", "sigma2", "sigma2_a", "n_pairs", "s2", *BCOLS])
    df["target_date"], df["rv"] = target_of(p, rows)
    df = df[["d", "target_date", "rv", "sigma2", "sigma2_a", "n_pairs", "s2", *BCOLS]]
    gd = (df["target_date"] - df["d"]).dt.days
    df["gap_days"], df["gap"] = gd, gd > P.GAP_DAYS
    df["cross_seg"] = p.seg[rows] != p.seg[rows + 1]
    return df


def from_bars(bars: pd.DataFrame, index_days="auto") -> pd.DataFrame:
    """Боевой путь: бары -> K1."""
    return series(HD.core(bars, index_days))


def corrupt_after(bars: pd.DataFrame, d0: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """Все бары ПОСЛЕ дня d0 (даты > d0) умножены на случайный множитель — главный тест обрезки (прогноз на d <= d0 не должен измениться)."""
    rng = np.random.default_rng(seed)
    late = (bars["begin"].dt.normalize() > d0).to_numpy()
    out = bars.copy()
    f = np.exp(rng.normal(0.0, 0.5, int(late.sum())))
    for c in ("open", "high", "low", "close"):
        out.loc[late, c] = out.loc[late, c].to_numpy() * f
    out.loc[late, "volume"] = (out.loc[late, "volume"].to_numpy() * f).astype(out["volume"].dtype)
    return out


def qlike_table(k1: dict) -> pd.DataFrame:
    """QLIKE K1 по бумагам и годам цели (target_date); строки year='all' и 'all_no_gap'."""
    rows = []
    for t, s in k1.items():
        q = pd.Series(metrics.qlike(s["rv"], s["sigma2"]), index=s.index)
        for name, m in [("all", np.ones(len(s), bool)), ("all_no_gap", ~s["gap"].to_numpy())] + \
                       [(str(y), (s["target_date"].dt.year == y).to_numpy()) for y in sorted(s["target_date"].dt.year.unique())]:
            if m.any():
                rows.append({"ticker": t, "group": P.group(t) if t != "IMOEX" else "IMOEX", "year": name, "n": int(m.sum()), "qlike": float(q[m].mean())})
    return pd.DataFrame(rows)
