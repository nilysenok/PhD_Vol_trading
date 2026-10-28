"""Общие функции этапа 2: загрузка прогнозов окон, потери, DM (из etap0_qlike_dm)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).parent))
from etap0_qlike_dm import by_year, dm_simple, dm_hac_time, in_years  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
WF = ROOT / "results/final/data/predictions_walkforward"
WIN = ["E", "R1", "R2", "R3"]
FILES = {1: "walkforward_all_h1", 5: "walkforward_all_h5_corrected", 22: "walkforward_all_h22"}


def ql(y, p):
    p = np.clip(p, 1e-10, None)
    return y / p - np.log(y / p) - 1


def qmean(y, p):
    return float(np.mean(ql(np.asarray(y), np.asarray(p))))


def load_windows():
    return pd.read_parquet(ROOT / "results/etap2/har_windows.parquet")


def saved(h):
    d = pd.read_parquet(WF / (FILES[h] + ".parquet"))
    d["date"] = pd.to_datetime(d["date"])
    return d


def wide(w, h, kind, col="p"):
    """таблица (step,date,ticker,y) x spec"""
    s = w[(w.h == h) & (w.kind == kind)]
    x = s.pivot_table(index=["step", "date", "ticker"], columns="spec", values=col)
    y = s[s.spec == "E"].set_index(["step", "date", "ticker"])["y"]
    x["y"] = y
    x.index = x.index.set_levels(pd.to_datetime(x.index.levels[1]), level=1)
    return x


def pooled_loss(y, p):
    """Series потерь с индексом (date,ticker), только y>0 и годы 2017-2026."""
    l = pd.Series(ql(y.values, p.values), index=y.index)
    l = l[y.values > 0]
    return l.droplevel(0) if l.index.nlevels == 3 else l


def summary(l):
    """среднее годовых 2017-2025 и значение 2026 отдельно"""
    yr = l.index.get_level_values("date").year
    return by_year(l), float(l[yr == 2026].mean())


def dm_row(l_other, l_hyb, h):
    """t>0 - гибрид лучше. (t_nominal, t_hac, p_nominal, p_hac) на 2017-2025"""
    a, b = in_years(l_other), in_years(l_hyb)
    t0 = dm_simple(a, b); th = dm_hac_time(a, b, h) if h > 1 else t0
    p = lambda t: float(2 * (1 - stats.norm.cdf(abs(t))))
    return t0, th, p(t0), p(th)


def holm(ps):
    o = np.argsort(ps); m = len(ps); out = np.zeros(m); mx = 0
    for r, i in enumerate(o):
        mx = max(mx, (m - r) * ps[i]); out[i] = min(1, mx)
    return out
