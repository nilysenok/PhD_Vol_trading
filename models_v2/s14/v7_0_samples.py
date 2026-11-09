"""V7-0, G5: число дневных примеров (закрытых пар) на каждую дату помесячной переоценки 2017-01 … 2026-02. Дата переоценки D = первая дата d общего ряда в месяце; закрытая пара — строка с t⁺ ≤ D.
(а) корректор HAR-J: пары с первой даты, где есть сохранённый H и окно 5 сессий; (б) мета-модель с прогнозами K2/K3: пары с первой даты, где сохранены конечные σ̂² K2 и K3.
Для справки: `har_fit_pairs` — пар в обучении самого HAR-J (колонка n_pairs файла K1; не пары корректора) и `a_potential_2015` — сколько пар (H, RV) было бы, если бы H пересчитать с 2015-07-07 (ряд не сохранён, K1 не переобучается). Только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s14 import v7_0_ctrl as C, v7_0_io as IO

P_LIST = [200, 500, 1000, 2000, 5000]
W_MIN = 5
H_POTENTIAL_FROM = pd.Timestamp("2015-07-07")


def reestimation_dates(idx: pd.DatetimeIndex, first="2017-01", last="2026-02") -> list:
    """Первая дата ряда в каждом месяце first…last."""
    out = []
    for p in pd.period_range(first, last, freq="M"):
        m = idx[(idx >= p.start_time) & (idx <= p.end_time)]
        if len(m):
            out.append(m[0])
    return out


def closed(t_plus: pd.Series, mask: np.ndarray, D) -> int:
    return int(((t_plus.to_numpy() <= np.datetime64(D)) & mask).sum())


def first_with_window(idx: pd.DatetimeIndex, sessions: pd.DatetimeIndex) -> pd.Timestamp:
    """Первая дата d, у которой в канонических сессиях ≤ d есть W_MIN дней (окно 5 дней)."""
    pos = sessions.searchsorted(idx, side="right")
    return idx[pos >= W_MIN][0]


def table() -> tuple:
    w = C.aligned()
    k = pd.read_parquet(IO.PAN / "k23_SBER.parquet")
    k = k[np.isfinite(k["sigma2_k2"]) & np.isfinite(k["sigma2_k3"])]
    d = pd.read_parquet(IO.DAILY)
    d.index = pd.DatetimeIndex(pd.to_datetime(d.index))
    h = pd.read_parquet(IO.PAN / "har_SBER.parquet")
    h.index = pd.DatetimeIndex(pd.to_datetime(h["d"]))
    first_a = first_with_window(w.index, d.index)
    first_b = pd.DatetimeIndex(pd.to_datetime(k["date"])).min()
    nxt = pd.Series(d.index[1:], index=d.index[:-1])
    seg_ok = (d["seg"].shift(-1) == d["seg"]).iloc[:-1]
    pot = nxt[seg_ok.to_numpy() & (nxt.index >= H_POTENTIAL_FROM)]
    rows = []
    for D in reestimation_dates(w.index):
        rows.append(dict(section="month", date=D.date().isoformat(), n_a=closed(w["t_plus"], (w.index >= first_a).astype(bool), D),
                         n_b=closed(pd.Series(pd.to_datetime(k["t_plus"]).to_numpy()), (pd.to_datetime(k["date"]) >= first_b).to_numpy(), D),
                         a_potential_2015=int((pot.to_numpy() <= np.datetime64(D)).sum()), har_fit_pairs=int(h.loc[D, "n_pairs"])))
    m = pd.DataFrame(rows)
    return m, dict(first_a=first_a.date().isoformat(), first_b=first_b.date().isoformat())


def ratios(m: pd.DataFrame) -> pd.DataFrame:
    """Шаблон «параметры против дней»: n_дней / P на первую дату оценки каждого года."""
    y = m[m["date"].str[5:7] == "01"]
    rows = [dict(section="ratio", date=r["date"], n_a=r["n_a"], n_b=r["n_b"], P=p, ratio_a=r["n_a"] / p, ratio_b=r["n_b"] / p) for _, r in y.iterrows() for p in P_LIST]
    return pd.DataFrame(rows)


def samples() -> tuple:
    m, info = table()
    return pd.concat([m, ratios(m)], ignore_index=True), info
