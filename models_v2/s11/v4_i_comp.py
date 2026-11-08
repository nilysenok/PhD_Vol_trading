"""V4-I: разложение RV = C + J и признаки компонент (Ж-222 п. 10). Определения НЕ меняются: J = max(RV − BV, 0), C = RV − J — как в `models_v2/daily.build_daily` и в HAR-J K1 (`hp_har`, Ж-214); берутся из `daily_<T>.parquet`.
Признаки строки d известны на конец дня d (дневные величины d, окна 5 и 22 дня внутри сегмента — как `daily.add_windows`); цели — C, J в день t⁺ бумаги.
HAR-C: ln C_d, ln C̄_5, ln C̄_22. Модель скачка: событие J > 0 (в данных J уже обнулён: порог значимости проекта — сам знак max(RV − BV, 0)); признаки p̂ — EWMA₂₀ частоты события, J_d/RV_d, ln RV_d.
m̂-признак — EWMA (полураспад 60) величины J по дням J > 0 и число таких дней. `lead` — только для теста-мутации («J_{d+1} в признаках p̂»); боевое 0."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as DY
from models_v2.s11 import v4_io as IO

HL_P, HL_M = 20, 60
HAR_F = ["lc", "lcw", "lcm"]
JMP_F = ["ewf", "jr", "lrv"]
FEATS = HAR_F + JMP_F + ["m_ew", "n_ev"]
ID_TOL = 1e-15            # относительная погрешность тождества C + J = RV (один ulp = 2,2e-16)


def split(rv, j):
    """C = RV − J. Мутация «C := BV» (двойной счёт скачка, класс № 111) подменяет эту функцию в тесте."""
    return np.asarray(rv, float) - np.asarray(j, float)


def identity_ok(rv, c, j, tol: float = ID_TOL) -> bool:
    """C + J = RV: C = RV − J побитно (тот же float), сумма — до одного ulp (побитное равенство суммы в плавающей арифметике невозможно: ~95 % строк)."""
    rv, c, j = (np.asarray(x, float) for x in (rv, c, j))
    return bool(np.array_equal(c, rv - j) and np.all(np.abs(c + j - rv) <= tol * np.abs(rv)))


def _win(x: np.ndarray, seg: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for s in np.unique(seg):
        m = seg == s
        out[m] = DY._window_mean(x[m], w)
    return out


def daily_features(d: pd.DataFrame, lead: int = 0) -> pd.DataFrame:
    """Дневной ряд бумаги (rv, j, seg; индекс — дата) -> компоненты и признаки, индекс — дата d."""
    rv, j = d["rv"].to_numpy(float), d["j"].to_numpy(float)
    c = split(rv, j)
    seg = d["seg"].to_numpy()
    ev = (j > 0).astype(float)
    lg = lambda a: np.log(np.where(a > 0, a, np.nan))
    jp = pd.Series(np.where(ev > 0, j, np.nan), index=d.index)
    pos = jp.dropna()
    m_ew = pos.ewm(halflife=HL_M).mean().reindex(d.index).ffill()
    n_ev = pd.Series(ev, index=d.index).cumsum()
    f = pd.DataFrame({"rv": rv, "j": j, "c": c, "ev": ev, "lc": lg(c), "lcw": lg(_win(c, seg, DY.W_WEEK)), "lcm": lg(_win(c, seg, DY.W_MONTH)),
                      "ewf": pd.Series(ev, index=d.index).ewm(halflife=HL_P).mean().to_numpy(), "jr": np.where(rv > 0, j / np.where(rv > 0, rv, 1.0), np.nan), "lrv": lg(rv),
                      "m_ew": m_ew.to_numpy(), "n_ev": n_ev.to_numpy()}, index=d.index)
    if lead:
        f[FEATS] = f[FEATS].shift(-lead)
    return f


def build(P: pd.DataFrame, dailies: dict, lead: int = 0) -> pd.DataFrame:
    """Панель P (ticker, date, t_plus, …) и {ticker: дневной ряд} -> длинная таблица: ticker, date, FEATS (на d) и цели c_t, j_t, ev_t (в t⁺). Порядок строк — как в P."""
    parts = []
    for t, g in P.groupby("ticker", sort=False):
        f = daily_features(dailies[t], lead)
        at = f[["c", "j", "ev"]].reindex(pd.DatetimeIndex(g["t_plus"]))
        a = f[FEATS].reindex(pd.DatetimeIndex(g["date"]))
        a.index = g.index
        a = a.assign(c_t=at["c"].to_numpy(), j_t=at["j"].to_numpy(), ev_t=at["ev"].to_numpy())
        parts.append(pd.concat([g[["ticker", "date"]], a], axis=1))
    return pd.concat(parts).loc[P.index].reset_index(drop=True)


def load_dailies() -> dict:
    return {t: IO.load_daily(t)[["rv", "j", "seg"]] for t in IO.T17}


def input_identity() -> dict:
    """{бумага: C + J = RV} по сохранённым дневным рядам (независимо от `split`: проверяется сохранённая колонка c)."""
    out = {}
    for t in IO.T17:
        d = IO.load_daily(t)
        out[t] = identity_ok(d["rv"], d["c"], d["j"])
    return out
