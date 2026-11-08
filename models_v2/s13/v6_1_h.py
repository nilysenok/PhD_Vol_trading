"""V6-1a, G10: прогрев H. H = K1 = HAR-J(б) PIT SBER (`har.forecast_at`, ключ `harj_b`, расширяющееся окно по парам t⁺ ≤ d, MIN_PAIRS = 252). Сохранённый K1 (`har_SBER.parquet`) начинается с d = 2016-12-30 и НЕ меняется;
для прогрева 2015-07-07 … 2016-12-29 (377 дат) H считается ТОЙ ЖЕ функцией `har.forecast_at` на дневном ряде `daily_SBER.parquet`. На всех датах K1 до 2017-03-31 пересчёт обязан совпасть с K1 побитно.
Таблица `h_table`: строка = день d с целью в следующей строке панели; H для d ≥ 2016-12-30 берётся из сохранённого K1 (замороженный вход), для прогрева — пересчёт. Только SBER."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s10 import hp_har, hp_paths as P
from models_v2.s11 import v4_io as IO4

WARM_FIRST, K1_FIRST, BITWISE_TO, N_WARM = "2015-07-07", "2016-12-30", "2017-03-31", 377


def daily_sber() -> pd.DataFrame:
    d = pd.read_parquet(P.OUT / "daily_SBER.parquet")
    d.index = pd.to_datetime(d.index)
    return d


def forecast_rows(p: H.Panel, rows) -> np.ndarray:
    """H_t = `har.forecast_at(p, i)['harj_b']` — функция K1 без изменений (мутации подменяют эту функцию)."""
    return np.array([H.forecast_at(p, int(i))["harj_b"] for i in rows])


def h_calc_table(p: H.Panel) -> pd.DataFrame:
    """Все дни d с PIT-прогнозом H (пар ≥ MIN_PAIRS): index d, колонки i, t_plus, rv_next, h_calc."""
    rows = hp_har.grid(p, "2014-01-01", P.MIN_PAIRS)
    return pd.DataFrame({"i": rows, "t_plus": p.dates[rows + 1], "rv_next": p.rv[rows + 1], "h_calc": forecast_rows(p, rows)}, index=pd.DatetimeIndex(p.dates[rows]))


def k1_stored() -> pd.Series:
    f = pd.read_parquet(IO4.path_of("H", "SBER"))
    return pd.Series(f["sigma2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(f["d"])))


def h_table(p: H.Panel, k1: pd.Series) -> pd.DataFrame:
    """index d: i, t_plus, rv_next, h_calc, h (K1 где он сохранён, иначе пересчёт), src ('K1' | 'warm')."""
    t = h_calc_table(p)
    t = t[t.index >= WARM_FIRST].copy()
    stored = k1.reindex(t.index)
    use = (t.index >= K1_FIRST) & stored.notna().to_numpy()
    t["h"] = np.where(use, stored.to_numpy(), t["h_calc"].to_numpy())
    t["src"] = np.where(use, "K1", "warm")
    return t


def check_h(t: pd.DataFrame, k1: pd.Series) -> dict:
    """G10: число дат прогрева, пересчёт = K1 побитно на d ≤ 2017-03-31 (max |diff|), по всей истории K1 (справочно)."""
    warm = t[(t.index >= WARM_FIRST) & (t.index < K1_FIRST)]
    cmp_ = pd.DataFrame({"calc": t["h_calc"], "k1": k1.reindex(t.index)}).dropna()
    early = cmp_[cmp_.index <= BITWISE_TO]
    return dict(n_warm=len(warm), first_warm=str(warm.index[0].date()) if len(warm) else "", last_warm=str(warm.index[-1].date()) if len(warm) else "",
                n_early=len(early), max_diff_early=float(np.abs(early["calc"] - early["k1"]).max()) if len(early) else float("nan"),
                n_all=len(cmp_), max_diff_all=float(np.abs(cmp_["calc"] - cmp_["k1"]).max()))
