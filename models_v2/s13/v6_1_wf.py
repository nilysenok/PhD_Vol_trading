"""V6-1a, G13: помесячная переоценка (PIT). В первый торговый день каждого месяца (строка таблицы H с минимальной датой месяца) модель оценивается на ЗАКРЫТЫХ парах — строки, у которых t⁺ ≤ d (расширяющееся окно с 2015-07-07);
нормировка признаков — по тем же парам; прогнозы g дней месяца — по параметрам этой переоценки. Первая переоценка — первый торговый день января 2017: до неё (d = 2016-12-30) модели нет, g = 0, F = H побитно.
Таблица `tab` — `v6_1_h.h_table` (index d, t_plus, rv_next, h), `S` — признаки на тех же днях. Обучатель — функция `fit(X, e) -> модель` с методом `g(X)` (LIN, GBM; следующие листы добавляют наборы признаков и обучатели, не трогая движок)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s13 import v6_1_fit as FT

FIRST_REFIT = "2017-01-01"


def closed_rows(t_plus: np.ndarray, d0) -> np.ndarray:
    """Закрытые пары на дату переоценки d0: t⁺ ≤ d0. МУТАНТ «закрыто + 1» (t⁺ ≤ d0 + 1 торговый день) подменяет эту функцию в тесте."""
    return np.flatnonzero(np.asarray(t_plus, "datetime64[ns]") <= np.datetime64(d0))


def refit_positions(index: pd.DatetimeIndex, first: str = FIRST_REFIT) -> list:
    """Позиции первых торговых дней месяцев (в индексе таблицы) с `first`."""
    per = index.to_period("M")
    pos = [int(np.flatnonzero(per == m)[0]) for m in per.unique() if m.start_time >= pd.Timestamp(first)]
    return pos


def residual(tab: pd.DataFrame) -> np.ndarray:
    return np.log(tab["rv_next"].to_numpy(float) / tab["h"].to_numpy(float))


def walk_forward(tab: pd.DataFrame, S: pd.DataFrame, fit, first: str = FIRST_REFIT) -> dict:
    """-> {g: Series по дням таблицы (0 до первой переоценки), fits: список переоценок (d0, n_train, объект модели)}."""
    X, e, tp = S.to_numpy(float), residual(tab), tab["t_plus"].to_numpy()
    g = np.zeros(len(tab))
    pos = refit_positions(tab.index, first)
    fits = []
    for a, b in zip(pos, pos[1:] + [len(tab)]):
        d0 = tab.index[a]
        tr = closed_rows(tp, d0)
        m = fit(X[tr], e[tr])
        g[a:b] = m.g(X[a:b])
        fits.append(dict(d0=d0, n_train=len(tr), model=m))
    return dict(g=pd.Series(g, index=tab.index, name="g"), fits=fits)


def forecasts(tab: pd.DataFrame, g: pd.Series) -> pd.Series:
    """F = H·exp(clip(g, −1, 1)) на днях таблицы."""
    return pd.Series(FT.forecast(tab["h"].to_numpy(float), g.to_numpy(float)), index=tab.index, name="F")
