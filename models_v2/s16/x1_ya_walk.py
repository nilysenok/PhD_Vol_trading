"""Y-A, помесячная переоценка (план раздел 8): в первый торговый день D каждого месяца с 2018-07-02 по ВСЕМ закрытым к D парам (t⁺ ≤ D; расширяющееся окно с 2018-01-04); параметры действуют на дни D … конец месяца.
До 2018-07-02 F = X1 побитно, `active` = 0. Бегун сам проверяет ≥ 120 закрытых пар на 2018-07-02 (ожидается 124) и отказывает иначе. Пара j: (d_j, t⁺_j, X1_j, RV(t⁺_j), признаки дня d_j)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_ya_feat as FE, x1_ya_fit as FT

ACTIVATION, MIN_PAIRS = pd.Timestamp("2018-07-02"), 120
OUT_COLS = ["d", "t_plus", "rv", "X1", "F", "g", "active", "lam", "a_n", "a_e", "kappa"]


def closed_date(tp, d):
    """День, с которого пара доступна для обучения: t⁺ (мутация «закрыто + 1»: дата прогноза пары d — пара видна за торговый день до реализации RV)."""
    return tp


def closed_pool(tp, d, D) -> np.ndarray:
    """Индексы (в порядке дат) пар, закрытых к дню D: closed_date ≤ D."""
    return np.flatnonzero(np.asarray(closed_date(np.asarray(tp, "datetime64[ns]"), np.asarray(d, "datetime64[ns]"))) <= np.datetime64(D))


def month_starts(d) -> list:
    """Позиции первого торгового дня каждого месяца (по idx прогнозов)."""
    d = pd.DatetimeIndex(d)
    return [int(i) for i in np.flatnonzero(~pd.Series(list(zip(d.year, d.month))).duplicated().to_numpy())]


def check_activation(d, tp) -> int:
    """Число закрытых пар на 2018-07-02; отказ при < 120 (ожидается 124) и если дня активации нет в рядах."""
    d = pd.DatetimeIndex(d)
    if ACTIVATION not in set(d):
        raise SystemExit("нет первого торгового дня месяца 2018-07-02")
    n = len(closed_pool(tp, d, ACTIVATION))
    if n < MIN_PAIRS:
        raise SystemExit("закрытых пар на %s: %d < %d — включение отказано" % (ACTIVATION.date(), n, MIN_PAIRS))
    return n


def walk(x1: pd.DataFrame, ft: dict, mode: str) -> pd.DataFrame:
    """x1: d, t_plus, rv, F (=X1), s; ft — признаки `x1_ya_feat.features`. -> прогнозы Y (OUT_COLS)."""
    d, tp = pd.DatetimeIndex(x1["d"]), pd.DatetimeIndex(x1["t_plus"])
    rv, X1, e = x1["rv"].to_numpy(float), x1["F"].to_numpy(float), FE.e_state(x1).astype(float)
    s, delta, u = np.asarray(ft["s"], float), np.asarray(ft["delta"], float), rv / X1
    check_activation(d, tp)
    n = len(d)
    g, active, lam_a, an, ae, ka = (np.zeros(n), np.zeros(n), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    starts = [p for p in month_starts(d) if d[p] >= ACTIVATION] + [n]
    for a, b in zip(starts[:-1], starts[1:]):
        pool = closed_pool(tp, d, d[a])
        lam, th = FT.refit(pool, u, e, s, delta, mode)
        gi, _ = FT._g(th, e[a:b], s[a:b], delta[a:b], mode)
        g[a:b], active[a:b], lam_a[a:b], an[a:b], ae[a:b] = gi, 1.0, lam, th[0], th[1]
        ka[a:b] = th[2] if mode == "Y1" else 0.0
    return pd.DataFrame(dict(d=d, t_plus=tp, rv=rv, X1=X1, F=X1 * np.exp(g), g=g, active=active.astype(int), lam=lam_a, a_n=an, a_e=ae, kappa=ka))[OUT_COLS]
