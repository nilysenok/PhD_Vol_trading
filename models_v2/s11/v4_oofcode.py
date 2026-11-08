"""V4-0, G2: проверка по коду, что обучающие пары прогноза дня d закрыты (цель t⁺ <= d). Вызываются общие функции отбора пар, которыми пользовались K1, K2/K3 и LGBM-D2; модели не обучаются.
Синтетика с известным ответом: цель строки i возрастает, поэтому любая пара с t⁺ > d дала бы y > ln RV_d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import har
from models_v2.s4 import fit
from models_v2.s10 import hp_engine as E


def check_k1(n: int = 60) -> bool:
    """K1: `har.pairs(p, i)` берёт пары (t, t+1) только с t+1 <= i."""
    rv = np.exp(np.arange(n, dtype=float) / 10)
    X = np.column_stack([np.log(rv)] * 4)
    p = har.Panel(np.arange(n).astype("datetime64[D]").astype("datetime64[ns]"), np.zeros(n, int), rv, X)
    for i in range(10, n):
        _, y = har.pairs(p, i, har.COLS_HARJ)
        if len(y) == 0 or y.max() > p.lnrv[i] + 1e-12 or len(y) != i:
            return False
    return True


def check_k23(n: int = 60) -> bool:
    """K2/K3: `fit.pair_index(..., T=d)` — строки t с t+1 <= d (боевой путь `jobs.pairs`/`job.pairs`)."""
    seg, X, y = np.zeros(n, int), np.ones((n, 4)), np.arange(n, dtype=float)
    for d in range(5, n):
        idx = fit.pair_index(seg, X, y, (0, 1, 2, 3), d)
        if len(idx) != d or idx.max() + 1 > d:
            return False
    return True


def check_d2(n: int = 400) -> bool:
    """D2 (движок гибридов): `train_sets` — пары s⁺ <= d; ранняя остановка — последние es_dates дат, вне обучения, не раньше обучения."""
    tp = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    tk, y, base = np.zeros(n, int), np.ones(n), np.ones(n)
    hp = dict(E.HP, es_dates=30)
    for d in (tp[150], tp[300]):
        tr, es = E.train_sets(tp, tk, y, base, d, (), hp)
        if len(tr) == 0 or tp[tr].max() > d or tp[es].max() > d or np.intersect1d(tr, es).size or tp[tr].max() >= tp[es].min():
            return False
    return True


def check_closed_choice() -> bool:
    """D2: выбор (глубина, c, λ) на год Y — по закрытым прогнозам года Y−1 (t⁺ < 01.01.Y): `closed_mask`."""
    d = pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02"])
    t = pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"])
    m = E.closed_mask(d, t, 2020)
    return bool(m.tolist() == [True, False, False])


def run() -> dict:
    return {"K1_pairs_closed": check_k1(), "K2K3_pairs_closed": check_k23(), "D2_train_closed_es_outside": check_d2(), "D2_choice_closed_year": check_closed_choice()}
