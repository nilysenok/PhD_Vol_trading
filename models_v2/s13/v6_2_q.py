"""V6-2, G24: признаки качества измерения RV по M10 (план V6, п. 4) и индекс неустойчивости u. Только бары самого дня d (q8 — ещё число баров сессий ≤ d своего года); тиков и спредов нет, о шуме микроструктуры ничего не утверждается.
x_k — M10-доходности основной сессии (r_1 = ln(c/o), далее close-к-close), RV10 = Σx². Сетка m·10 минут — суммы m подряд идущих доходностей, сдвиг s = 0…m−1 (начало с (s+1)-го бара), неполный хвост отбрасывается; RV на редкой сетке НЕ масштабируется.
q1, q2 = ln(RV20/RV10), ln(RV30/RV10) (RV20, RV30 — средние по 2 и 3 сдвигам); q3 = СКО (ddof = 0) RV30 по трём сдвигам / среднее; q4 = доля x = 0; q5 = Σx⁴/(Σx²)²; q6 = Σ_{k≥2} x_k x_{k−1} / Σx²;
q7 = Σ (vol_k/Vol)² (Херфиндаль объёма); q8 = n_d / мода n за сессии года ≤ d (при равенстве частот — большее n).
u_d = среднее z(q3), z(|q6|), z(q1), z(q2); z — среднее и СКО (`v6_1_fit.norm_stats`, как у признаков движка) по ЗАКРЫТЫМ парам на дату переоценки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s9 import hv2_m10 as M
from models_v2.s13 import v6_1_fit as FT

Q_NAMES = ["q%d" % i for i in range(1, 9)]
U_COMP = ("q3", "q6", "q1", "q2")          # q6 берётся по модулю
GRID = (2, 3)


def grid_rv(x: np.ndarray, m: int) -> np.ndarray:
    """RV на сетке m·10 минут для сдвигов 0…m−1 (m значений; NaN, если нет ни одного полного блока)."""
    out = np.full(m, np.nan)
    for s in range(m):
        k = (len(x) - s) // m
        if k > 0:
            out[s] = float((x[s:s + k * m].reshape(k, m).sum(axis=1) ** 2).sum())
    return out


def day_q(x, vol) -> np.ndarray:
    """q1…q7 одного дня по M10-доходностям x и объёмам vol (q8 — позже, нужна история чисел баров)."""
    x, vol = np.asarray(x, float), np.asarray(vol, float)
    q2 = x * x
    rv, tot = float(q2.sum()), float(vol.sum())
    g20, g30 = grid_rv(x, 2), grid_rv(x, 3)
    return np.array([np.log(g20.mean() / rv), np.log(g30.mean() / rv), float(g30.std() / g30.mean()), float((x == 0).mean()), float((q2 * q2).sum() / rv ** 2),
                     float((x[1:] * x[:-1]).sum() / rv), float(((vol / tot) ** 2).sum()) if tot > 0 else np.nan])


def mode_share(n: pd.Series) -> pd.Series:
    """q8 = n_d / мода n среди сессий того же года с датой ≤ d (PIT: накопление по порядку дат). Мутант «мода по всему году» подменяет эту функцию в тесте."""
    out, cnt, year = {}, {}, None
    for d, v in n.items():
        if d.year != year:
            cnt, year = {}, d.year
        cnt[int(v)] = cnt.get(int(v), 0) + 1
        mode = max(cnt, key=lambda k: (cnt[k], k))
        out[d] = v / mode
    return pd.Series(out)


def day_table(bars: pd.DataFrame) -> pd.DataFrame:
    """Бары -> по сессиям: q1…q7 и n. Только бары самого дня. МУТАНТ «бары дня t⁺ в Q дня d» сдвигает результат в тесте."""
    main = D.main_session(bars).sort_values("begin").reset_index(drop=True)
    r = M.bar_returns(bars)
    x, vol = r["r"].to_numpy(float), main["volume"].to_numpy(float)
    rows = {d: np.r_[day_q(x[ix], vol[ix]), len(ix)] for d, ix in r.groupby("date").indices.items()}
    return pd.DataFrame.from_dict(rows, orient="index", columns=Q_NAMES[:7] + ["n"]).rename_axis("date").sort_index()


def features(bars: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """-> q1…q8 на днях панели `daily.index`."""
    t = day_table(bars).reindex(daily.index)
    t["q8"] = mode_share(t["n"].dropna()).reindex(t.index)
    return t[Q_NAMES]


def components(Q: np.ndarray) -> np.ndarray:
    """(N × 4): q3, |q6|, q1, q2 — компоненты u до нормировки."""
    ix = [Q_NAMES.index(c) for c in U_COMP]
    c = np.asarray(Q, float)[:, ix].copy()
    c[:, 1] = np.abs(c[:, 1])
    return c


def u_norm(C: np.ndarray, tr: np.ndarray) -> tuple:
    """Среднее и СКО компонент по закрытым парам tr. МУТАНТ «по всей выборке» подменяет эту функцию в тесте."""
    return FT.norm_stats(C[tr])


def u_values(Q: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """u для всех строк с нормировкой по закрытым парам tr."""
    C = components(Q)
    mu, sd = u_norm(C, tr)
    return ((C - mu) / sd).mean(axis=1)


def u_threshold(u: np.ndarray, tr: np.ndarray, level: float = 0.8) -> float:
    """Порог верхнего квинтиля u — по закрытым парам tr."""
    return float(np.quantile(u[tr], level))
