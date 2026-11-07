"""P0, G5 (R4): межакционная синхронизация. Поперечные (панельные) величины дня t считаются ТОЛЬКО по бумагам, у которых есть собственный день t
(день t закончился к отсечке 18:50: каноническая строка daily.build_daily); бумага без дня t в поперечные величины дня t не входит — не ffill, не день t+1.
Строка t широкой матрицы — единственный вход поперечной величины строки t: ничего из строк t+1… (тест: подмена любого другого дня t+1 не меняет величины на t)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CROSS = 5     # меньше бумаг в дне — поперечные величины NaN (не оценка по горстке)


def wide(series: dict) -> pd.DataFrame:
    """{бумага: Series(date)} -> матрица дата × бумага. Нет дня у бумаги — NaN (без заполнения)."""
    return pd.DataFrame(series).sort_index()


def cross_stats(lnrel: pd.DataFrame, up: pd.DataFrame) -> pd.DataFrame:
    """По строке t: число бумаг, медиана и разброс (std, ddof=1) ln RV_d/RV_m, доля бумаг с RV_d > RV_w (up — 1/0/NaN). Бумаг < MIN_CROSS — NaN."""
    n = lnrel.notna().sum(axis=1)
    out = pd.DataFrame({"n_cross": n, "m_med": lnrel.median(axis=1), "m_disp": lnrel.std(axis=1, ddof=1),
                        "m_share_up": up.where(lnrel.notna()).mean(axis=1)})
    out.loc[n < MIN_CROSS, ["m_med", "m_disp", "m_share_up"]] = np.nan
    return out


def cross_rank(lnrel: pd.DataFrame, med: pd.Series) -> tuple:
    """(ранг бумаги в дне t — доля бумаг дня с меньшим значением, отклонение от панельной медианы дня); бумаги без дня t — NaN."""
    n = lnrel.notna().sum(axis=1)
    rank = lnrel.rank(axis=1, pct=True)
    dev = lnrel.sub(med, axis=0)
    rank[n < MIN_CROSS] = np.nan
    dev[n < MIN_CROSS] = np.nan
    return rank, dev
