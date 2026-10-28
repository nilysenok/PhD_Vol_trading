"""Заглушка-минимум для хука остановки (Ж-031 Э1-а): тестировщик заменяет файл своим набором.
Метрики ряда: Шарп с ddof=1, год-разбивка, сырой эксцесс (нормаль = 3), доля дней в рынке."""
import numpy as np
import pandas as pd
import pytest

from fastbt import sweep_stat as ss


def frame(n=500, seed=0):
    idx = pd.bdate_range("2018-01-01", periods=n)
    x = np.random.default_rng(seed).normal(0.0005, 0.01, n)
    return pd.DataFrame(dict(excess=x, exposure=(x > 0) * 1.0, turnover=np.abs(x), sides=(x > 0) * 1.0), index=idx)


def test_metrics_row_sharpe_ddof1_and_by_year():
    df = frame()
    r = ss.metrics_row(df, (2018, 2019))
    x = df["excess"].values
    assert r["sharpe"] == pytest.approx(x.mean() / x.std(ddof=1) * np.sqrt(252))
    assert set(r["by_year"]) == {2018, 2019} and r["T"] == len(x)
    assert abs(r["kurt"] - 3.0) < 1.0 and 0.0 < r["dni_v_rynke"] < 1.0


def test_metrics_row_year_filter_excludes_other_years():
    df = frame()
    r = ss.metrics_row(df, (2018,))
    assert r["T"] == int((df.index.year == 2018).sum()) and set(r["by_year"]) == {2018}
