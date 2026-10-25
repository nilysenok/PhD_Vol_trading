#!/usr/bin/env python3
"""etap0: QLIKE и DM на сохранённых прогнозах СВ (только чтение).

ОПРЕДЕЛЕНИЕ ПЕЧАТИ (найдено проверяющим, PR #1):
  QLIKE = простое среднее 9 годовых QLIKE 2017-2025 (каждый год - пул по тикерам),
  формула RV/F - ln(RV/F) - 1 в дисперсиях, rv_actual > 0, остановка
  28.02-24.03.2022 НЕ исключается.
  DM = простой t без HAC на пуле строк (дата, тикер) 2017-2025, V1 против HAR-J.
Дополнительно печатается "пул 2017-2026 по строкам" (остановка исключена) и
HAC по времени на дневном ряду средних по пулу потерь (лаг h-1).
Пустой вход (файл, год, тикер) - явная ошибка.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2] / "moex_volatility"
WF = ROOT / "results/final/data/predictions_walkforward"
HALT = ("2022-02-28", "2022-03-24")
YEARS = range(2017, 2026)


def load(name, h):
    d = pd.read_parquet(WF / f"{name}_h{h}_annual.parquet")
    d = d[d.rv_actual > 0]
    if len(d) == 0:
        raise ValueError(f"пустой вход: {name} h={h}")
    return d.set_index(["date", "ticker"]).sort_index()


def drop_halt(d):
    return d[~d.index.get_level_values("date").to_series().between(*HALT).values]


def loss(d):
    if len(d) == 0:
        raise ValueError("пустой вход в loss()")
    p = d.rv_pred.clip(lower=1e-10)
    return d.rv_actual / p - np.log(d.rv_actual / p) - 1


def by_year(l, years=YEARS):
    """Простое среднее годовых QLIKE (год - пул по тикерам); пустой год - ошибка."""
    yr = l.index.get_level_values("date").year
    out = []
    for y in years:
        s = l[yr == y]
        if len(s) == 0:
            raise ValueError(f"пустой год {y}")
        out.append(s.mean())
    return float(np.mean(out))


def check_tickers(d, tickers=None):
    tk = d.index.get_level_values("ticker")
    for t in (tickers or []):
        if (tk == t).sum() == 0:
            raise ValueError(f"тикер без строк: {t}")
    if tk.nunique() == 0:
        raise ValueError("нет тикеров")


def in_years(l, years=YEARS):
    yr = l.index.get_level_values("date").year
    return l[(yr >= min(years)) & (yr <= max(years))]


def dm_simple(l1, l2):
    d = (l1 - l2).dropna().values
    if len(d) < 2:
        raise ValueError("DM: мало наблюдений")
    return d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))  # t>0: вторая модель лучше


def dm_hac_time(l1, l2, h):
    """HAC по времени: дневной ряд средних по пулу потерь, лаг h-1 (Бартлетт)."""
    d = (l1 - l2).dropna().groupby(level="date").mean().sort_index().values
    n, lag = len(d), h - 1
    if n < 2:
        raise ValueError("DM-HAC: мало дней")
    g = np.var(d, ddof=1)
    for k in range(1, lag + 1):
        g += 2 * (1 - k / (lag + 1)) * np.cov(d[k:], d[:-k])[0, 1]
    return d.mean() / np.sqrt(g / n)


def main():
    names = ["har", "hybrid_adaptive", "hybrid_best", "hybrid_v2"]
    for h in (1, 5, 22):
        raw = {n: load(n, h) for n in names}
        for n in names:
            check_tickers(raw[n])
        l_all = {n: loss(raw[n]) for n in names}                 # без исключения остановки
        l_hlt = {n: loss(drop_halt(raw[n])) for n in names}      # с исключением
        print(f"h={h} QLIKE определение печати (среднее годовых 2017-2025, остановка не исключена):",
              {n: round(by_year(l_all[n]), 4) for n in names})
        print(f"h={h} QLIKE пул 2017-2026 по строкам (остановка исключена):",
              {n: f"{l_hlt[n].mean():.4f} (n={len(l_hlt[n])})" for n in names})
        p = {n: in_years(l_all[n]) for n in names}
        print(f"h={h} n строк 2017-2025: " + ", ".join(f"{n}={len(p[n])}" for n in names))
        for n in names[1:]:
            t0 = dm_simple(p["har"], p[n])
            th = dm_hac_time(p["har"], p[n], h) if h > 1 else t0
            print(f"  DM {n} vs har, определение печати (простой t, пул 2017-2025): t={t0:+.2f}; "
                  f"HAC по времени (дневной ряд средних, лаг {h-1}): t={th:+.2f}")
    t19 = pd.read_parquet(ROOT / "data/predictions/test_2019/hybrid_h1.parquet")
    print("test 2019 hybrid_h1 rows:", len(t19))
    print("hybrid_best h1 rows (все даты):", len(pd.read_parquet(WF / "hybrid_best_h1_annual.parquet")))


if __name__ == "__main__":
    main()
