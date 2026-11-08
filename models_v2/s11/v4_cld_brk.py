"""V4-CLD: признак «время после структурного излома» (Ж-225 п. 5). Двусторонний CUSUM на медиане по 17 бумагам ln RV_d (`med_lrv`), стандартизованной по ПРОШЛОМУ окну (W = 252 дней, строго до дня d, не меньше 60 наблюдений):
S⁺ = max(0, S⁺ + z − k), S⁻ = max(0, S⁻ − z − k), z = (x_d − μ_{d−W..d−1})/σ_{d−W..d−1}; порог h = 5, дрейф k = 0,5 (в с. о.); срабатывание при S > h, затем сброс обеих сумм. Даты изломов вручную не задаются.
Признак brk_d = ln(1 + число торговых дней ряда с последнего срабатывания ≤ d); до первого срабатывания — с начала ряда. Всё по данным ≤ d. `lead` — только для теста-мутации («срабатывание с данными d+1»); боевое 0.
Новый признак хранится отдельно (`BRK.parquet`), Z не перезаписывается."""
from __future__ import annotations

import numpy as np
import pandas as pd

W, MINW, K, H = 252, 60, 0.5, 5.0


def past_stats(x: pd.Series) -> tuple:
    """Среднее и с. о. по окну W строго до дня d (shift(1)): день d в собственную стандартизацию не входит."""
    r = x.shift(1).rolling(W, min_periods=MINW)
    return r.mean().to_numpy(), r.std().to_numpy()


def cusum(x: pd.Series, lead: int = 0) -> tuple:
    """x — ряд по датам (индекс — отсортированные даты). -> (brk, fired): brk — ln(1 + дни с последнего срабатывания), fired — булев ряд срабатываний (день срабатывания)."""
    v = x.shift(-lead) if lead else x
    mu, sd = past_stats(v)
    a = v.to_numpy(float)
    n = len(a)
    brk, fired = np.zeros(n), np.zeros(n, bool)
    sp = sm = 0.0
    last = 0
    for i in range(n):
        if np.isfinite(a[i]) and np.isfinite(mu[i]) and sd[i] > 1e-12:
            z = (a[i] - mu[i]) / sd[i]
            sp, sm = max(0.0, sp + z - K), max(0.0, sm - z - K)
            if sp > H or sm > H:
                fired[i], last, sp, sm = True, i, 0.0, 0.0
        brk[i] = np.log1p(i - last)
    return pd.Series(brk, index=x.index), pd.Series(fired, index=x.index)


def median_series(Zl: pd.DataFrame) -> pd.Series:
    """Медиана ln RV по бумагам набора (колонка `med_lrv` длинной Z — уже поперечная медиана по набору: 17 или 16 в LOTO) по датам, отсортированная."""
    return Zl.groupby("date")["med_lrv"].first().sort_index()


def brk_by_date(Zl: pd.DataFrame, lead: int = 0) -> pd.Series:
    return cusum(median_series(Zl), lead)[0]


def build_table(Zl: pd.DataFrame) -> pd.DataFrame:
    """Таблица для `BRK.parquet`: date, med_lrv, brk, fired."""
    x = median_series(Zl)
    b, f = cusum(x)
    return pd.DataFrame({"date": x.index, "med_lrv": x.to_numpy(), "brk": b.to_numpy(), "fired": f.to_numpy()})
