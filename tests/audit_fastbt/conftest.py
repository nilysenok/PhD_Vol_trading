"""Аудит fastbt с подложенными ошибками (владелец 26.10, 22:20).

Каждый модуль test_NN_*.py проверяет правильное поведение на синтетике с ответом, посчитанным
вручную (и, где уместно, на данных разработки ≤ 2021), и объявляет MUTATIONS — словарь
{имя: функция(monkeypatch)}: намеренная ошибка вносится только через monkeypatch, исходники не
правятся. Мутация включается переменной окружения AUDIT_MUTATION=<имя> (так её запускает
run_mutations.py); на мутации тесты модуля обязаны падать.

Данные ≥ 2022 не читаются: файлы открываются через read_dev() с фильтром pyarrow date ≤ 2021-12-31,
а fastbt.data.daily_df / vpred_df подменяются отфильтрованными (фикстура dev_data).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "analysis", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastbt.data import MAIN, OHLCV  # noqa: E402

DEV_END = pd.Timestamp("2021-12-31")
SIGMA_H1 = MAIN / "data" / "regen" / "sigma_pred_h1_final3_full.parquet"
KEY_RATE = MAIN / "moex_discovery" / "data" / "external" / "macro" / "raw" / "key_rate_cbr.parquet"
MUT_ENV = "AUDIT_MUTATION"


def read_dev(path, columns=None) -> pd.DataFrame:
    """Только строки с date ≤ 31.12.2021 (фильтр на чтении, строки ≥ 2022 не материализуются)."""
    t = pq.read_table(path, columns=columns, filters=[("date", "<=", DEV_END.to_pydatetime())])
    d = t.to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    assert d["date"].max() <= DEV_END
    return d


def bdays(start, n):
    return pd.bdate_range(start, periods=n).values.astype("datetime64[ns]")


@pytest.fixture(autouse=True)
def _mutation(request, monkeypatch):
    name = os.environ.get(MUT_ENV)
    muts = getattr(request.module, "MUTATIONS", {})
    if name and name in muts:
        muts[name](monkeypatch)
    yield


@pytest.fixture
def dev_data(monkeypatch):
    """fastbt.data читает только ≤ 2021 (бары и σ_pred h1 финала)."""
    from fastbt import data as fd
    ohlcv = read_dev(OHLCV)
    sig = read_dev(SIGMA_H1)
    monkeypatch.setattr(fd, "daily_df", lambda: ohlcv)
    monkeypatch.setattr(fd, "vpred_df", lambda: sig)
    fd.load_ticker.cache_clear()
    yield fd
    fd.load_ticker.cache_clear()
