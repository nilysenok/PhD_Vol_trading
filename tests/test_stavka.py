"""Ряд ставки ЦБ в разработке 2018–2021 (Ъ владельца 27.10, 12:00): даты вступления в силу и значения.

Ряд владельца — fastbt/rates/key_rate_cbr_vladelec.csv; `msp.rate_cbr()` читает его на ≤ 2021 и
parquet ЦБ на ≥ 2022. Тесты ловят: константу на отрезке, сдвиг даты на день, возврат к parquet
(у него 09.02.2018 вместо 12.02 и 29.10.2019 вместо 28.10). Мутации — только в памяти."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastbt import msp  # noqa: E402

START, END = pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")
CAL = pd.date_range(START, END)


def owner() -> pd.Series:
    return msp.rate_vladelec()


def daily(s: pd.Series, days=CAL) -> pd.Series:
    return s.reindex(s.index.union(days)).sort_index().ffill().reindex(days)


def longest_run_days(x: pd.Series) -> int:
    g = (x != x.shift()).cumsum()
    return int(x.groupby(g).size().max())


def check_series(x: pd.Series) -> None:
    """Ряд на календарных днях 2018–2021 равен ряду владельца и не имеет отрезка-константы
    длиннее наибольшего интервала между решениями."""
    o = owner()
    inner = o[(o.index >= START) & (o.index <= END)].index
    gap = int(o.index.to_series().diff().dt.days.loc[inner].max())
    ref = daily(o)
    assert (x.reindex(CAL) - ref).abs().max() < 1e-12, "ряд расходится с рядом владельца"
    assert longest_run_days(x.reindex(CAL)) <= gap, "отрезок-константа длиннее интервала между решениями"


def test_spot_values():
    r = msp.rate_cbr()
    assert r.asof(pd.Timestamp("2020-07-01")) == pytest.approx(0.045)
    assert r.asof(pd.Timestamp("2021-11-01")) == pytest.approx(0.075)
    assert r.asof(pd.Timestamp("2018-03-01")) == pytest.approx(0.075)


def test_owner_file_shape():
    o = owner()
    assert len(o) == 40 and o.index.is_monotonic_increasing and not o.index.has_duplicates
    assert o.index[0] == pd.Timestamp("2017-12-18") and o.iloc[0] == pytest.approx(0.0775)
    assert o.index[-1] == pd.Timestamp("2025-10-27") and o.iloc[-1] == pytest.approx(0.165)


def test_each_decision_date_new_value_day_before_old():
    r, o = msp.rate_cbr(), owner()
    for i in range(1, len(o)):
        d = o.index[i]
        if d >= pd.Timestamp("2022-01-01"):
            break
        assert r.asof(d) == pytest.approx(o.iloc[i]), d
        assert r.asof(d - pd.Timedelta(days=1)) == pytest.approx(o.iloc[i - 1]), d


def test_rate_cbr_equals_owner_on_dev_years():
    check_series(daily(msp.rate_cbr()))


def test_known_parquet_disagreements_fixed():
    r = msp.rate_cbr()
    assert r.asof(pd.Timestamp("2018-02-09")) == pytest.approx(0.0775)    # parquet: 7,50
    assert r.asof(pd.Timestamp("2018-02-12")) == pytest.approx(0.075)
    assert r.asof(pd.Timestamp("2019-10-28")) == pytest.approx(0.065)     # parquet: 7,00


def test_mutation_constant_is_caught():
    const = pd.Series(0.075, index=CAL)
    with pytest.raises(AssertionError):
        check_series(const)


def test_mutation_date_shift_by_one_day_is_caught():
    x = daily(msp.rate_cbr())
    for k in (1, -1):
        with pytest.raises(AssertionError):
            check_series(x.shift(k))


def test_parquet_series_is_caught():
    """Прежний путь (parquet ЦБ) на 2018–2021 расходится с владельцем в двух днях — тест его ловит."""
    k = pd.read_parquet(msp.KEY_RATE)
    p = k.set_index(pd.to_datetime(k["date"]))["value"] / 100.0
    with pytest.raises(AssertionError):
        check_series(daily(p))


def test_restored_rate_backfilled_constant_is_caught():
    """rate_restored начинается 03.01.2022; msp.portfolio тянет её назад bfill → константа 8,5 %."""
    r = msp.rate_restored()
    x = r.reindex(r.index.union(CAL)).sort_index().ffill().bfill().reindex(CAL)    # как в msp.portfolio
    assert longest_run_days(x.reindex(CAL)) == len(CAL)
    with pytest.raises(AssertionError):
        check_series(x)


def test_dev_paths_use_rate_cbr_not_restored():
    for f in ("fastbt/s9.py", "fastbt/fast_grid.py", "scripts/analysis/s9_shag0.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "rate_restored()" not in src.replace("rate_restored», ", ""), f
