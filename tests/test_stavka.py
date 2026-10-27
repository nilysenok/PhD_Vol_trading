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


D0, D1 = pd.Timestamp("2021-12-30"), pd.Timestamp("2022-01-03")      # позиция d -> d_next через новогодние
SEAM_DAILY = 4 * 0.085 / 365.0                                          # дни 30.12, 31.12, 01.01, 02.01


def seam_accrual(r: pd.Series, days: pd.DatetimeIndex) -> float:
    """Ставка за набор календарных дней по ряду r (ffill — как asof), доли."""
    return float(sum(r.asof(d) for d in days) / 365.0)


def check_seam(r: pd.Series, days: pd.DatetimeIndex) -> None:
    """Стык 2021/2022: 8,50 на обеих датах; ряд без дубля и дыры; ровно 4 календарных дня [30.12, 03.01),
    начисление = ряд владельца. У обеих сторон стыка значение одно (8,50), поэтому сдвиг на день ловится
    числом дней (дыра/двойной учёт) и шагом 20.12 (test_each_decision_date...), а не значением на стыке."""
    assert r.index.is_unique and r.index.is_monotonic_increasing, "дубль даты на стыке"
    assert r.asof(D0) == pytest.approx(0.085) and r.asof(D1) == pytest.approx(0.085)
    assert len(days) == 4 and days[0] == D0 and days[-1] == D1 - pd.Timedelta(days=1), "дыра или двойной учёт дня"
    cal = msp.cal_days(pd.DatetimeIndex([D0, D1]), forward=True)
    assert list(cal) == [4.0, 1.0], "дни ставки позиции 30.12 -> 03.01"
    assert seam_accrual(r, days) == pytest.approx(SEAM_DAILY, abs=1e-15)
    assert seam_accrual(r, days) == pytest.approx(seam_accrual(owner(), days), abs=1e-15)   # = ряд владельца
    win = daily(r, pd.date_range("2021-12-20", "2022-01-10"))
    assert win.notna().all() and (win == 0.085).all(), "дыра в ряду на стыке"


def test_seam_2021_2022_values_and_accrual():
    check_seam(msp.rate_cbr(), pd.date_range(D0, D1, inclusive="left"))


def test_seam_mutations_are_caught():
    r, ok = msp.rate_cbr(), pd.date_range(D0, D1, inclusive="left")
    for bad in (pd.date_range(D0, D1),                                                       # оба края: 5 дней, двойной учёт
                pd.date_range(D0, D1, inclusive="neither"),                                  # 3 дня, дыра
                pd.date_range(D0 + pd.Timedelta(days=1), D1 + pd.Timedelta(days=1), inclusive="left")[1:],   # сдвиг +1: 3 дня
                pd.date_range(D0 - pd.Timedelta(days=1), D1 - pd.Timedelta(days=1), inclusive="left")[:3]):  # сдвиг −1: 3 дня
        with pytest.raises(AssertionError):
            check_seam(r, bad)
    k = pd.read_parquet(msp.KEY_RATE)
    p = k.set_index(pd.to_datetime(k["date"]))["value"].sort_index() / 100.0
    dup = pd.concat([owner(), p[p.index >= pd.Timestamp("2021-12-01")]])                      # разреза нет: дубль 20.12
    with pytest.raises(AssertionError):
        check_seam(dup, ok)
    hole = r[(r.index < pd.Timestamp("2021-12-01")) | (r.index > pd.Timestamp("2022-02-01"))]  # ряд без 8,50
    with pytest.raises(AssertionError):
        check_seam(hole, ok)
    const75 = pd.Series(0.075, index=r.index)
    with pytest.raises(AssertionError):
        check_seam(const75, ok)


def test_dev_paths_use_rate_cbr_not_restored():
    for f in ("fastbt/s9.py", "fastbt/fast_grid.py", "scripts/analysis/s9_shag0.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "rate_restored()" not in src.replace("rate_restored», ", ""), f
