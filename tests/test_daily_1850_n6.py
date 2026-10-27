"""Н6-правки (Ж-029, замечания к #297): утренние бары, граница 18:50, --data-root только новый ряд,
атомарный ensure(). Продолжение tests/test_daily_1850.py (там — вечерние бары и сверка со старым рядом)."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850 as D

MAIN = D.default_main()
CANDLES = MAIN / "moex_discovery" / "data" / "raw" / "candles_10m"
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def tiny() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["X"], "date": [pd.Timestamp("2021-12-07")], "open": [1.0], "high": [1.0],
                         "low": [1.0], "close": [1.0], "volume": [1]})


# ---- утренние бары (с 06.12.2021 есть 06:50–09:40): в день не входят ----
def synth_morning(morning: bool) -> pd.DataFrame:
    rows = [("2021-12-07 09:50", 100, 101, 99, 100.5, 10), ("2021-12-07 18:40", 100.5, 102, 100, 101.5, 30)]
    if morning:                       # утренние бары шире диапазона и с другой ценой
        rows += [("2021-12-07 06:50", 80, 300, 1, 90, 500), ("2021-12-07 09:40", 90, 400, 2, 95, 700)]
    return pd.DataFrame(rows, columns=COLS).assign(timestamp=lambda x: pd.to_datetime(x["timestamp"]))


def check_morning_invariant(open_min=D.OPEN_MIN):
    a = D.aggregate(synth_morning(False), "X", open_min=open_min)
    b = D.aggregate(synth_morning(True), "X", open_min=open_min)
    pd.testing.assert_frame_equal(a, b)
    r = b.iloc[0]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100, 102, 99, 101.5, 40)


def test_morning_bars_do_not_change_day():
    check_morning_invariant()


@pytest.mark.parametrize("open_min", [9 * 60 + 40, 6 * 60 + 50])      # «окно с 09:40», «окно с 06:50»
def test_mutation_open_is_caught(open_min):
    with pytest.raises(AssertionError):
        check_morning_invariant(open_min=open_min)


def test_mutation_le_1850_is_caught():
    """Бар ровно 18:50 в день не входит; мутация «≤ 18:50» (cut на минуту дальше) роняет тест."""
    def check(cut_min):
        b = synth_morning(False)
        b.loc[len(b)] = [pd.Timestamp("2021-12-07 18:50"), 1, 999, 0.1, 7, 5000]
        r = D.aggregate(b, "X", cut_min).iloc[0]
        assert (r["high"], r["low"], r["close"], r["volume"]) == (102, 99, 101.5, 40)
    check(D.CUT_MIN)
    with pytest.raises(AssertionError):
        check(D.CUT_MIN + 1)


@pytest.mark.parametrize("ticker", ["SBER", "LKOH", "LSRG"])
def test_december_2021_open_is_bar_0950(ticker):
    if not (CANDLES / f"{ticker}.parquet").exists():
        pytest.skip("нет общих данных")
    bars = D.read_bars(CANDLES / f"{ticker}.parquet")
    mod = bars["timestamp"].dt.hour * 60 + bars["timestamp"].dt.minute
    late = bars["timestamp"] >= "2021-12-06"
    assert ((mod < D.OPEN_MIN) & late).any(), "в данных нет утренних баров декабря 2021 — тест пуст"
    b950 = bars[(mod == D.OPEN_MIN) & late]
    days = b950["timestamp"].dt.normalize()
    assert len(days) >= 15
    new = D.aggregate(bars, ticker).set_index("date")
    np.testing.assert_array_equal(new.loc[days, "open"].to_numpy(), b950["open"].to_numpy())
    mut = D.aggregate(bars, ticker, open_min=9 * 60 + 40).set_index("date")        # мутация «окно с 09:40»
    assert not np.array_equal(mut.loc[days, "open"].to_numpy(), b950["open"].to_numpy())


# ---- --data-root: только новый ряд ----
def new_dir(tmp_path):
    tiny().to_parquet(tmp_path / "ohlcv_daily_1850.parquet", index=False)
    return tmp_path


def test_data_root_new_series_accepted(tmp_path):
    assert D.check_data_root(new_dir(tmp_path)).name == "ohlcv_daily_1850.parquet"


def test_data_root_old_series_refused(tmp_path):
    d = new_dir(tmp_path)
    old = tiny().assign(date=pd.Timestamp("2023-01-03"))
    old.to_parquet(d / "ohlcv_daily_full.parquet", index=False)
    with pytest.raises(SystemExit, match="запрещён"):                   # каталог с ohlcv_daily_full
        D.check_data_root(d)
    with pytest.raises(SystemExit, match="запрещён"):                   # прямой путь на старый файл
        D.check_data_root(d / "ohlcv_daily_full.parquet")
    (d / "ohlcv_daily_full.parquet").unlink()
    old.to_parquet(d / "ohlcv_daily_1850.parquet", index=False)         # старый ряд под новым именем: даты ≥ 2022
    with pytest.raises(SystemExit, match="2022"):
        D.check_data_root(d)


def test_data_root_empty_dir_refused(tmp_path):
    with pytest.raises(SystemExit):
        D.check_data_root(tmp_path)


def test_dev_cli_refuses_old_series(tmp_path):
    from fastbt import dev
    d = new_dir(tmp_path)
    tiny().to_parquet(d / "ohlcv_daily_full.parquet")
    with pytest.raises(SystemExit):
        dev.main(["--smoke", "--data-root", str(d), "--no-log"])


# ---- ensure(): атомарная запись и готовность по хэшу ----
def fake_build(monkeypatch, calls):
    def fb(main=None, tickers=None):
        calls.append(1)
        return tiny()
    monkeypatch.setattr(D, "build", fb)


def test_ensure_writes_marker_and_is_idempotent(tmp_path, monkeypatch):
    calls = []
    fake_build(monkeypatch, calls)
    p = tmp_path / "s.parquet"
    D.ensure(p)
    D.ensure(p)
    assert D.is_ready(p) and len(calls) == 1
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]


def test_truncated_file_is_not_ready(tmp_path, monkeypatch):
    calls = []
    fake_build(monkeypatch, calls)
    p = tmp_path / "s.parquet"
    D.ensure(p)
    good = p.read_bytes()
    p.write_bytes(good[: len(good) // 2])             # оборвано посередине записи (маркер — от целого файла)
    assert not D.is_ready(p)
    D.ensure(p)                                       # пересобирается, а не «файл есть — значит готов»
    assert D.is_ready(p) and len(calls) == 2
    pd.read_parquet(p)
    p.write_bytes(good[: len(good) // 2])
    D.marker(p).unlink()                              # и без маркера — не готов
    assert not D.is_ready(p)


def _worker(path):
    import time
    D.build = lambda main=None, tickers=None: (time.sleep(1.0), tiny())[1]
    D.ensure(Path(path))
    pd.read_parquet(path)                             # после ensure файл читается целиком


def test_parallel_ensure_never_reads_partial(tmp_path):
    p = tmp_path / "s.parquet"
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=_worker, args=(str(p),)) for _ in range(2)]
    [x.start() for x in ps]
    [x.join(30) for x in ps]
    assert [x.exitcode for x in ps] == [0, 0]
    assert D.is_ready(p)
