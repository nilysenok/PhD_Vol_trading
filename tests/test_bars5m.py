"""Тесты fastbt.bars5m на синтетике (без реальных данных)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastbt import bars5m  # noqa: E402

COLS = ["begin", "open", "high", "low", "close", "volume", "value"]


def mk(day: str, hm: str, o, h, lo, c, v=10, val=None):
    return [pd.Timestamp(f"{day} {hm}"), o, h, lo, c, v, float(v * c if val is None else val)]


def df(rows):
    return pd.DataFrame(rows, columns=COLS)


def test_agregatsiya_ohlc_value_i_pusto():
    rows = [mk("2019-03-12", "10:00", 10, 12, 9, 11, 5, 50), mk("2019-03-12", "10:02", 11, 15, 8, 14, 7, 70),
            mk("2019-03-12", "10:04", 14, 14, 13, 13.5, 3, 40),
            mk("2019-03-12", "10:10", 20, 21, 19, 20.5, 1, 20)]       # 10:05 — пусто
    out = bars5m.aggregate(df(rows))
    a, b, c = out.iloc[0], out.iloc[1], out.iloc[2]
    assert (a.open, a.high, a.low, a.close, a.volume, a.value, a.pusto) == (10, 15, 8, 13.5, 15, 160, False)
    assert b.pusto and np.isnan(b.open) and np.isnan(b.close) and b.volume == 0 and b.value == 0
    assert b.ts == pd.Timestamp("2019-03-12 10:05") and c.close == 20.5 and not c.pusto
    assert len(out) == 106 and out.pusto.sum() == 104     # полная сетка основной сессии


def test_okno_sessii():
    rows = [mk("2019-03-12", "09:59", 1, 1, 1, 1), mk("2019-03-12", "10:00", 2, 2, 2, 2),
            mk("2019-03-12", "09:50", 5, 5, 5, 5), mk("2019-03-12", "18:49", 3, 3, 3, 3),
            mk("2019-03-12", "19:00", 9, 9, 9, 9)]                      # вечер — вне обоих окон
    o = bars5m.aggregate(df(rows), "osnovnaya")
    p = bars5m.aggregate(df(rows), "s_predotkrytiem")
    assert len(o) == 106 and len(p) == 108
    assert o.ts.iloc[0] == pd.Timestamp("2019-03-12 10:00") and o.ts.iloc[-1] == pd.Timestamp("2019-03-12 18:45")
    assert p.ts.iloc[0] == pd.Timestamp("2019-03-12 09:50")
    assert o.iloc[0].close == 2 and o.iloc[0].volume == 10        # 09:59 не вошла
    assert p.iloc[1].close == 1 and p.iloc[1].ts == pd.Timestamp("2019-03-12 09:55")
    assert o.iloc[-1].close == 3 and not (o.close == 9).any()


def test_propushchennaya_minuta_vnutri_pyatiminutki():
    rows = [mk("2019-03-12", "10:00", 10, 10, 10, 10, 1), mk("2019-03-12", "10:01", 11, 11, 11, 11, 1),
            mk("2019-03-12", "10:03", 12, 12, 12, 12, 1), mk("2019-03-12", "10:04", 13, 13, 13, 13, 1)]
    a = bars5m.aggregate(df(rows)).iloc[0]
    assert not a.pusto and (a.open, a.close, a.volume) == (10, 13, 4)   # без заполнения дыры


def test_den_bez_torgov_i_pustoy_vkhod():
    assert bars5m.aggregate(df([])).empty
    rows = [mk("2019-03-11", "10:00", 1, 1, 1, 1), mk("2019-03-13", "10:00", 2, 2, 2, 2)]
    out = bars5m.aggregate(df(rows))
    assert sorted(out.date.dt.strftime("%Y-%m-%d").unique()) == ["2019-03-11", "2019-03-13"]  # 12-го нет
    assert len(out) == 212


def write_year(src: Path, ticker: str, year: int, rows):
    src.mkdir(parents=True, exist_ok=True)
    d = df(rows)
    d["begin"] = d["begin"].dt.strftime("%Y-%m-%d %H:%M:%S")
    d.to_csv(src / f"{ticker}_{year}.csv", index=False)


def test_kesh_invalidiruetsya_pri_izmenenii_istochnika(tmp_path):
    src, cache = tmp_path / "src", tmp_path / "cache"
    write_year(src, "AAA", 2019, [mk("2019-03-12", "10:00", 1, 2, 1, 2)])
    a = bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cache)
    assert len(list(cache.glob("*.parquet"))) == 1 and a.iloc[0].close == 2
    bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cache)
    assert len(list(cache.glob("*.parquet"))) == 1               # попадание в кэш
    write_year(src, "AAA", 2019, [mk("2019-03-12", "10:00", 1, 2, 1, 7)])
    b = bars5m.load_bars5m("AAA", "2019-01-01", "2019-12-31", src=src, cache=cache)
    assert b.iloc[0].close == 7 and len(list(cache.glob("*.parquet"))) == 2


def test_chtenie_strogo_po_periodu(tmp_path, monkeypatch):
    src = tmp_path / "src"
    write_year(src, "AAA", 2021, [mk("2021-12-30", "10:00", 1, 1, 1, 1), mk("2022-01-03", "10:00", 9, 9, 9, 9)])
    write_year(src, "AAA", 2022, [mk("2022-02-01", "10:00", 8, 8, 8, 8)])
    opened = []
    orig = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda p, *a, **k: (opened.append(Path(p).name), orig(p, *a, **k))[1])
    out = bars5m.load_bars5m("AAA", src=src, cache=tmp_path / "c")  # период по умолчанию
    assert opened == ["AAA_2021.csv"]                              # файл 2022 не открыт
    assert out.ts.max() < pd.Timestamp("2022-01-01") and out.date.dt.year.max() == 2021


def test_mutatsiya_chteniya_za_predelami_lovitsya(tmp_path, monkeypatch):
    """Мутация: _clip возвращает всё. Проверка периода обязана упасть."""
    src = tmp_path / "src"
    write_year(src, "AAA", 2021, [mk("2021-12-30", "10:00", 1, 1, 1, 1), mk("2022-01-03", "10:00", 9, 9, 9, 9)])

    def check(i):
        out = bars5m.load_bars5m("AAA", cache=tmp_path / f"c{i}", src=src)
        assert out.date.max() <= pd.Timestamp(bars5m.END)

    check(0)                                                       # исходный код проходит
    monkeypatch.setattr(bars5m, "_clip", lambda d, s, e: d)
    monkeypatch.setenv("FASTBT_NO_CACHE", "1")
    with pytest.raises(AssertionError):
        check(1)


def test_nevernoe_okno():
    with pytest.raises(ValueError):
        bars5m.load_bars5m("AAA", window="x")
