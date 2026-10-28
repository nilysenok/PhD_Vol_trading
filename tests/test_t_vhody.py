"""t_vhody: сборка inp для T16–T20; строки ≥ 2022 не попадают в inp даже из файла, где они есть (синтетика во временном каталоге).

Мутация (№ 17, чтение за границей периода): CUT сдвинут на конец 2022 — тест обязан упасть.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import t_vhody

DNI = ["2021-12-29", "2021-12-30", "2022-01-04"]


def _bars10(days, base=100.0):
    rows = []
    for j, d in enumerate(days):
        for k in range(53):                                              # бары 09:50 … 18:40
            ts = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=50 + 10 * k)
            rows.append(dict(timestamp=ts, open=base, high=base + 1, low=base - 1, close=base + 0.01 * k + j, volume=10))
    return pd.DataFrame(rows)


def _fayly(tmp):
    main = tmp / "main"
    (main / "moex_discovery" / "data" / "raw" / "candles_10m").mkdir(parents=True)
    _bars10(DNI).to_parquet(main / "moex_discovery" / "data" / "raw" / "candles_10m" / "A.parquet")
    oh = pd.DataFrame({"ticker": "A", "date": pd.to_datetime(DNI), "open": 100.0, "high": 101.0, "low": 99.0,
                       "close": [100.0, 101.0, 102.0], "volume": [5, 6, 7]})
    oh.to_parquet(tmp / "ohlcv.parquet")
    im = _bars10(DNI, 3000.0)
    im["begin"] = im["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    im.drop(columns="timestamp").assign(value=1.0).to_parquet(tmp / "IMOEX.parquet")
    return main, tmp / "ohlcv.parquet", tmp / "IMOEX.parquet"


def _inp(tmp):
    main, oh, im = _fayly(tmp)
    return t_vhody.zagruzit(["A"], main=main, ohlcv=oh, imoex=im)


def test_stroki_2022_ne_popadayut_v_inp(tmp_path):
    inp = _inp(tmp_path)
    x = inp["tickers"]["A"]
    assert [str(d) for d in x["dates"]] == ["2021-12-29", "2021-12-30"]
    assert inp["dates"].max() < np.datetime64("2022-01-01")
    for k in ("close", "high", "low", "vol", "sigma", "sigma5", "sigma66", "mkt_close", "mkt_sigma"):
        assert len(x[k]) == 2                                           # ряды выровнены на даты тикера, без 2022-01-04
    assert list(x["close"]) == [100.0, 101.0] and list(x["vol"]) == [5.0, 6.0]
    assert np.isfinite(x["mkt_close"]).all()                            # IMOEX на обе даты есть
    assert np.isnan(x["sigma"]).all()                                   # истории 22 дня нет — NaN, без подстановки (Д4)


def test_mutatsiya_cut_za_granitsey_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(t_vhody, "CUT", pd.Timestamp("2022-12-31"))
    inp = _inp(tmp_path)
    assert str(inp["dates"].max()) != "2021-12-30"                      # мутант читает 2022 — основной тест выше упал бы


def test_dividendy_tablitsa_obrezka(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame({"ticker": ["A", "A"], "last_cum_date": ["2021-07-01", "2022-07-01"], "announce_date": ["2021-05-01", "2022-05-01"],
                  "dps_rub": [1.0, 2.0]}).to_csv(p, index=False)
    t = t_vhody.dividendy_tablitsa(p)
    assert list(t["dps_rub"]) == [1.0]
