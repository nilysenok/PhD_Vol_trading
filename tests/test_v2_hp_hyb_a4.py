"""P2, A4 (hp_hyb_a4): признаки дизайна K2 (Б-2 + F4) SBER на реальных барах (пропуск, если данных нет): обрезка будущего (бары после d повреждены -> признаки на даты <= d не меняются),
z — по прошлому (первые 252 строки NaN), даты строк совпадают с панелью движка. Обучения K2 нет."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_ctrl as CT, hp_hyb_a4 as A4, hp_har as HH, hp_k23 as K23, hp_k23_world as KW, hp_paths as P

real = pytest.mark.skipif(not (P.bars_path("SBER").exists() and CT.ENGINE_PANEL.exists()), reason="нет баров / панели движка")


@pytest.fixture(scope="module")
def world():
    imoex = K23.read_bars("IMOEX")
    return imoex, KW.pool_papers(K23.read_bars), K23.read_bars("SBER")


@real
def test_a4_sber_obrezka_budushchego_data_i_proshloe(world):
    imoex, pool, bars = world
    d0 = pd.Timestamp("2021-06-30")
    a = A4.ticker_frame(bars, imoex, pool, "SBER")
    b = A4.ticker_frame(HH.corrupt_after(bars, d0), imoex, pool, "SBER")
    cols = A4.cols_of(a)
    assert len(cols) >= 10
    ma, mb = a[a["date"] <= d0 - pd.Timedelta(days=5)].set_index("date")[cols], b[b["date"] <= d0 - pd.Timedelta(days=5)].set_index("date")[cols]
    assert len(ma) > 1000 and ma.index.equals(mb.index)
    assert np.array_equal(ma.to_numpy(), mb.to_numpy(), equal_nan=True)                 # признаки на d не видят баров после d
    assert a[cols].iloc[:200].isna().all().all()                                          # z: ≥ 252 наблюдений прошлого
    pan = pd.read_parquet(CT.ENGINE_PANEL, columns=["ticker", "date"])
    pan = pan[(pan["ticker"] == "SBER") & (pan["date"] >= "2017-01-03")]
    assert pan["date"].isin(a["date"]).mean() > 0.99                                      # даты панели покрыты миром K2
