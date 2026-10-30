"""Лист sobytiya-ne-wf: файлы содержат 2022+ ради wf, а период train4 (не wf) их не видит.

Эталоны (SHA-256) сняты на базе kanon-v22-pereschet (файлы до докачки wf): события — прежним кодом sobytiya_pit на старом
events.csv (запросы раз в 30 дней 2016–2021, vpered=400); ставка < 2022, бары SBER и IMOEX < 2022 — на старых parquet.
Сеть не нужна. Мутации: «фильтр снят» (падение на чтении строк 2022+), «фильтр пропускает 2022».
"""
import hashlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from fastbt import daily_1850 as d18
from fastbt import msp
from fastbt import sobytiya_pit as pit
from fastbt import vneshnie_1850 as v18

ROOT = Path(__file__).resolve().parent.parent
SOB_SHA, SOB_N = "c029e3a18769d224545517dd8d43f6594b2f2d1ed4859f343aaf71fcb65ed2ba", 2953
RATE_SHA = "f33f92507cf438417664bb4f089e749bf60dd099b69f354eea139045bf365415"
SBER_SHA = "ba1ee3427335f9fffa6ef4032659e5091b8cdd17d8c872bf31402f4ea881cd2d"
IMOEX_SHA = "e781838b43b73798546dbb4a9656cc0351ba71072c37f1009a880d907632c755"


def _sha(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def _sobytiya():
    """(SHA, число) событий train4 по реальному файлу; 400 дней вперёд заходят за 2021-12-31."""
    h, n, d = hashlib.sha256(), 0, date(2016, 1, 1)
    while d < date(2022, 1, 1):
        for r in pit.sobytiya_na(d, None, 400):
            h.update(f"{d}|{r['ticker']}|{r['tip']}|{r['data_obyavleniya']}|{r['data_sobytiya']}\n".encode())
            n += 1
        d += timedelta(days=30)
    return h.hexdigest(), n


def test_fajl_soderzhit_2022_dlya_wf():
    assert any(r["data_sobytiya"] >= date(2022, 1, 1) for r in pit._chitat())


def test_sobytiya_train4_pobajtovo_kak_na_baze():
    assert _sobytiya() == (SOB_SHA, SOB_N)


def test_zapros_2022_na_train4_otkaz():
    for d in ("2022-01-01", "2023-05-01", "2026-02-13"):
        with pytest.raises(PermissionError):
            pit.sobytiya_na(d)


def test_zapros_2021_vpered_za_granicu_ne_padaet_i_ne_vidit_2022():
    r = pit.sobytiya_na("2021-12-31", vpered=400)
    assert all(x["data_sobytiya"] < date(2022, 1, 1) for x in r)


def test_mutatsiya_filtr_snyat_lovitsya(monkeypatch):
    """Фильтр снят: строки 2022+ доходят до проверки — PermissionError на чтении (регрессия wf-dannye)."""
    monkeypatch.setattr(pit, "_rows_dev", lambda path=None: pit._rows(path))
    with pytest.raises(PermissionError):
        _sobytiya()


def test_mutatsiya_filtr_propuskaet_2022_lovitsya(monkeypatch):
    monkeypatch.setattr(pit, "DEV_END", date(2023, 1, 1))
    try:
        got = _sobytiya()
    except PermissionError:      # граница сдвинута — режим dev не включается, строгое чтение отказывает: тоже поймано
        return
    assert got != (SOB_SHA, SOB_N)


def test_stavka_do_2022_pobajtovo():
    r = msp.rate_cbr()
    assert _sha(r[r.index < "2022-01-01"].to_frame("v")) == RATE_SHA
    assert len(msp.rate_on(pd.date_range("2018-01-01", "2021-12-31"))) == 1461


def test_bary_do_2022_pobajtovo():
    sber = d18.read_bars(ROOT / "moex_discovery/data/raw/candles_10m/SBER.parquet", pd.Timestamp("2022-01-01"))
    assert _sha(sber) == SBER_SHA
    imoex = v18.read_bars(ROOT / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet")
    assert _sha(imoex) == IMOEX_SHA
    assert imoex["timestamp"].max() < pd.Timestamp("2022-01-01")


def test_mutatsiya_granitsa_bary_sdvinuta_lovitsya(monkeypatch):
    monkeypatch.setattr(v18, "END", pd.Timestamp("2023-01-01"))
    imoex = v18.read_bars(ROOT / "moex_discovery/data/final_v6/candles_10m/external/IMOEX.parquet")
    assert _sha(imoex) != IMOEX_SHA
