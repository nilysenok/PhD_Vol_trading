"""Л6: контракты всех объектов, читаемых ПОСЛЕ замка (аудит после падения открытия 11.11: `check_sealed` читал `date` как колонку, а в `rv_fwd_SEALED.parquet` это индекс).
Писатель запечатывания — настоящий код `scripts/dannye/fwd_sber_m10_check.py` (импортом, синтетика в tmp); схемы настоящих файлов — pyarrow footer без значений."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/fwd"))
sys.path.insert(0, str(ROOT / "scripts/dannye"))
import otkrytie_raschet as RS  # noqa: E402
import fwd_sber_m10_check as CHK  # noqa: E402
import fwd_sber_m10 as WR  # noqa: E402
import fwd_mkt_m10 as WM  # noqa: E402

from models_v2 import daily as D  # noqa: E402

REAL_SEALED = ROOT / "data/fwd/sber_m10/rv_fwd_SEALED.parquet"
REAL_NEW_SBER = ROOT / "data/fwd/sber_m10/SBER_m10_fwd.parquet"
REAL_NEW_IMOEX = ROOT / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet"


def synth_bars(days: int = 8, seed: int = 1) -> pd.DataFrame:
    """Синтетические M10: будни января 2026, основная сессия 10:00–18:50 + вечерняя 19:00–23:50 (признак proxy IMOEX), типы писателя `parse_pages`."""
    rng = np.random.default_rng(seed)
    rows, px = [], 300.0
    for d in pd.bdate_range("2026-01-12", periods=days):
        for ts in list(pd.date_range(d + pd.Timedelta(hours=10), d + pd.Timedelta(hours=18, minutes=50), freq="10min")) + list(pd.date_range(d + pd.Timedelta(hours=19), d + pd.Timedelta(hours=23, minutes=50), freq="10min")):
            o, px = px, px * np.exp(rng.normal(0, 0.001))
            rows.append([ts, o, max(o, px), min(o, px), px, 1000 + int(rng.integers(0, 50)), px * 1000.0])
    df = pd.DataFrame(rows, columns=WR.COLS)
    df["begin"] = pd.to_datetime(df["begin"]).astype("datetime64[ms]")
    df["volume"] = df["volume"].astype("int64")
    return df


def write_sealed(bars: pd.DataFrame, p: Path) -> pd.DataFrame:
    """Запечатывание как в `fwd_sber_m10_check.rv_rows`: `D.build_daily(fwd, proxy_days(fwd))[0].to_parquet(p)`."""
    d = D.build_daily(bars, CHK.proxy_days(bars))[0]
    d.to_parquet(p)
    return d


def test_vosproizvedenie_check_sealed_chitaet_sealed_nastoyashchego_pisatelya(tmp_path):
    bars = synth_bars()
    daily = write_sealed(bars, tmp_path / "rv_fwd_SEALED.parquet")
    sealed = pd.read_parquet(tmp_path / "rv_fwd_SEALED.parquet")
    assert "date" not in sealed.columns and sealed.index.name == "date"  # контракт писателя: date — индекс
    row = RS.check_sealed(daily, sealed)
    assert row["status"] == "ok" and "общих дат 8" in row["detail"]
    assert RS.check_sealed(daily, sealed.reset_index())["status"] == "ok"  # колонка date тоже читается
    bad = sealed.copy()
    bad.iloc[2, bad.columns.get_loc("rv")] *= 1.001
    assert RS.check_sealed(daily, bad)["status"] == "↺"  # расхождение RV по-прежнему ловится


def test_realnaya_skhema_sealed_date_indeks_kak_u_pisatelya(tmp_path):
    write_sealed(synth_bars(), tmp_path / "s.parquet")
    mine, real = pq.read_schema(tmp_path / "s.parquet"), pq.read_schema(REAL_SEALED)  # только footer, значения не читаются
    assert mine.names == real.names and [str(t) for t in mine.types] == [str(t) for t in real.types]
    assert real.pandas_metadata["index_columns"] == ["date"] == mine.pandas_metadata["index_columns"]


def test_skhema_novykh_bar_pisatelyami_ravna_realnym(tmp_path):
    """SBER и IMOEX новые файлы: типы колонок = то, что пишут `fwd_sber_m10.parse_pages` / `fwd_mkt_m10.parse_pages` (читатель `Src` получает те же типы)."""
    b = synth_bars(2)
    p = tmp_path / "w.parquet"
    b.to_parquet(p, index=False)
    for real in (REAL_NEW_SBER, REAL_NEW_IMOEX):
        r = pq.read_schema(real)
        assert r.names == list(b.columns) and [str(t) for t in r.types] == [str(t) for t in pq.read_schema(p).types]
    assert pq.read_schema(real).pandas_metadata["index_columns"] == []  # index=False: begin — колонка


ARCH = {  # файл архивного выхода Л1–Л4 -> колонки, нужные читателю `run_nodes` (сверка `compare` берёт все колонки файла)
    "k1": ["d", "target_date", "rv", "sigma2", "gap"], "k23": ["date", "t_plus", "rv", "sigma2_k2", "sigma2_k3"],
    "cfg8": ["d", "t_plus", "rv", "F"], "x1": ["d", "t_plus", "rv", "M", "C", "s", "F"], "zb": ["d", "t_plus", "rv", "F"], "zb0": ["d", "t_plus", "rv", "F"]}


@pytest.mark.parametrize("name", sorted(ARCH))
def test_arkhivnye_vykhody_l1_l4_imeyut_kolonki_chitatelya(name):
    names = pq.read_schema(ROOT / f"data/fwd/models/{name}.parquet").names
    assert set(ARCH[name]) <= set(names)


def test_kolonki_arkhivnykh_vykhodov_ravny_kolonkam_pisatelej():
    from models_v2.s14 import v7_3_cfg as C8
    from models_v2.s16 import x1_z_walk as ZW
    from models_v2.s17 import fwd_x1
    rd = lambda n: pq.read_schema(ROOT / f"data/fwd/models/{n}.parquet").names  # noqa: E731
    assert rd("cfg8") == C8.COLS and rd("x1") == fwd_x1.COLS and rd("zb") == ZW.OUT_COLS and rd("zb0") == ZW.OUT_COLS


def test_power_csv_kolonki_chitatelya_fwd_doc():
    """`fwd_doc.power_text` читает kind, window, hyp, n_new, block, power_holm7 из POWER.csv (комментарий `#` пропускается)."""
    cols = pd.read_csv(ROOT / "docs/otchety/harj_fwd/POWER.csv", comment="#", nrows=0).columns
    assert {"kind", "window", "hyp", "n_new", "block", "power_holm7"} <= set(cols)


def test_mkt_pisatel_i_sber_pisatel_dayut_odinakovye_kolonki():
    assert list(WM.S.COLS) == list(WR.COLS)  # `fwd_mkt_m10` импортирует писателя SBER
