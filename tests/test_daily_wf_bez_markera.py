"""wfП-9: ensure не затирает ряд wf (даты ≥ 2022) без маркера; громкий отказ. Синтетика, без сети."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from fastbt import daily_1850 as D


def _ryad(tag: float, god: str) -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["X", "X"], "date": pd.to_datetime([f"{god}-02-26", f"{god}-02-27"]),
                         "open": [tag, tag], "high": [tag + 1, tag + 1], "low": [tag, tag], "close": [tag, tag], "volume": [5, 6]})


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _ne_zvat(*a, **k):
    raise AssertionError("ensure пересобрал ряд wf")


def test_wf_bez_markera_otkaz_fajl_ne_tronut(tmp_path, monkeypatch):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    _ryad(2.0, "2026").to_parquet(p, index=False)                       # ряд wf без маркера
    s0, t0 = _sha(p), p.stat().st_mtime_ns
    monkeypatch.setattr(D, "build", _ne_zvat)
    with pytest.raises(RuntimeError, match="маркера нет") as e:
        D.ensure(p, tmp_path)
    assert str(p) in str(e.value)
    assert (_sha(p), p.stat().st_mtime_ns) == (s0, t0) and not D.marker(p).exists()


def test_dev_bez_markera_prezhnyaya_pobajtno(tmp_path, monkeypatch):
    p, ref = tmp_path / "ohlcv_daily_1850.parquet", tmp_path / "ref.parquet"
    _ryad(5.0, "2021").to_parquet(p, index=False)
    new = _ryad(9.0, "2021")
    new.to_parquet(ref, index=False)
    monkeypatch.setattr(D, "build", lambda main=None: new)
    D.ensure(p, tmp_path)
    assert p.read_bytes() == ref.read_bytes() and D.is_ready(p)


def test_net_fajla_sobiraetsya(tmp_path, monkeypatch):
    p = tmp_path / "ohlcv_daily_1850.parquet"
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021"))
    D.ensure(p, tmp_path)
    assert D.is_ready(p)


def test_mutaciya_ensure_pishet_poverh_wf_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_ne_zatirat_wf", lambda path: None)         # мутация: защита снята
    p = tmp_path / "ohlcv_daily_1850.parquet"
    _ryad(2.0, "2026").to_parquet(p, index=False)
    monkeypatch.setattr(D, "build", lambda main=None: _ryad(9.0, "2021"))
    D.ensure(p, tmp_path)
    assert pd.read_parquet(p)["open"].iloc[0] == 9.0                    # затёрт: тест отказа выше эту мутацию роняет
