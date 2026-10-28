"""Входы зоны Z2 (scripts/dannye/z2_vkhody.py): синтетика, реальные данные не читаются.
python3 -m pytest tests/test_z2_vkhody.py -q"""
import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fastbt import daily_1850

_spec = importlib.util.spec_from_file_location("z2", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "z2_vkhody.py")
z2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(z2)


def _src(tmp_path):
    """Мини-«основная копия»: 10-минутки одного тикера 2021 / 2022 / 2024 / 2025 (+ вечерний бар 19:00 и бар 09:50)."""
    d = tmp_path / "moex_discovery" / "data" / "raw" / "candles_10m"
    d.mkdir(parents=True)
    ts = []
    for day in ["2021-06-01", "2021-12-30", "2022-01-03", "2024-12-30", "2025-01-09"]:
        ts += [pd.Timestamp(f"{day} {h}") for h in ("09:50", "10:00", "18:40", "19:00")]
    n = len(ts)
    pd.DataFrame({"timestamp": ts, "open": range(1, n + 1), "high": range(2, n + 2), "low": range(n), "close": [x + 0.5 for x in range(n)],
                  "volume": [10] * n}).to_parquet(d / "AAA.parquet")
    return tmp_path


def _dev(main):
    return daily_1850.build(main, ["AAA"])           # зона dev: < 2022


def test_ryad_do_2025_i_ravenstvo_2021_s_dev(tmp_path):
    main = _src(tmp_path)
    new = z2.build_daily(main, ["AAA"])
    assert new["date"].max() == pd.Timestamp("2024-12-30")
    assert (new["date"] < pd.Timestamp("2025-01-01")).all()
    assert z2.gate_dev(new, _dev(main)) == 2


def test_mutatsiya_otkrytie_09_50_lovitsya_gejtom_2021(tmp_path, monkeypatch):
    main = _src(tmp_path)
    dev = _dev(main)
    monkeypatch.setattr(daily_1850, "OPEN_MIN", 10 * 60)         # «новый» ряд собран иным правилом, чем dev
    with pytest.raises(AssertionError):
        z2.gate_dev(z2.build_daily(main, ["AAA"]), dev)


def test_otkaz_pri_stroke_2025():
    with pytest.raises(ValueError):
        z2.assert_cut(pd.Series(pd.to_datetime(["2024-12-30", "2025-01-02"])), "x")
    z2.assert_cut(pd.Series(pd.to_datetime(["2024-12-30"])), "x")


def _imoex(tmp_path):
    p = tmp_path / "IMOEX.parquet"
    b = ["2021-12-30 18:50:00", "2022-01-03 10:00:00", "2024-12-30 18:50:00", "2025-01-09 10:00:00"]
    pd.DataFrame({"begin": b, "end": b, "open": 1.0, "close": 1.0, "high": 1.0, "low": 1.0, "value": 1.0, "volume": 1}).to_parquet(p)
    return p


def test_imoex_otsechka_2025_pri_chtenii(tmp_path, monkeypatch):
    p = _imoex(tmp_path)
    assert list(z2.build_imoex(p)["begin"]) == ["2021-12-30 18:50:00", "2022-01-03 10:00:00", "2024-12-30 18:50:00"]
    seen = []
    orig = pq.read_table
    monkeypatch.setattr(z2.pq, "read_table", lambda *a, **k: (seen.append(k.get("filters")), orig(*a, **k))[1])
    z2.build_imoex(p)
    assert seen and seen[0] == [("begin", "<", "2025-01-01")]       # фильтр на чтении, а не после материализации


def test_mutatsiya_bez_otsechki_2025_lovitsya(tmp_path, monkeypatch):
    p = _imoex(tmp_path)
    monkeypatch.setattr(z2, "END", pd.Timestamp("2030-01-01"))     # отсечка снята: строка 2025 прочитана
    with pytest.raises(ValueError, match="2025"):
        z2.build_imoex(p)


def test_sdvig_lovit_sdvinutyj_ryad():
    dates = pd.bdate_range("2022-01-03", periods=400)
    rng = pd.Series(range(400)).map(lambda i: 0.01 + 0.005 * ((i * 7919) % 13) / 13).to_numpy()
    daily = pd.DataFrame({"ticker": "A", "date": dates, "high": 100 * (1 + rng), "low": 100.0})
    rv = pd.DataFrame({"ticker": "A", "date": dates, "rv": (2 * rng) ** 2 / 4})
    ok = z2.sdvig(daily, rv)
    assert ok.loc[ok["corr"].idxmax(), "shift"] == 0
    bad = z2.sdvig(daily.assign(date=daily["date"].shift(-1)).dropna(), rv)     # ряд сдвинут на день
    assert bad.loc[bad["corr"].idxmax(), "shift"] != 0
