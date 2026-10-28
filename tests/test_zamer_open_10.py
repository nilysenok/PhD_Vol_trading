"""Справочный замер open 10:00 (Ж-087 п. 5): синтетика с известным ответом и мутация «чтение ≥ 2020»."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import zamer_open_10 as z  # noqa: E402
from fastbt import bars5m  # noqa: E402

COLS = ["begin", "open", "high", "low", "close", "volume", "value"]


def mk(day, hm, o, c=None):
    c = o if c is None else c
    return [pd.Timestamp(f"{day} {hm}"), o, o, o, c, 1, 1.0]


def write(src: Path, t: str, y: int, rows):
    src.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(src / f"{t}_{y}.csv", index=False)


def den(day, first_hm="09:55", o_first=100.0, o10=101.0, c1455=100.0, o15=102.0):
    return [mk(day, first_hm, o_first), mk(day, "10:00", o10), mk(day, "14:55", c1455), mk(day, "15:00", o15)]


def test_znacheniya_i_dolya_pervogo_bara(tmp_path):
    src = tmp_path / "s"
    write(src, "AAA", 2018, den("2018-03-01") + den("2018-03-02", "09:50", o_first=50.0, o10=51.0, c1455=100.0, o15=99.0))
    d = z.po_dnyam(z.chitat("AAA", src))
    assert d["a"].round(6).tolist() == [1.0, 2.0] and d["c"].round(6).tolist() == [2.0, 1.0]
    assert d["first_t"].tolist() == [595, 590]
    t = z.tablica(d)
    assert t[(t.metrika == "pervyj_bar_0950") & (t.god == "all")]["dolya_dnej"].iloc[0] == 0.5
    assert t[(t.metrika == "open10_vs_pervyj") & (t.god == "all")]["max_pct"].iloc[0] == pytest.approx(2.0)


def test_net_baru_10_den_ne_v_metriku(tmp_path):
    src = tmp_path / "s"
    write(src, "AAA", 2018, [mk("2018-03-01", "09:55", 100.0), mk("2018-03-01", "14:55", 100.0), mk("2018-03-01", "15:00", 102.0)])
    d = z.po_dnyam(z.chitat("AAA", src))
    assert pd.isna(d["a"].iloc[0]) and d["c"].iloc[0] == pytest.approx(2.0)


def test_pustaya_lenta():
    assert len(z.po_dnyam(pd.DataFrame(columns=COLS))) == 0
    assert z.statistika(pd.Series([], dtype=float))[0] == 0


def test_fayly_2020_ne_otkryvayutsya(tmp_path):
    src = tmp_path / "s"
    write(src, "AAA", 2019, den("2019-03-01"))
    (src / "AAA_2020.csv").write_text("мусор, который нельзя читать")
    assert len(z.chitat("AAA", src)) == 4


def test_mutatsiya_chteniya_za_predelami_lovitsya(tmp_path, monkeypatch):
    """Мутация: _clip возвращает всё. Строка 2020 в файле 2019 обязана вызвать отказ."""
    src = tmp_path / "s"
    write(src, "AAA", 2019, den("2019-12-30") + den("2020-01-09"))
    assert z.chitat("AAA", src)["begin"].max() < pd.Timestamp("2020-01-01")
    monkeypatch.setattr(bars5m, "_clip", lambda d, s, e: d)
    with pytest.raises(AssertionError):
        z.chitat("AAA", src)
