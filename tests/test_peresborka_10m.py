"""Пересборка 10-мин ряда из 1-мин (Ж-116 п. 1): синтетика, без данных проекта и сети."""
import importlib.util
from pathlib import Path

import pandas as pd

from fastbt import daily_1850 as d18

_p = Path(__file__).resolve().parents[1] / "scripts" / "data" / "peresborka_10m_2014_2015.py"
_s = importlib.util.spec_from_file_location("peresb", _p)
pb = importlib.util.module_from_spec(_s)
_s.loader.exec_module(pb)

DAY = "2015-03-02"


def _m(rows):
    """rows: (HH:MM, open, high, low, close, volume)."""
    return pd.DataFrame([{"begin": f"{DAY} {t}:00", "open": o, "high": h, "low": lo, "close": c, "volume": v}
                         for t, o, h, lo, c, v in rows])


def _slots(out):
    return [f"{x:%H:%M}" for x in out["timestamp"]]


def test_konstanty_kak_v_proekte():
    assert (pb.OPEN_MIN, pb.CUT_MIN) == (d18.OPEN_MIN, d18.CUT_MIN)


def test_agregaciya_sohranyaet_ohlcv():
    m = _m([("10:09", 12, 13, 11, 12.5, 7), ("10:00", 10, 10.5, 9.5, 10.2, 3), ("10:05", 10.2, 14, 10, 12, 5)])
    out = pb.agg_1m_to_10m(m)  # вход намеренно не по порядку: open/close берутся по времени
    r = out.iloc[0]
    assert _slots(out) == ["10:00"]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (10, 14, 9.5, 12.5, 15)
    assert out["volume"].dtype == "int64"


def test_bary_vne_okna_otbrasyvayutsya():
    m = _m([("09:49", 1, 1, 1, 1, 1), ("09:59", 2, 2, 2, 2, 1), ("18:49", 3, 3, 3, 3, 1),
            ("18:50", 4, 4, 4, 4, 1), ("18:59", 5, 5, 5, 5, 1), ("19:02", 6, 6, 6, 6, 1)])
    assert _slots(pb.agg_1m_to_10m(m)) == ["09:50", "18:40"]


def test_mutaciya_bar_1850_voshyol_lovitsya():
    m = _m([("18:49", 3, 3, 3, 3, 1), ("18:50", 4, 4, 4, 4, 1)])
    mut = pb.agg_1m_to_10m(m, cut_min=pb.CUT_MIN + 10)  # подложено: окно до 19:00
    assert "18:50" in _slots(mut)
    assert "18:50" not in _slots(pb.agg_1m_to_10m(m))


def test_dyra_ne_zapolnyaetsya():
    m = _m([("10:00", 1, 1, 1, 1, 1), ("12:30", 2, 2, 2, 2, 1)])
    assert _slots(pb.agg_1m_to_10m(m)) == ["10:00", "12:30"]


def _bars(n=3, vol=10):
    ts = pd.date_range(f"{DAY} 10:00", periods=n, freq="10min")
    return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": vol})


def test_sverka_sovpadenie_i_dyra():
    new, src = _bars(3), _bars(3)
    r = pb.compare(new, src)
    assert (r["bad_on_equal_days"], r["bad_common"], r["only_rebuilt"]) == (0, 0, 0)
    r = pb.compare(new, src.iloc[:2])  # дыра в сверяемом ряду: день неполный, лишний бар виден
    assert r["only_rebuilt"] == 1 and r["days_equal"] == 0 and r["bad_common"] == 0


def test_sverka_lovit_podmenu_obyoma_i_ceny():
    new = _bars(3)
    for col, val in (("volume", 11), ("close", 1.5 + 1e-6)):
        src = _bars(3)
        src.loc[1, col] = val
        r = pb.compare(new, src)
        assert r["bad_on_equal_days"] == 1 and r["bad_common"] == 1, col


def test_sverka_dopusk_cen():
    src = _bars(3)
    src.loc[1, "close"] = 1.5 + 1e-12
    assert pb.compare(_bars(3), src)["bad_common"] == 0
