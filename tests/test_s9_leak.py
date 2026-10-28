"""С9: «будущее не влияет» для признаков затворов и весов (fastbt.s9_stats, fastbt.s9).

Вход после бара t заменяется шумом → значения на барах ≤ t обязаны совпасть побайтно.
Плюс: текущий бар не входит в окно ранга/медианы (сдвиг на 1); окна валидации кончаются
на _year_bounds(Y)[0]; σ-источники С9 не содержат дат после 31.12.2021.
python3 -m pytest tests/test_s9_leak.py -q
"""
import numpy as np
import pandas as pd
import pytest

from fastbt.masks import hyst_gate
from fastbt.s9_stats import band_hold, ema, minp, past_median, past_rank

RNG = np.random.RandomState(7)
N = 900
S = np.exp(RNG.normal(-4, 0.3, N))
S[:20] = np.nan


def _noise_after(x, t):
    y = x.copy()
    y[t + 1:] = np.exp(RNG.normal(-3, 1.0, len(y) - t - 1))
    return y


def _feats(s):
    out = {}
    for W in (126, 252, 504):
        out[f"rank{W}"] = past_rank(s.copy(), W, minp(W))
        out[f"med{W}"] = past_median(s, W)
    for sp in (3, 5, 10):
        out[f"ema{sp}"] = ema(s, sp)
    r = out["rank252"]
    out["hyst"] = hyst_gate(np.ascontiguousarray(r), 0.6, 0.8).astype(float)
    chg = np.full(len(s), np.nan)
    chg[5:] = s[5:] / s[:-5] - 1
    out["chg5"] = chg
    w = np.minimum(2.0, out["med252"] / s)
    out["band20"] = band_hold(np.ascontiguousarray(w), 0.2)
    return out


@pytest.mark.parametrize("t", [300, 555, 800])
def test_future_noise(t):
    base = _feats(S)
    pert = _feats(_noise_after(S, t))
    for k in base:
        a, b = base[k][:t + 1], pert[k][:t + 1]
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), k


def test_current_bar_not_in_window():
    s = np.arange(1.0, 400.0)
    r = past_rank(s.copy(), 252, minp(252))
    assert r[300] == 1.0                     # все прошлые меньше: текущий не входит в окно
    m = past_median(s, 252)
    assert m[300] == np.median(s[300 - 252:300])


def test_validation_windows_end_before_year():
    from fastbt.data import year_bounds
    dates = pd.bdate_range("2014-06-01", "2022-06-01").values.astype("datetime64[ns]")
    for Y in (2018, 2019, 2020, 2021):
        ve = year_bounds(dates, Y)[0]
        assert pd.Timestamp(dates[ve - 1]) < pd.Timestamp(f"{Y}-01-01")
        assert pd.Timestamp(dates[ve]) >= pd.Timestamp(f"{Y}-01-01")


def test_sources_cut():
    from fastbt import s9
    for lab, (p, col, _) in s9.SOURCES.items():
        if not p.exists():
            pytest.skip("σ-источники С9 не собраны")
        d = s9.load_source(lab)
        assert d["date"].max() <= s9.CUT, lab
    if s9.RVI.exists():
        assert pd.read_parquet(s9.RVI)["date"].max() <= s9.CUT
