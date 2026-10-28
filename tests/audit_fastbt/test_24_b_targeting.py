"""24. Подход B (С9, таргетирование): вес = min(cap, медиана σ / σ) с полосой бездействия (FA2b).

s9.S9.b_weights: медиана — по прошлым 252 барам без текущего (минимум 50 значений), σ — строка с
датой ≤ d, вес бара d умножает позицию A бара d (доход d→d+1); полоса: вес пересчитывается, только
когда |w_d / w_держ − 1| > полосы; cap 1 / 1,5 / 2. Ответ вручную на ряду σ h1:
  σ[0..49] = 0,01 / 0,03 поочерёдно; σ[50] = 0,012; σ[51] = 0,04; σ[52] = 0,03; σ[53] = 0,028;
  σ[54] = 0,0295; далее 0,03.
  d=50: медиана прошлых = 0,02 → w = 1,667 → cap 1,5 → 1,5 (с текущим днём медиана была бы 0,012);
  d=51: медиана 0,012 → 0,3;  d=52: медиана (0,012+0,03)/2 = 0,021 → 0,7;
  d=53: медиана 0,03 → 1,0714;  d=54: медиана 0,029 → 0,9831, |0,9831/1,0714 − 1| = 8,2% ≤ 10% →
  держим 1,0714 (без полосы — 0,9831). До d=50 медианы нет (< 50 значений) → вес 1.
Ряд σ h5 другой (σ[50] = 0,04 при 0,02 остальных): вес h5 на d=50 — 0,5, а не вес h1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import read_dev
from fastbt import s9, s9_stats
from fastbt.data import OHLCV

N = 70
DATES = pd.bdate_range("2019-01-01", periods=N).values.astype("datetime64[ns]")


def _h1():
    s = np.full(N, 0.03)
    s[0:50:2] = 0.01
    s[50:55] = [0.012, 0.04, 0.03, 0.028, 0.0295]
    return s


def _h5():
    s = np.full(N, 0.02)
    s[50] = 0.04
    return s


def ref_weights(s, cap, band, W=252, minp=50):
    """Независимый расчёт: цикл, numpy.median по s[t−W..t−1]."""
    out, cur = np.empty(len(s)), 1.0
    for t in range(len(s)):
        past = s[max(0, t - W):t]
        past = past[~np.isnan(past)]
        v = np.nan
        if len(past) >= minp and not np.isnan(s[t]):
            v = min(cap, float(np.median(past)) / s[t])
        if v == v and v > 0 and (band <= 0 or abs(v / cur - 1.0) > band):
            cur = v
        out[t] = cur
    return out


def _engine(dates, tickers=("ZZZ",)):
    e = s9.Engine.__new__(s9.Engine)
    e.tickers = list(tickers)
    e.cache = {t: {"dates": dates} for t in tickers}
    e.n_t = np.array([len(dates)] * len(tickers))
    e.nmax = len(dates)
    e.sig = {}
    return e


def _b(e):
    b = s9.S9.__new__(s9.S9)
    b.e = e
    return b


@pytest.fixture
def synth(monkeypatch):
    def load_source(label):
        _, col, _ = s9.SOURCES[label]
        s = {"sigma_pred_h1": _h1, "sigma_pred_h5": _h5}[col]()
        return pd.DataFrame({"date": pd.to_datetime(DATES), "ticker": "ZZZ", "sigma": s})
    monkeypatch.setattr(s9, "load_source", load_source)
    return _b(_engine(DATES))


def test_b_weights_by_hand(synth):
    W = synth.b_weights("hyb_h1", 1.5, 0.10)[0]
    assert np.allclose(W[:50], 1.0)
    assert np.allclose(W[50:55], [1.5, 0.3, 0.7, 0.03 / 0.028, 0.03 / 0.028])
    W0 = synth.b_weights("hyb_h1", 1.5, 0.0)[0]
    assert np.isclose(W0[54], 0.029 / 0.0295)                 # без полосы — пересчёт каждый день
    W2 = synth.b_weights("hyb_h1", 2.0, 0.0)[0]
    assert np.isclose(W2[50], 0.02 / 0.012)                   # cap 2 не режет 1,667
    W1 = synth.b_weights("hyb_h1", 1.0, 0.0)[0]
    assert W1.max() <= 1.0 + 1e-12 and np.isclose(W1[50], 1.0)


@pytest.mark.parametrize("cap", (1.0, 1.5, 2.0))
@pytest.mark.parametrize("band", (0.0, 0.10, 0.20))
def test_b_weights_match_reference(synth, cap, band):
    assert np.allclose(synth.b_weights("hyb_h1", cap, band)[0], ref_weights(_h1(), cap, band))
    assert np.allclose(synth.b_weights("hyb_h5", cap, band)[0], ref_weights(_h5(), cap, band))


def test_b_h5_weights_use_h5(synth):
    assert np.isclose(synth.b_weights("hyb_h5", 1.5, 0.0)[0][50], 0.5)
    for src in (s9.SOURCES, s9.SOURCES_V2):
        for lab in ("hyb_h1", "hyb_h5", "harj_h1"):
            if lab in src:
                p, col, h = src[lab]
                assert h == lab[-2:] and col == f"sigma_pred_{h}" and f"_{h}" in Path(p).name, lab


def test_b_weight_applies_to_same_bar():
    """g_row: вес бара d → строка позиции бара d (доход lr[d] = d→d+1), без сдвига."""
    e = _engine(DATES)
    e.r = {"tick": np.zeros(3, np.int64), "bar": np.array([5, 6, 7]), "cat": np.zeros(3, np.int64)}
    G = np.arange(N, dtype=float)[None, :]
    assert e.g_row(G).tolist() == [5.0, 6.0, 7.0]


def test_b_weights_dev_sber_h5(monkeypatch):
    """Данные ≤ 2021: SBER, σ гибрида h=5 набора v2 — b_weights = независимый расчёт."""
    ch4 = s9.CH4 / "sigma_hyb_h5_dev.parquet"
    if not ch4.exists():
        pytest.skip("нет σ v2")
    monkeypatch.setattr(s9, "SOURCES", dict(s9.SOURCES_V2))
    bars = read_dev(OHLCV, ["ticker", "date"])
    dates = np.sort(bars.loc[bars["ticker"] == "SBER", "date"].values).astype("datetime64[ns]")
    b = _b(_engine(dates, ("SBER",)))
    sg = read_dev(ch4)
    sg = sg[sg["ticker"] == "SBER"].sort_values("date")
    ref_s = pd.merge_asof(pd.DataFrame({"date": pd.to_datetime(dates)}),
                          sg[["date", "sigma_pred_h5"]], on="date")["sigma_pred_h5"].values
    assert np.isfinite(ref_s).sum() > 1000
    for cap, band in ((1.0, 0.0), (1.5, 0.10), (2.0, 0.20)):
        assert np.allclose(b.b_weights("hyb_h5", cap, band)[0], ref_weights(ref_s, cap, band))


# ---------- мутации ----------

def _median_today(s, W):
    return pd.Series(s).rolling(W, min_periods=s9_stats.minp(W)).median().values


def _sigma_next(f):
    def g(dates, d):
        out = f(dates, d)
        return np.r_[out[1:], np.nan]
    return g


def _cap_ignored(mp):
    orig = s9.S9.b_weights
    mp.setattr(s9.S9, "b_weights", lambda self, src, cap, band: orig(self, src, 1e9, band))


def _h5_from_h1(mp):
    mp.setitem(s9.SOURCES, "hyb_h5", s9.SOURCES["hyb_h1"])
    mp.setitem(s9.SOURCES_V2, "hyb_h5", s9.SOURCES_V2["hyb_h1"])


MUTATIONS = {
    "median_with_today": lambda mp: mp.setattr(s9_stats, "past_median", _median_today),
    "sigma_next_day": lambda mp: mp.setattr(s9, "align", _sigma_next(s9.align)),
    "band_off": lambda mp: mp.setattr(s9_stats, "band_hold", lambda w, band: np.where(
        np.isnan(w), 1.0, w)),
    "cap_ignored": _cap_ignored,
    "h5_uses_h1": _h5_from_h1,
}
