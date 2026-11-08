"""V6-1a, G11: загрузчик только SBER (аудит путей), тождество Σr² = RV, признаки S (s1–s9, известные ответы, без NaN после прогрева), PIT-инвариантность и мутации:
признак дня t⁺ (аггрегаты со сдвигом −1), среднее объёма с будущими днями, бары дня t⁺ в признаке дня d — ловятся через боевой `raw_features` (изменение баров после d не должно менять строки ≤ d)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import daily as D
from models_v2.s10 import hp_paths as P
from models_v2.s13 import v6_1_feat as FE, v6_1_h as HH

N_DAYS, N_BARS = 70, 53


def make_bars(seed: int = 3, days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d0 = pd.Timestamp("2016-01-04")
    rows, px = [], 100.0
    for k in range(days):
        day = d0 + pd.offsets.BDay(k)
        for j in range(N_BARS):
            o = px
            px = px * np.exp(rng.normal(0, 0.002))
            rows.append((day + pd.Timedelta(minutes=600 + 10 * j), o, max(o, px), min(o, px), px, float(rng.integers(100, 1000))))
    return pd.DataFrame(rows, columns=["begin", "open", "high", "low", "close", "volume"])


def daily_of(bars):
    return D.build_daily(bars, index_days=None)[0]


def corrupt_after(bars: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Все бары сессий > day заменены другими (другая цена и объём): признаки дней ≤ day не должны измениться."""
    b = bars.copy()
    late = b["begin"].dt.normalize() > day
    rng = np.random.default_rng(99)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, int(late.sum()))))
    b.loc[late, "close"] = px
    b.loc[late, "open"] = np.r_[px[:1], px[:-1]]
    b.loc[late, "volume"] = rng.integers(1, 5000, int(late.sum())).astype(float)
    return b


def leaks(day_idx: int = 45) -> bool:
    """True, если признаки дней ≤ day изменились после замены баров сессий > day."""
    bars = make_bars()
    day = daily_of(bars).index[day_idx]
    b2 = corrupt_after(bars, day)
    a = FE.raw_features(bars, daily_of(bars))
    c = FE.raw_features(b2, daily_of(b2))
    upto = a.index <= day
    return not np.allclose(a[upto].to_numpy(), c[upto].to_numpy(), equal_nan=True, rtol=0, atol=1e-12)


def test_no_leak_on_synthetic():
    assert leaks() is False


def test_mutation_features_of_day_t_plus_caught(monkeypatch):
    real = FE.day_aggregates
    monkeypatch.setattr(FE, "day_aggregates", lambda bars: real(bars).shift(-1))          # s4…s9 дня d из сессии t⁺
    assert leaks() is True


def test_mutation_volume_mean_with_future_days_caught(monkeypatch):
    monkeypatch.setattr(FE, "vol_ratio", lambda vol, win=FE.WIN_VOL: vol / vol.rolling(win, min_periods=1, center=True).mean())
    assert leaks() is True


def test_mutation_t_plus_bars_in_share_of_day_caught(monkeypatch):
    real = FE.day_aggregates

    def bad(bars):
        g = real(bars)
        g["rv_last"] = g["rv_last"].shift(-1)                                               # бары дня t⁺ в признаке дня d
        return g
    monkeypatch.setattr(FE, "day_aggregates", bad)
    assert leaks() is True


def test_known_answers_one_day():
    bars = make_bars(days=30)
    daily = daily_of(bars)
    day = daily.index[25]
    b = bars[bars["begin"].dt.normalize() == day].reset_index(drop=True)
    r = np.r_[np.log(b["close"][0] / b["open"][0]), np.diff(np.log(b["close"].to_numpy()))]
    raw = FE.raw_features(bars, daily).loc[day]
    q = r * r
    n = len(r)
    k = np.arange(n)
    assert raw["s4"] == pytest.approx(q[r < 0].sum() / q.sum(), rel=1e-12)
    assert raw["s5"] == pytest.approx(np.log(b["close"].iloc[-1] / b["open"].iloc[0]) / np.sqrt(q.sum()), rel=1e-12)
    assert raw["s6"] == pytest.approx(daily.loc[day, "j"] / daily.loc[day, "rv"], rel=1e-12)
    assert raw["s8"] == pytest.approx(q[3 * k < n].sum() / q.sum(), rel=1e-12) and raw["s9"] == pytest.approx(q[3 * k >= 2 * n].sum() / q.sum(), rel=1e-12)
    vols = bars.groupby(bars["begin"].dt.normalize())["volume"].sum()
    v20 = vols.loc[:day].iloc[-20:]
    assert raw["s7"] == pytest.approx(np.log(v20.iloc[-1] / v20.mean()), rel=1e-12)


def test_assemble_s1_s3_use_ln_h_and_identity():
    bars = make_bars(days=60)
    daily = daily_of(bars)
    raw = FE.raw_features(bars, daily)
    h = pd.Series(daily["rv"].to_numpy() * 1.5, index=daily.index).dropna().iloc[25:]
    S = FE.assemble(raw, h)
    assert np.allclose(S["s1"], np.log(1 / 1.5)) and list(S.columns) == FE.NAMES
    assert FE.identity_max(bars, daily) <= FE.TOL_ID


def test_loader_opens_only_sber_files(monkeypatch):
    seen = []
    real = pd.read_parquet
    monkeypatch.setattr(FE.pd, "read_parquet", lambda path, *a, **k: seen.append(str(path)) or real(path, *a, **k))
    FE.load_inputs()
    assert sorted(seen) == sorted(str(p) for p in FE.allowed_files())
    assert all("SBER" in p for p in seen) and not any("IMOEX" in p for p in seen)
    assert str(P.IMOEX) not in seen


def test_audit_catches_foreign_ticker_path():
    ok = {str(p) for p in FE.allowed_files()}
    assert str(P.bars_path("LKOH")) not in ok and str(P.IMOEX) not in ok


def test_real_s_dates_no_nan_and_identity():
    p, k1 = HH.H.make_panel(HH.daily_sber()), HH.k1_stored()
    tab = HH.h_table(p, k1)
    bars, daily = FE.load_inputs()
    S = FE.assemble(FE.raw_features(bars, daily), tab["h"])
    assert S.index[0] == pd.Timestamp("2015-07-07") and S.index[-1] == pd.Timestamp("2026-02-26") and not S.isna().any().any()
    assert FE.identity_max(bars, daily) <= FE.TOL_ID
