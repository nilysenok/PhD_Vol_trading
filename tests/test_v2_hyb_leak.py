"""GB5/G26: главный тест для строк Б-1/Б-2/Б-3/Б-5 — прогноз на d побитно не меняется при порче/обрезке баров после 18:50 дня d.
Мутации: остаток Б-1 по HAR-J(б) на всей выборке (m4); признаки +1 день (m1)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily, har  # noqa: E402
from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s3 import gen, pb, resid  # noqa: E402

D0 = 820
ROWS = {r.name: r for r in pb.rows()}
NAMES = ["b1_xgb_c100", "b2_252_lgbm", "b4_m_xgb", "b3_har_lgbm", "b5_lgbm_mono",
         "b1_xgb_c050", "b1_xgb_c075", "b1_xgb_c100_nobias"]  # три последние — строки Г1 этапа 3 (g1_c050, g1_c075, g1_c100_nobias)


def damage(bars, d, mode):
    day = bars["begin"].dt.normalize()
    mask = (day > d) | ((day == d) & (bars["begin"].dt.strftime("%H:%M") >= "18:50"))
    if mode == "drop":
        return bars[~mask]
    b = bars.copy()
    rng = np.random.default_rng(7)
    for c in ("open", "high", "low", "close"):
        b.loc[mask, c] = rng.uniform(50, 500, int(mask.sum()))
    return b


def synth(n_days=900, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-06", periods=n_days)
    lvl = np.exp(np.cumsum(rng.normal(0, 0.1, n_days)) * 0.3)
    frames, c0 = [], 100.0
    for k, day in enumerate(days):
        for start, m in (("10:00", 53), ("18:50", 4)):
            t0 = pd.Timestamp(str(day.date()) + " " + start)
            close = c0 * np.exp(np.cumsum(rng.normal(0, 0.001 * lvl[k], m)))
            opn = np.concatenate([[c0], close[:-1]])
            frames.append(pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * j) for j in range(m)], "open": opn,
                                        "high": np.maximum(opn, close), "low": np.minimum(opn, close),
                                        "close": close, "volume": 1, "value": 1.0}))
            c0 = close[-1]
    return pd.concat(frames, ignore_index=True), days


def forecast(bars, name, day, shift=False):
    F = ft.build(bars)
    d_idx = int(np.searchsorted(F.dates, np.datetime64(day.date())))
    if shift:  # m1: признаки строки t берутся из дня t+1
        F = ft.Fp(F.dates, F.seg, F.rv, np.vstack([F.X[1:], np.full((1, F.X.shape[1]), np.nan)]))
    D, off = pb.build_designs(F)
    r = ROWS[name]
    res = gen.run_design(D[r.design], pb.spec_of(r.lib, r.mono), 3, 0.5, d_idx, [d_idx])
    return float(off[r.design][d_idx] + r.c * res["f"][0] + (res["s2_es"] / 2 if r.bias else 0.0))


@pytest.fixture(scope="module")
def data():
    bars, days = synth()
    return bars, days[D0]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mode", ["drop", "garble"])
def test_main_forecast_unchanged_by_data_after_d(data, name, mode):
    bars, day = data
    base = forecast(bars, name, day)
    assert np.float64(forecast(damage(bars, day, mode), name, day)).tobytes() == np.float64(base).tobytes()


def test_mutation_m1_shifted_features_caught(data):
    bars, day = data
    a = forecast(bars, "b2_252_lgbm", day, shift=True)
    b = forecast(damage(bars, day, "garble"), "b2_252_lgbm", day, shift=True)
    assert a != b


def test_mutation_m4_lb_on_full_sample_caught(data, monkeypatch):
    bars, day = data
    orig = resid.harj_ab
    monkeypatch.setattr(resid, "harj_ab", lambda p, **k: orig(p, full=True))
    a = forecast(bars, "b1_xgb_c100", day)
    b = forecast(damage(bars, day, "garble"), "b1_xgb_c100", day)
    assert a != b
