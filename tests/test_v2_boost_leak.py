"""G17: прогноз бустинга на d не меняется при порче/обрезке данных после 18:50 дня d (побитно); детерминизм;
мутации: признаки +1 день; HAR-признак из подгонки на всей выборке; блок остановки без эмбарго."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2 import daily, har, prov  # noqa: E402
from models_v2.s2 import boost  # noqa: E402
from models_v2.s2 import features as ft  # noqa: E402

SPECS = [boost.Spec("xgb", "mse", False, False), boost.Spec("lgbm", "hub", True, True),
         boost.Spec("xgb", "hub", True, False), boost.Spec("lgbm", "mse", False, True)]
DEPTH, DELTA = 3, 0.5


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


def prod(F, spec, i):
    return boost.forecast_at(F, spec, DEPTH, DELTA, i)


def prod_monthly(F, spec, i):
    return boost.forecast_at(F, spec, DEPTH, DELTA, i, daily=False)


def shifted(F, spec, i):  # мутация: признаки строки t берутся из дня t+1
    X = np.vstack([F.X[1:], np.full((1, F.X.shape[1]), np.nan)])
    return prod(ft.Fp(F.dates, F.seg, F.rv, X), spec, i)


def har_full(F, spec, i):  # мутация (Б2): HAR-признак — одна подгонка на всей выборке
    p = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    Xt, yt = har.pairs(p, len(F.rv) - 1, har.COLS_HARJ)
    beta, s2, _ = har.ols(Xt, yt)
    X = F.X.copy()
    X[:, ft.I_HAR] = np.where(np.isfinite(F.X[:, ft.I_HAR]), beta[0] + F.X[:, :4] @ beta[1:] + s2 / 2, np.nan)
    return prod(ft.Fp(F.dates, F.seg, F.rv, X), spec, i)


def result(bars, d, spec, fc):
    F = ft.build(bars)
    i = int(np.flatnonzero(F.dates == np.datetime64(d))[0])
    return np.float64(fc(F, spec, i)).tobytes()


def invariant(bars, d, spec, fc):
    ref = result(bars, d, spec, fc)
    try:
        return all(result(damage(bars, d, m), d, spec, fc) == ref for m in ("garbage", "drop"))
    except (IndexError, ValueError):
        return False


@pytest.fixture(scope="module")
def sb():
    return synth()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("day_no", [840, 860])  # разные месяцы, не первые дни месяца
def test_synthetic_bitwise_invariance_and_determinism(sb, spec, day_no):
    bars, days = sb
    d = days[day_no]
    assert invariant(bars, d, spec, prod)
    assert result(bars, d, spec, prod) == result(bars, d, spec, prod)  # детерминизм


@pytest.mark.parametrize("spec", [SPECS[0], SPECS[1]], ids=lambda s: s.name)
def test_synthetic_monthly_mode_and_retrain_day_itself(sb, spec):
    bars, days = sb
    m = days.to_series().groupby(days.to_period("M")).first()
    d = [x for x in m if x >= days[840]][0]  # первый годный день месяца — день переобучения
    assert invariant(bars, d, spec, prod_monthly) and invariant(bars, days[850], spec, prod_monthly)


def test_mutation_features_plus_one_day_caught(sb):
    bars, days = sb
    for spec in (SPECS[0], SPECS[1]):
        assert not invariant(bars, days[850], spec, shifted)


def test_mutation_har_feature_full_sample_caught(sb):
    bars, days = sb
    assert not invariant(bars, days[850], SPECS[1], har_full)


def test_embargo_structure_and_mutation_caught():
    t = np.arange(30, 700)
    tr, es = boost.split_pairs(t)
    assert len(es) == boost.ES_N and tr.max() + 1 < es.min() - 0 and es.min() - tr.max() == boost.EMBARGO + 1
    tr_bad, es_bad = t[:-boost.ES_N], t[-boost.ES_N:]  # мутация: без эмбарго
    assert not (tr_bad.max() + 1 < es_bad.min())  # цель последней обучающей пары попадает в день признаков ES


def test_split_refuses_short_window():
    with pytest.raises(ValueError):
        boost.split_pairs(np.arange(300))


@pytest.mark.parametrize("d", ["2020-04-01", "2023-07-20"])
@pytest.mark.parametrize("spec", [SPECS[1], SPECS[2]], ids=lambda s: s.name)
def test_real_data_bitwise_invariance(d, spec):
    bars = pd.read_parquet(prov.BARS)
    assert invariant(bars, pd.Timestamp(d), spec, prod)


def test_real_data_mutations_caught():
    bars = pd.read_parquet(prov.BARS)
    d = pd.Timestamp("2023-07-20")
    assert not invariant(bars, d, SPECS[1], shifted)
    assert not invariant(bars, d, SPECS[1], har_full)


def test_main_session_boundary_bar_1850_not_used(sb):
    bars, days = sb
    d = days[850]
    b = bars.copy()
    m = b["begin"] == pd.Timestamp(str(d.date()) + " 18:50")
    for c in ("open", "high", "low", "close"):
        b.loc[m, c] = 777.0
    assert result(b, d, SPECS[0], prod) == result(bars, d, SPECS[0], prod)
    saved = daily.T_TO
    try:
        daily.T_TO = "18:50"
        assert result(b, d, SPECS[0], prod) != result(bars, d, SPECS[0], prod)
    finally:
        daily.T_TO = saved
