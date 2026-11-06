"""A8 (Ж-206, волна A): прогноз LightGBM-аналога Б-2 (base, S1, S2) на d побитно не меняется при порче/обрезке баров после 18:50 дня d;
режим PIT не использует будущее. Мутации: признаки +1 день (№ 1); пары с t+1 = d+1; режим с квантилем, включающим день d (№ 22)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models_v2.s2 import features as ft  # noqa: E402
from models_v2.s4 import design, fit, runner  # noqa: E402
from models_v2.s5 import pit  # noqa: E402
from models_v2.s6 import diag, engine6  # noqa: E402

HP = fit.HP(depth=3, min_child=20.0, max_trees=40)
VARIANTS = ("base", "S1", "S2")
N_DAYS, DAY_NO = 800, 760


def synth(n_days=N_DAYS, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-06", periods=n_days)
    lvl = np.exp(np.cumsum(rng.normal(0, 0.1, n_days)) * 0.3)
    frames, c0 = [], 100.0
    for k, day in enumerate(days):
        for start, m in (("10:00", 53), ("18:50", 4)):
            t0 = pd.Timestamp(str(day.date()) + " " + start)
            close = c0 * np.exp(np.cumsum(rng.normal(0, 0.001 * lvl[k], m)))
            opn = np.concatenate([[c0], close[:-1]])
            frames.append(pd.DataFrame({"begin": [t0 + pd.Timedelta(minutes=10 * j) for j in range(m)], "open": opn, "high": np.maximum(opn, close),
                                        "low": np.minimum(opn, close), "close": close, "volume": 1, "value": 1.0}))
            c0 = close[-1]
    return pd.concat(frames, ignore_index=True), days


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


def prod(F, i, variant):
    dz = design.make(F, "m")
    m = engine6.fit_var(dz.X, dz.y, runner.pairs_upto(dz, F, i), HP, 42, variant)
    return float(m["predict"](dz.X[[i]])[0])


def shifted(F, i, variant):  # мутация № 1: признаки строки t берутся из дня t+1
    X = np.vstack([F.X[1:], np.full((1, F.X.shape[1]), np.nan)])
    return prod(ft.Fp(F.dates, F.seg, F.rv, X), i, variant)


def pairs_plus_one(F, i, variant):  # мутация: обучающие пары с t+1 = d+1
    dz = design.make(F, "m")
    m = engine6.fit_var(dz.X, dz.y, runner.pairs_upto(dz, F, i + 1), HP, 42, variant)
    return float(m["predict"](dz.X[[i]])[0])


def result(bars, d, variant, fc):
    F = ft.build(bars)
    i = int(np.flatnonzero(F.dates == np.datetime64(d))[0])
    return np.float64(fc(F, i, variant)).tobytes()


def invariant(bars, d, variant, fc):
    ref = result(bars, d, variant, fc)
    try:
        return all(result(damage(bars, d, m), d, variant, fc) == ref for m in ("garbage", "drop"))
    except (IndexError, ValueError):
        return False


@pytest.fixture(scope="module")
def sb():
    return synth()


@pytest.mark.parametrize("variant", VARIANTS)
def test_bitwise_invariance_and_determinism(sb, variant):
    bars, days = sb
    d = days[DAY_NO]
    assert invariant(bars, d, variant, prod)
    assert result(bars, d, variant, prod) == result(bars, d, variant, prod)


@pytest.mark.parametrize("variant", VARIANTS)
def test_mutation_features_plus_one_day_caught(sb, variant):
    bars, days = sb
    assert not invariant(bars, days[DAY_NO], variant, shifted)


@pytest.mark.parametrize("variant", VARIANTS)
def test_mutation_pairs_with_target_d_plus_1_caught(sb, variant):
    bars, days = sb
    assert not invariant(bars, days[DAY_NO], variant, pairs_plus_one)


def test_predictions_regime_invariant_to_future_and_mutation_caught():
    rng = np.random.default_rng(11)
    x = np.exp(rng.normal(0, 0.5, 700))
    stress = rng.random(700) < 0.2
    i = 600
    ref = diag.regime_pit(x, stress)[:i + 1]
    xs = x.copy()
    xs[i + 1:] = 1e3 * rng.random(len(x) - i - 1)
    assert (diag.regime_pit(xs, stress)[:i + 1] == ref).all()           # будущее не влияет
    manual = [sum(int(x[k] > np.nanquantile(x[k - pit.WIN:k], q)) for q in (.2, .4, .6, .8)) for k in range(300, 330)]
    assert [int(r.split("|")[0][1:]) for r in diag.regime_pit(x, stress)[300:330]] == manual   # ровно 252 предыдущих, без дня d
    mut = diag.regime_pit(x, stress, include_current=True)               # мутация № 22
    assert (mut[pit.WIN:] != diag.regime_pit(x, stress)[pit.WIN:]).any()
