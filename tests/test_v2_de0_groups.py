"""DE0, G3 и G4 (гребень): пороги групп и обучение — только по прошлому; мутации «пороги по всей выборке», «z энкодера года d + 1», «закрыто + 1» ловятся."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s15 import de0_groups as G, de0_ridge as RG


def _whole_sample_terciles(x, min_past=250):
    """Мутация: пороги по ВСЕЙ выборке (с будущим)."""
    q1, q2 = np.quantile(x[np.isfinite(x)], [1 / 3, 2 / 3])
    out = (x > q1).astype(int) + (x > q2).astype(int)
    out[:min_past] = -1
    return out


def test_terciles_known_answer_and_first_250_without_group():
    x = np.arange(600, dtype=float)
    lab = G.terciles_past(x)
    assert (lab[:250] == -1).all() and (lab[250:] == 2).all()                       # растущий ряд: всегда выше прошлых порогов
    y = np.concatenate([np.arange(300.0), [-5.0]])
    assert G.terciles_past(y)[-1] == 0                                              # ниже всех прошлых → нижний терциль


def test_thresholds_use_only_past_and_whole_sample_mutation_is_caught():
    rng = np.random.default_rng(1)
    x = rng.normal(size=700)
    x2 = x.copy()
    x2[500:] += 50.0                                                                # изменено только будущее относительно дня 499
    assert (G.terciles_past(x)[:500] == G.terciles_past(x2)[:500]).all()
    assert (_whole_sample_terciles(x)[:500] != _whole_sample_terciles(x2)[:500]).any()


def test_extreme_flag_uses_only_bars_before_day():
    rng = np.random.default_rng(2)
    n = 400
    store = dict(days=pd.bdate_range("2020-01-01", periods=n), r=rng.normal(0, 1, (n, 5)))
    st = G.day_stats(store, store["days"])
    f1 = G.extreme_flag(store, store["days"], st)
    store2 = dict(store, r=store["r"].copy())
    store2["r"][300:] *= 100.0
    f2 = G.extreme_flag(store2, store2["days"], G.day_stats(store2, store2["days"]))
    assert (f1[:300] == f2[:300]).all() and (f1[:250] == -1).all() and set(f1[250:]) <= {0, 1}


def test_ewma_uses_only_closed_pairs():
    dates = pd.bdate_range("2021-01-04", periods=6).to_numpy("datetime64[ns]")
    tp = np.append(dates[1:], np.datetime64("2021-02-01", "ns"))
    err = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 99.0])
    out = G.ewma_closed_err(err, tp, dates)
    assert np.isnan(out[0]) and np.allclose(out[1:], 1.0)                           # пара 5 (t⁺ в феврале) не закрыта ни в один день
    assert not np.isclose(out[-1], 99.0)


def _loader():
    days = pd.bdate_range("2018-01-01", "2020-12-31")

    def load(year, seed):
        if year > 2020:
            raise FileNotFoundError
        rng = np.random.default_rng(year * 10 + seed)
        return days, rng.normal(size=(len(days), 4)) * (1 + (year - 2018))
    return load


def test_z_distance_uses_encoder_of_year_d_and_year_plus_one_mutation_is_caught():
    ld = _loader()
    dates = pd.bdate_range("2019-03-01", periods=20)
    real = G.z_distance(dates, ld, seeds=(0, 1))
    days, z0 = ld(2019, 0)
    _, z1 = ld(2019, 1)
    p = days.get_loc(dates[5])
    ref = np.mean([np.sqrt((((z[p] - z[p - 60:p].mean(0)) / z[p - 60:p].std(0)) ** 2).sum()) for z in (z0, z1)])
    assert abs(real[5] - ref) < 1e-12
    mut = G.z_distance(dates, ld, enc_year=lambda d: d.year + 1, seeds=(0, 1))
    assert np.nanmax(np.abs(mut - real)) > 1e-3


def test_z_distance_needs_60_prior_days_and_known_day():
    ld = _loader()
    days, _ = ld(2018, 0)
    assert np.isnan(G.z_distance(days[:30], ld, seeds=(0,))).all()
    assert np.isnan(G.z_distance(pd.DatetimeIndex(["2019-06-08"]), ld, seeds=(0,))).all()   # суббота: дня нет в энкодере


def test_group_table_counts_partition_the_window():
    from models_v2.s15 import de0_core as C
    rng = np.random.default_rng(3)
    n = 800
    tp = pd.bdate_range("2018-01-03", periods=n)
    year = np.asarray(tp.year)
    rv = np.exp(rng.normal(size=n))
    f = {k: rv * np.exp(rng.normal(0, .3, n)) for k in ("K1", "K2", "A2", "cfg8")}
    lab = {"a_uroven": G.terciles_past(rng.normal(size=n), 100)}
    g = G.group_table(lab, np.arange(n), rv, f, year)
    for w in ("2018_2021", "2022_2025"):
        m = C.wmask(year, w)
        parts = g[(g["feature"] == "a_uroven") & (g["window"] == w)]["n"].sum()
        assert parts == int((m & (lab["a_uroven"] >= 0)).sum()) <= int(m.sum())


def _pit_data(n=700, seed=4):
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2018-01-03", periods=n)
    tp = d.shift(1)
    X = rng.normal(size=(n, 3))
    return d, tp, X, X @ np.array([1.0, -0.5, 0.0]) + rng.normal(0, .3, n), rng


def test_ridge_known_answer_signal_vs_noise():
    d, tp, X, y, rng = _pit_data()
    assert RG.pit_oos(lambda Y: X, y, d, tp)["r2_oos"].min() > 0.8
    assert abs(RG.pit_oos(lambda Y: X, rng.normal(size=len(d)), d, tp)["r2_oos"].mean()) < 0.1


def test_ridge_training_uses_only_closed_pairs_and_leak_mutation_is_caught(monkeypatch):
    d, tp, X, y, _ = _pit_data()
    pred, _ = RG.pit_predict(lambda Y: X, y, d, tp)
    sel = np.flatnonzero((d.year == 2019) & (d.month == 6))
    y2 = y.copy()
    y2[sel[0]:] += 100.0                                                            # будущее относительно первого дня месяца
    pred2, _ = RG.pit_predict(lambda Y: X, y2, d, tp)
    assert np.allclose(pred[sel], pred2[sel], equal_nan=True)
    monkeypatch.setattr(RG, "train_rows", lambda tp_, d0, first: np.arange(first + 5))   # мутация: пары, не закрытые к d0
    pred3, _ = RG.pit_predict(lambda Y: X, y2, d, tp)
    assert not np.allclose(pred[sel], pred3[sel], equal_nan=True)
