"""Ж-212: PIT-примитивы и главный тест обрезки. Прошлые ошибки/расхождения — только по закрытым прогнозам (target_date <= d).
Мутация «закрыто + 1» обязана ловиться."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s8 import hs_pit, hs_table


def grid(n=40):
    d = pd.bdate_range("2020-01-01", periods=n).to_numpy("datetime64[ns]")
    return d[:-1], d[1:]          # d_k, target_date_k = d_{k+1}


def test_closed_count_known_answer():
    d, td = grid()
    assert (hs_pit.closed_count(d, td) == np.arange(len(d))).all()      # на d_k закрыты строки 0..k−1
    assert (hs_pit.closed_count(d, td, extra=1)[:3] == [1, 2, 3]).all()  # мутация сдвигает на одну


def test_past_mean_and_last_known_answer():
    d, td = grid()
    x = np.arange(len(d), dtype=float)
    pm = hs_pit.past_mean(x, d, td, window=5, min_obs=5)
    assert np.isnan(pm[:5]).all()
    assert pm[10] == pytest.approx(np.mean(x[5:10]))        # строки 5..9 закрыты на d_10
    assert hs_pit.past_last(x, d, td)[10] == 9.0 and np.isnan(hs_pit.past_last(x, d, td)[0])


def test_truncation_future_corruption_does_not_change_past_fields():
    """Главный тест обрезки: порча всех строк, не закрытых на d_k, не меняет значение на d_k."""
    d, td = grid()
    rng = np.random.default_rng(1)
    x = rng.normal(size=len(d))
    base = hs_pit.past_mean(x, d, td, 22)
    for k in (30, 20, 8):
        bad = hs_pit.truncate_future(x, d, td, k)
        bad[int(hs_pit.closed_count(d, td)[k]):] = 1e9
        assert hs_pit.past_mean(bad, d, td, 22)[k] == base[k]
        assert hs_pit.past_last(bad, d, td)[k] == hs_pit.past_last(x, d, td)[k]


def test_mutation_closed_plus_one_is_caught():
    d, td = grid()
    x = np.arange(len(d), dtype=float)
    k = 30
    bad = x.copy()
    bad[k] = 1e9                                              # строка k закрывается только на d_{k+1}
    assert hs_pit.past_mean(bad, d, td, 22)[k] == hs_pit.past_mean(x, d, td, 22)[k]
    assert hs_pit.past_mean(bad, d, td, 22, extra=1)[k] != hs_pit.past_mean(x, d, td, 22, extra=1)[k]   # мутация видна
    assert hs_pit.past_last(bad, d, td, extra=1)[k] == 1e9


def test_nan_in_series_and_empty_history():
    d, td = grid()
    x = np.full(len(d), np.nan)
    assert np.isnan(hs_pit.past_mean(x, d, td, 22)).all()
    x[3] = 1.0
    assert np.isfinite(hs_pit.past_mean(x, d, td, 22, min_obs=1)[10]) and np.isnan(hs_pit.past_mean(x, d, td, 22, min_obs=2)[10])


def frame(n=60):
    d, td = grid(n + 1)
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"d": d, "target_date": td, "rv": np.exp(rng.normal(size=n)), "HAR": np.exp(rng.normal(size=n)), "XJ": np.exp(rng.normal(size=n)), "LF": np.exp(rng.normal(size=n)),
                       "st_g4": False, "st_rv": False})
    return hs_table.base_columns(df)


def test_table_pit_columns_ignore_unclosed_rows_and_mutation_caught():
    o = frame()
    ref = hs_table.pit_columns(o)
    k = 40
    bad = o.copy()
    bad.loc[k:, ["e_HAR", "e_XJ", "e_LF", "dq_HAR_XJ", "dq_HAR_LF"]] = 1e6          # строки k.. не закрыты на d_k
    cols = [c for c in ref.columns if c.startswith("past_")]
    got = hs_table.pit_columns(bad)
    assert np.array_equal(got.loc[k, cols].to_numpy(float), ref.loc[k, cols].to_numpy(float), equal_nan=True)
    assert ref.loc[k, "past_eH_last"] == o.loc[k - 1, "e_HAR"]
    mut = hs_table.pit_columns(bad, extra=1)
    assert not np.allclose(mut.loc[k, cols].to_numpy(float), ref.loc[k, cols].to_numpy(float), equal_nan=True)


def test_base_columns_known_answer_and_sign():
    o = frame()
    r = o.iloc[3]
    assert r["e_HAR"] == pytest.approx(np.log(r["rv"] / r["HAR"]))
    assert r["q_HAR"] == pytest.approx(r["rv"] / r["HAR"] - np.log(r["rv"] / r["HAR"]) - 1)
    assert r["dq_HAR_XJ"] == pytest.approx(r["q_HAR"] - r["q_XJ"])           # плюс — бустинг лучше
    assert r["delta_HX"] == pytest.approx(np.log(r["HAR"] / r["XJ"]))
