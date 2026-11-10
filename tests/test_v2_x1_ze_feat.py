"""Лист ZE (Ж-247 п. 4), признак G_d и предусловие: G_d = ln RV_d − ln X1_d по складу M10 (бары дня d, X1 строки d). Мутации ловятся на боевом пути `gap`: RV дня t⁺ в G; X1_d со строки t⁺; G из таблицы пар (на d = 2022-04-29 пропуск);
ZEp без сдвига; известный ответ — предусловие 179 пар, корреляции дирижёра (0,693; −0,172)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_stage as SA, x1_z_feat as ZF, x1_ze_feat as FE


def store(n=40, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return dict(days=pd.bdate_range("2018-01-03", periods=n), r2=rng.random((n, 53)) + 0.1, present=np.ones((n, 53), bool))


def test_g_known_answer_rv_of_day_d_and_x1_of_row_d() -> None:
    st = store()
    d, X1 = st["days"][25:30], np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    G, bad = FE.gap(st, d, X1)
    rv = st["r2"].sum(1)
    assert bad == 0 and np.array_equal(G, np.log(rv[25:30]) - np.log(X1))


def test_future_rows_do_not_change_g_and_mutation_rv_of_t_plus_is_caught(monkeypatch) -> None:
    st, d = store(), store()["days"][25:30]
    X1 = np.linspace(1, 3, 5)
    base = FE.gap(st, d, X1)[0]
    st2 = dict(st, r2=st["r2"].copy())
    st2["r2"][26:] *= 10.0                                      # дни d + 1 и позже
    assert np.array_equal(FE.gap(st2, d[:1], X1[:1])[0], base[:1])      # G дня 25 не видит дня 26
    monkeypatch.setattr(FE, "rv_row", lambda p: p + 1)           # мутация: RV дня t⁺
    assert not np.array_equal(FE.gap(st2, d[:1], X1[:1])[0], base[:1])


def test_mutation_x1_from_row_t_plus_is_caught(monkeypatch) -> None:
    st, d = store(), store()["days"][25:30]
    X1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base = FE.gap(st, d, X1)[0]
    monkeypatch.setattr(FE, "x1_row", lambda i: i + 1)
    assert not np.allclose(FE.gap(st, d, X1)[0], base)


def test_degenerate_rv_gives_zero_and_counter() -> None:
    st = store()
    st["r2"][27] = 0.0
    G, bad = FE.gap(st, st["days"][26:29], np.ones(3))
    assert bad == 1 and G[1] == 0.0 and G[0] != 0.0


def test_shift_g_is_cyclic_252_and_mutation_without_shift_is_caught(monkeypatch) -> None:
    G = np.arange(600.0)
    s = FE.shift_g(G)
    assert s[252] == G[0] and s[0] == G[-252] and not np.array_equal(s, G)
    monkeypatch.setattr(FE, "shift_g", lambda G: np.asarray(G, float))
    assert np.array_equal(FE.shift_g(G), G)                       # мутант совпадает с G — тест выше падает на нём


def test_real_day_2022_04_29_g_from_store_and_pair_table_mutation_is_caught() -> None:
    """Реальные данные до 2026-02-27: в таблице пар нет t⁺ = 2022-04-29 (перерыв торгов), у d = 2022-04-29 e_d = 1; G из склада определён."""
    x1, st = pd.read_parquet(SA.X1_FILE), ZF.load_store()
    d, i = pd.DatetimeIndex(x1["d"]), int(np.flatnonzero(pd.DatetimeIndex(x1["d"]) == pd.Timestamp("2022-04-29"))[0])
    G, bad = FE.gap(st, d, x1["F"].to_numpy(float))
    assert x1["s"].to_numpy(int)[i] == 1 and np.isfinite(G[i]) and G[i] != 0.0 and bad == 0
    bypair = pd.Series(x1["rv"].to_numpy(float), index=pd.DatetimeIndex(x1["t_plus"])).reindex(d).to_numpy()      # мутация: RV_d из таблицы пар по t⁺ = d
    assert np.isnan(bypair[i]) and np.isfinite(bypair).sum() < len(d)
    ok = np.isfinite(bypair)
    assert np.allclose(bypair[ok], np.exp(G[ok] + np.log(x1["F"].to_numpy(float)[ok])), rtol=1e-5)      # там, где строка есть, RV_d склада = rv пары


def test_real_precondition_known_answer_conductor_numbers() -> None:
    from models_v2.s16 import x1_ze_inp as I
    pre = I.pre_numbers(SA.load())[2]
    assert pre["n"] == 179 and pre["ok"] and abs(pre["corr_G_v"] - 0.693) < 1e-3 and abs(pre["corr_G_s"] + 0.172) < 1e-3 and pre["max_abs"] <= 0.8


def test_precondition_false_when_duplicate_and_ignores_pairs_after_2021() -> None:
    n = 400
    tp = pd.bdate_range("2018-01-03", periods=n)
    rng = np.random.default_rng(1)
    v = rng.normal(0, 1, n)
    e = np.ones(n, int)
    G = 2.0 * v + 0.01 * rng.normal(0, 1, n)                      # дубль v_d
    pre = FE.precondition(G, v, rng.normal(0, 1, n), rng.normal(0, 1, n), e, tp)
    assert not pre["ok"] and pre["max_abs"] > 0.99
    tp2 = pd.bdate_range("2021-06-01", periods=n)                  # половина пар после 2021-12-31: в расчёт не входят
    assert FE.precondition(G, v, v, v, e, tp2)["n"] == int((tp2 <= FE.DEV_END).sum())
