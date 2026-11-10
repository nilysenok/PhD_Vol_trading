"""ZD (Ж-246), PIT: модель D на месяц учится только на парах t⁺ ≤ D (мутация «закрыто + 1» ловится); стандартизация z̃ — по пулу (мутация «вся выборка» ловится); признаки x_d, v_d — только день d и прошлое (мутация: бары дня t⁺);
контроль (г) со сдвигом 252 ≠ (а) на боевых данных. Синтетика — известный ответ: подмена будущего не меняет прогноз месяца."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_walk as W, x1_zd_core as C, x1_zd_stage as S


def synth(n: int = 330, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    d = pd.bdate_range("2018-01-03", periods=n + 1)
    Z = rng.normal(size=(n, 4))
    Z[:, 0] = (rng.random(n) < 0.2).astype(float)
    D = 0.5 * Z[:, 2] + rng.normal(size=n)
    return Z, D, d[:n], d[1:n + 1]


def first_month(d) -> tuple:
    starts = [p for p in W.month_starts(d) if d[p] >= W.ACTIVATION]
    return starts[0], starts[1]


def pred_first_month(Z, D, d, tp) -> pd.DataFrame:
    a, b = first_month(d)
    # (г) — циклический сдвиг по всему ряду: у него строки с конца ряда попадают в начало по построению, поэтому PIT-тест берёт (а), (б), (в)
    return C.walk(Z, D, d, tp).query("idx >= %d and idx < %d" % (a, b))[["idx", "pa", "pb", "pc"]].reset_index(drop=True)


def test_future_pair_does_not_change_month(monkeypatch) -> None:
    Z, D, d, tp = synth()
    a, _ = first_month(d)
    D2 = D.copy()
    D2[a] += 50.0                       # пара дня d[a]: t⁺ = d[a+1] > D → не в пуле
    pd.testing.assert_frame_equal(pred_first_month(Z, D, d, tp), pred_first_month(Z, D2, d, tp))
    monkeypatch.setattr(W, "closed_date", lambda tp_, d_: pd.DatetimeIndex(tp_) - pd.offsets.BDay(1))     # мутация «закрыто + 1»
    assert not pred_first_month(Z, D, d, tp).equals(pred_first_month(Z, D2, d, tp))


def test_standardization_pool_only_and_whole_sample_mutation(monkeypatch) -> None:
    Z, D, d, tp = synth()
    _, b = first_month(d)
    Z2 = Z.copy()
    Z2[b + 5] += 7.0                    # строка вне пула и вне месяца
    pd.testing.assert_frame_equal(pred_first_month(Z, D, d, tp), pred_first_month(Z2, D, d, tp))
    monkeypatch.setattr(C, "std_rows", lambda pool, n: np.arange(n))                                      # мутация: вся выборка
    assert not pred_first_month(Z, D, d, tp).equals(pred_first_month(Z2, D, d, tp))


def make_store(n: int = 60, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    return dict(days=pd.bdate_range("2018-01-03", periods=n), r2=rng.random((n, 53)) + 0.1, present=np.ones((n, 53), bool))


def test_features_use_only_day_d_and_past(monkeypatch) -> None:
    st = make_store()
    d, e, s = st["days"][30:34], np.zeros(4), np.zeros(4)
    base = C.zfeatures(st, d, e, s)
    st2 = dict(st, r2=st["r2"].copy())
    st2["r2"][31:] *= 9.0               # бары дня d+1 и позже
    st2["r2"][31] += 3.0
    assert np.array_equal(base[0], C.zfeatures(st2, d, e, s)[0])
    monkeypatch.setattr(C, "zrow", lambda p: p + 1)
    assert not np.array_equal(base[0], C.zfeatures(st2, d, e, s)[0])


def test_features_known_answer() -> None:
    r2, pres = np.ones(53), np.ones(53, bool)
    r2[10] = 4.0                        # RV = 56, n = 53, max = 4: x = ln 4 − ln(56/53)
    x, v = C.day_xv(r2, pres, np.full(20, 28.0))
    assert x == pytest.approx(np.log(4.0) - np.log(56.0 / 53.0)) and v == pytest.approx(np.log(56.0 / 28.0))
    assert C.day_xv(np.zeros(53), pres, np.full(20, 28.0)) == (0.0, 0.0)


def test_shift_control_differs_from_z_on_real_data() -> None:
    inp = S.load()
    Z = inp["Z"]
    assert not np.array_equal(np.roll(Z, C.LAG, axis=0), Z) and Z.shape == (2018, 4)
    t, _ = S.compute(inp)
    r = t["PRED"].query("y == 'Y2' and period == 'all'").iloc[0]
    assert r.auc_g != r.auc_a and 0.3 < r.auc_g < 0.7
