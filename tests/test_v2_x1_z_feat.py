"""Этап Z (Ж-246), признаки затвора: x_d и v_d — известный ответ; PIT (бары/дни после d не влияют) и мутации «x_d с баром дня t⁺», «v_d с днём d + 1»; стандартизация z̃ по пулу (мутация «по всей выборке»); ZCp — циклический сдвиг (мутация «без сдвига»)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_feat as FE


def store(nd=60, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-01-01", periods=nd)
    r2 = rng.gamma(2.0, 1.0, (nd, 53))
    present = rng.random((nd, 53)) < 0.97
    present[:, :40] = True
    return dict(days=days, r2=r2, present=present)


def test_x_and_v_known_answer() -> None:
    st = store()
    d = st["days"][30:33]
    x, v, bad = FE.x_v(st, d)
    assert bad == 0
    for i, p in enumerate(range(30, 33)):
        r = np.where(st["present"][p], st["r2"][p], 0.0)
        n = st["present"][p].sum()
        want_x = np.log(r.max()) - np.log(r.sum() / n)
        h = np.array([np.where(st["present"][q], st["r2"][q], 0.0).sum() for q in range(p - 20, p)])
        want_v = np.log(r.sum()) - np.log(h.mean())
        assert abs(x[i] - want_x) < 1e-12 and abs(v[i] - want_v) < 1e-12


def test_future_bars_and_days_do_not_change_features_and_mutations_are_caught(monkeypatch) -> None:
    st = store()
    d = st["days"][30:33]
    x0, v0, _ = FE.x_v(st, d)
    st2 = {**st, "r2": st["r2"].copy()}
    st2["r2"][31:] *= 7.0                                       # бары дней d + 1 … (d = days[30])
    x1, v1, _ = FE.x_v(st2, d[:1])
    assert x1[0] == x0[0] and v1[0] == v0[0]                    # день d = days[30]: будущее не влияет
    monkeypatch.setattr(FE, "feat_row", lambda i: i + 1)        # мутация: бары дня t⁺ в x_d
    xm0, _, _ = FE.x_v(st, d[:1])
    xm1, _, _ = FE.x_v(st2, d[:1])
    assert xm0[0] != xm1[0] and xm0[0] != x0[0]
    monkeypatch.undo()
    monkeypatch.setattr(FE, "hist_rows", lambda p, win=FE.WIN: np.arange(p - win + 2, p + 2))   # мутация: окно v_d доходит до дня d + 1
    _, vm0, _ = FE.x_v(st, d[:1])
    _, vm1, _ = FE.x_v(st2, d[:1])
    assert vm0[0] != vm1[0] and vm0[0] != v0[0]


def test_degenerate_day_is_counted_and_zeroed() -> None:
    st = store()
    st["r2"][35] = 0.0
    x, v, bad = FE.x_v(st, st["days"][34:37])
    assert bad == 1 and x[1] == 0.0 and v[1] == 0.0 and x[0] != 0.0


def test_standardize_uses_only_pool_and_whole_sample_mutation_is_caught(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(300, 4))
    pool = np.arange(150)
    Z2 = Z.copy()
    Z2[150:] += 100.0                                           # будущее (вне пула)
    assert np.array_equal(FE.standardize(Z, pool)[:150], FE.standardize(Z2, pool)[:150])
    zt = FE.standardize(Z, pool)[pool]
    assert np.allclose(zt.mean(0), 0, atol=1e-12) and np.allclose(zt.std(0), 1, atol=1e-12)
    monkeypatch.setattr(FE, "std_stats", lambda Z_, pool_: (Z_.mean(0), Z_.std(0)))
    assert not np.array_equal(FE.standardize(Z, pool)[:150], FE.standardize(Z2, pool)[:150])


def test_shift_is_cyclic_252_and_no_shift_mutation_is_caught(monkeypatch) -> None:
    Z = np.arange(600 * 4, dtype=float).reshape(600, 4)
    S = FE.shift_z(Z)
    assert np.array_equal(S[252:], Z[:-252]) and np.array_equal(S[:252], Z[-252:]) and not np.array_equal(S, Z)
    monkeypatch.setattr(FE, "shift_z", lambda Z_, k=252: Z_)
    assert np.array_equal(FE.shift_z(Z), Z)
