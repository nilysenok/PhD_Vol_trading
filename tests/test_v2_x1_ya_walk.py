"""Y-A (Ж-243), помесячная переоценка: a = 0 ⇒ F = X1 побитно; до 2018-07-02 F = X1 побитно (боевой ряд); отказ при < 120 закрытых парах; PIT («закрыто + 1» ловится на боевом пути walk);
Y2 с κ ≠ 0 ловится; параметры постоянны внутри месяца; e_d берётся из X1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_ya_fit as FT, x1_ya_synth as SY, x1_ya_walk as W


def test_activation_and_bitwise_before_inclusion_on_synthetic() -> None:
    x1, ft = SY.make(400, 0)
    for mode in ("Y1", "Y2"):
        y = W.walk(x1, ft, mode)
        pre = y["active"] == 0
        assert pre.sum() > 100 and (y.loc[pre, "d"] < W.ACTIVATION).all() and (y.loc[~pre, "d"] >= W.ACTIVATION).all()
        assert np.array_equal(y.loc[pre, "F"].to_numpy(), y.loc[pre, "X1"].to_numpy())                 # побитно
        assert (y.loc[pre, "g"] == 0).all() and y.loc[pre, ["lam", "a_n", "a_e"]].isna().all().all()
        assert not np.array_equal(y.loc[~pre, "F"].to_numpy(), y.loc[~pre, "X1"].to_numpy())


def test_zero_amplitudes_give_x1_bitwise() -> None:
    x1, ft = SY.make(400, 0)
    ft0 = {**ft, "s": np.zeros(len(x1))}                    # s_d ≡ 0 ⇒ a остаётся 0 ⇒ g = 0
    for mode in ("Y1", "Y2"):
        y = W.walk(x1, ft0, mode)
        assert (y["active"] == 1).any() and np.array_equal(y["F"].to_numpy(), y["X1"].to_numpy()) and (y["a_n"].dropna() == 0).all()
    th = np.array([0.0, 0.0, 0.5])
    assert (FT._g(th, np.ones(5), np.arange(5.0), np.arange(5.0), "Y1")[0] == 0).all()


def test_refuses_with_fewer_than_120_closed_pairs() -> None:
    x1, ft = SY.make(400, 0, start="2018-03-01")            # к 2018-07-02 закрыто ~86 пар
    with pytest.raises(SystemExit, match="закрытых пар"):
        W.walk(x1, ft, "Y1")
    x2, _ = SY.make(400, 0)
    assert W.check_activation(x2["d"], x2["t_plus"]) >= 120


def test_params_constant_within_month_and_applied_from_first_day() -> None:
    x1, ft = SY.make(500, 1)
    y = W.walk(x1, ft, "Y1")
    a = y[y["active"] == 1]
    assert (a.groupby(a["d"].dt.to_period("M"))[["a_n", "a_e", "kappa", "lam"]].nunique() == 1).all().all()
    assert y.loc[y["d"] == W.ACTIVATION, "lam"].notna().all()


def test_pit_targets_from_day_d_on_do_not_change_forecasts_and_closed_plus_one_mutation_is_caught(monkeypatch) -> None:
    x1, ft = SY.make(500, 2)
    base = W.walk(x1, ft, "Y1")
    starts = [p for p in W.month_starts(x1["d"]) if x1["d"][p] >= W.ACTIVATION]
    a = starts[3]                                            # D = d[a]; закрыты строки j < a (t⁺_{a−1} = d[a])
    b = starts[4]
    x2 = x1.copy()
    x2.loc[a:, "rv"] = x2.loc[a:, "rv"] * 5.0                # цели пар, НЕ закрытых к D (и позже), меняются
    ch = W.walk(x2, ft, "Y1")
    assert np.array_equal(base.loc[:b - 1, "F"].to_numpy(), ch.loc[:b - 1, "F"].to_numpy())           # прогнозы блока D … конец месяца прежние
    assert not np.array_equal(base["F"].to_numpy(), ch["F"].to_numpy())                                # дальше — меняются (законно: пары закрылись)
    monkeypatch.setattr(W, "closed_date", lambda tp, d: d)   # мутация «закрыто + 1»: пара видна в день прогноза
    mb, mc = W.walk(x1, ft, "Y1"), W.walk(x2, ft, "Y1")
    assert not np.array_equal(mb.loc[:b - 1, "F"].to_numpy(), mc.loc[:b - 1, "F"].to_numpy())


def test_y2_has_no_decay_and_kappa_nonzero_mutation_is_caught(monkeypatch) -> None:
    x1, ft = SY.make(500, 3)
    e = x1["s"].to_numpy(float)

    def is_undamped(y) -> bool:
        a = y[y["active"] == 1].index
        want = (y.loc[a, "a_n"] * (1 - e[a]) + y.loc[a, "a_e"] * e[a]) * np.asarray(ft["s"])[a]
        return bool(np.allclose(y.loc[a, "g"], want, rtol=1e-12, atol=0)) and bool((y.loc[a, "kappa"] == 0).all())

    assert is_undamped(W.walk(x1, ft, "Y2"))
    assert not is_undamped(W.walk(x1, ft, "Y1"))
    orig = FT._g
    monkeypatch.setattr(FT, "_g", lambda th, e_, s_, dl, mode: orig(np.append(th[:2], 0.5), e_, s_, dl, "Y1"))        # мутация: Y2 с κ = 0,5
    assert not is_undamped(W.walk(x1, ft, "Y2"))


def test_uses_e_state_from_x1(monkeypatch) -> None:
    from models_v2.s16 import x1_ya_feat as FE
    x1, ft = SY.make(400, 4)
    base = W.walk(x1, ft, "Y2")
    monkeypatch.setattr(FE, "e_state", lambda x: np.zeros(len(x), int))                  # мутация: e_d не из X1
    assert not np.allclose(W.walk(x1, ft, "Y2")["F"], base["F"], rtol=1e-12, atol=0)


def test_real_series_before_inclusion_bitwise_and_124_pairs() -> None:
    from models_v2.s16 import x1_ya_stage as S
    inp = S.load()
    assert W.check_activation(inp["d"], inp["tp"]) == 124
    for mode in ("Y1", "Y2"):
        y = W.walk(inp["x1"], inp["ft"], mode)
        pre = (y["active"] == 0).to_numpy()
        assert pre.sum() == 124 and np.array_equal(y["F"].to_numpy()[pre], inp["x1"]["F"].to_numpy()[pre]) and np.isfinite(y["F"]).all() and (y["F"] > 0).all()
        assert (y["d"][~pre].min() == W.ACTIVATION) and y["a_n"].notna().sum() == (~pre).sum()
    assert set(pd.DatetimeIndex(y["d"][~pre]).to_period("M")) >= {pd.Period("2018-07"), pd.Period("2026-02")}
