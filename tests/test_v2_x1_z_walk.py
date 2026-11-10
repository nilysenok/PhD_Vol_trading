"""Этап Z (Ж-246), помесячная переоценка: g ≡ 0 ⇒ F = X1 побитно (все 5); ZB, ZB0 вне e_d = 1 и до включения — F = X1 побитно; включение ZB/ZB0 при ≥ 30 закрытых экстремальных парах; ZCp = ZC на сдвинутом z;
PIT: «закрыто + 1», λ/ω по будущим парам, z̃ по всей выборке, валидация ZB по всем парам — мутации ловятся на боевом пути walk; известный ответ: g заложен только на области ⇒ ZB восстанавливает b, a."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_feat as FE, x1_z_fit as FT, x1_z_synth as SY, x1_z_walk as W


def bits(a, b) -> bool:
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def months(x1):
    st = [p for p in W.month_starts(x1["d"]) if x1["d"][p] >= W.ACTIVATION]
    return st


def pick(x1, D):
    """Месяц переоценки (a, b), у которого день D экстремальный (e = 1): его пара «закрыто + 1» входит в пул ZB."""
    st = months(x1)
    a = next(p for p in st[3:] if D["e"][p] == 1)
    return a, st[st.index(a) + 1]


@pytest.mark.parametrize("kind", W.KINDS)
def test_g_zero_gives_x1_bitwise(kind) -> None:
    x1, D = SY.make(500, 0, p_e=0.3)
    D = dict(D, u=np.ones(len(x1)))                          # RV = X1 точно: минимум QLIKE в g = 0
    y = W.walk(x1.assign(rv=x1["F"]), D, kind)
    assert (y["active"] == 1).any() and bits(y["F"], y["X1"]) and (y["g"] == 0).all()


def test_zb_zb0_x1_outside_area_and_before_activation_and_before_inclusion() -> None:
    x1, D = SY.make(500, 1, p_e=0.15)
    for kind in ("ZB", "ZB0"):
        y = W.walk(x1, D, kind)
        e = D["e"] == 1
        assert bits(y.loc[~e, "F"], y.loc[~e, "X1"])          # вне области — побитно
        pre = y["active"] == 0
        assert (y.loc[y["d"] < W.ACTIVATION, "active"] == 0).all() and bits(y.loc[pre, "F"], y.loc[pre, "X1"])          # до включения — побитно
        assert (y.loc[y["active"] == 1, "n_ext"] >= FT.MIN_EXT).all() and (y.loc[pre & (y["d"] >= W.ACTIVATION), "n_ext"] < FT.MIN_EXT).all()
        assert (y["active"] == 1).any()


def test_za_active_from_activation_and_zb_waits_for_30_extremes() -> None:
    x1, D = SY.make(500, 1, p_e=0.15)
    za, zb = W.walk(x1, D, "ZA"), W.walk(x1, D, "ZB")
    assert za.loc[za["d"] == W.ACTIVATION, "active"].iloc[0] == 1
    assert zb.loc[zb["d"] == W.ACTIVATION, "active"].iloc[0] == 0 and zb.loc[zb["d"] == W.ACTIVATION, "n_ext"].iloc[0] < FT.MIN_EXT
    assert zb["active"].max() == 1 and za["active"].sum() > zb["active"].sum()


def test_zcp_is_zc_on_shifted_z_and_no_shift_mutation_is_caught(monkeypatch) -> None:
    x1, D = SY.make(500, 2)
    zc, zcp = W.walk(x1, D, "ZC"), W.walk(x1, D, "ZCp")
    assert not bits(zc["F"], zcp["F"])
    assert bits(W.walk(x1, dict(D, Z=FE.shift_z(D["Z"])), "ZC")["F"], zcp["F"])
    monkeypatch.setattr(FE, "shift_z", lambda Z, k=FE.SHIFT: Z)
    assert bits(W.walk(x1, D, "ZCp")["F"], zc["F"])           # мутант: ZCp ≡ ZC — тест выше падал бы


@pytest.mark.parametrize("kind", ["ZA", "ZB", "ZC"])
def test_pit_future_targets_do_not_change_forecasts_and_closed_plus_one_is_caught(kind, monkeypatch) -> None:
    x1, D = SY.make(600, 3, p_e=0.3)
    a, b = pick(x1, D)
    x2, D2 = x1.copy(), dict(D)
    x2.loc[a:, "rv"] *= 5.0
    D2["u"] = D["u"].copy()
    D2["u"][a:] *= 5.0                                         # цели пар, НЕ закрытых к D = d[a], меняются
    base, ch = W.walk(x1, D, kind), W.walk(x2, D2, kind)
    assert bits(base.loc[: b - 1, "F"], ch.loc[: b - 1, "F"]) and not bits(base["F"], ch["F"])
    monkeypatch.setattr(W, "closed_date", lambda tp, d: d)       # мутация «закрыто + 1»
    mb, mc = W.walk(x1, D, kind), W.walk(x2, D2, kind)
    assert not bits(mb.loc[: b - 1, "F"], mc.loc[: b - 1, "F"])


@pytest.mark.parametrize("kind", ["ZA", "ZC"])
def test_lambda_omega_by_future_pairs_mutation_is_caught(kind, monkeypatch) -> None:
    x1, D = SY.make(600, 4, p_e=0.3)
    a, b = months(x1)[3], months(x1)[4]
    D2 = dict(D, u=D["u"].copy())
    D2["u"][a:] = 1.0                                           # будущие цели = X1 (поправка вредна): мутант выберет ω = 0 / λ = max
    assert bits(W.walk(x1, D, kind).loc[: b - 1, "F"], W.walk(x1, D2, kind).loc[: b - 1, "F"])
    n = len(x1)
    monkeypatch.setattr(FT, "val_idx", lambda pool, v: np.minimum(np.arange(pool[-1] + 1, pool[-1] + 1 + v), n - 1))     # мутация: блок валидации — пары ПОСЛЕ D
    assert not bits(W.walk(x1, D, kind).loc[: b - 1, "F"], W.walk(x1, D2, kind).loc[: b - 1, "F"])


def test_zc_standardization_by_whole_sample_mutation_is_caught(monkeypatch) -> None:
    x1, D = SY.make(600, 5, p_e=0.3)
    a, b = months(x1)[3], months(x1)[4]
    D2 = dict(D, Z=D["Z"].copy())
    D2["Z"][b:, 1] += 50.0                                      # z дней после месяца переоценки
    assert bits(W.walk(x1, D, "ZC").loc[: b - 1, "F"], W.walk(x1, D2, "ZC").loc[: b - 1, "F"])
    monkeypatch.setattr(FE, "std_stats", lambda Z, pool: (Z.mean(0), Z.std(0)))
    assert not bits(W.walk(x1, D, "ZC").loc[: b - 1, "F"], W.walk(x1, D2, "ZC").loc[: b - 1, "F"])


def test_zb_validation_uses_extreme_pairs_and_all_pairs_mutation_is_caught(monkeypatch) -> None:
    x1, D = SY.make(700, 6, p_e=0.25)
    y = W.walk(x1, D, "ZB")
    a = months(x1)[6]
    pool = W.closed_pool(pd.DatetimeIndex(x1["t_plus"]), pd.DatetimeIndex(x1["d"]), x1["d"][a])
    ext = FT.ext_pool(pool, D["e"])
    v = FT.n_valid(len(ext), FT.V_EXT)
    fit = lambda i, lam: FT.fit_lin(D["u"][i], D["s"][i], lam)
    sc = lambda th, i, o: FT.qlike_g(D["u"][i], D["e"][i] * (th[0] + th[1] * D["s"][i]))
    lam, _ = FT.select(ext[: len(ext) - v], ext[-v:], fit, sc)
    assert y.loc[a, "lam"] == lam and (D["e"][ext[-v:]] == 1).all()
    monkeypatch.setattr(FT, "val_block_b", lambda pool_, ext_, v_: pool_[-v_:])        # мутация: валидация по всем парам
    ym = W.walk(x1, D, "ZB")
    assert not np.array_equal(y["lam"].to_numpy(), ym["lam"].to_numpy(), equal_nan=True) or not bits(y["F"], ym["F"])


def test_known_answer_zb_recovers_b_a_when_g_only_on_area() -> None:
    x1, D = SY.make(2400, 7, p_e=0.25)
    y = W.walk(x1, D, "ZB")
    last = y[y["active"] == 1].iloc[-1]
    assert abs(last["b"] - SY.TRUE["b0"]) < 0.08 and abs(last["a"] - SY.TRUE["a0"]) < 0.08
    za = W.walk(x1, D, "ZA").iloc[-1]
    assert za["omega"] < 1.0 or abs(za["b"] - SY.TRUE["b0"]) > 0.05          # ZA без области вынуждена ужимать/смещаться


def test_params_constant_within_month_and_refuses_with_few_pairs() -> None:
    x1, D = SY.make(500, 8, p_e=0.3)
    y = W.walk(x1, D, "ZA")
    a = y[y["active"] == 1]
    assert (a.groupby(a["d"].dt.to_period("M"))[["lam", "omega", "b", "a"]].nunique() == 1).all().all()
    x2, D2 = SY.make(400, 8, start="2018-03-01")
    with pytest.raises(SystemExit, match="закрытых пар"):
        W.walk(x2, D2, "ZA")
