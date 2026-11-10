"""Лист ZE (Ж-247 п. 4), обучение и помесячная переоценка: g ≡ 0 ⇒ F = X1 побитно; c ≡ 0 ⇒ формула и потеря ZB побитно; вне e_d = 1 и до включения F = X1; c ∈ [0, 1];
мутации на боевом пути `walk`: «закрыто + 1», валидация по всем парам, ZEp без сдвига, c без границы; известный ответ: заложенные b, a, c на области восстанавливаются."""
from __future__ import annotations

import numpy as np
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_fit as FT, x1_z_synth as SY, x1_z_walk as ZW, x1_ze_feat as FE, x1_ze_fit as FZ, x1_ze_synth as SZ, x1_ze_walk as W


def bits(a, b) -> bool:
    return np.array_equal(np.asarray(a, float), np.asarray(b, float))


def months(x1):
    return [p for p in W.month_starts(x1["d"]) if x1["d"][p] >= W.ACTIVATION]


@pytest.mark.parametrize("kind", W.KINDS)
def test_g_zero_gives_x1_bitwise(kind) -> None:
    x1, D = SZ.make(500, 0)
    D = dict(D, u=np.ones(len(x1)))                          # RV = X1 точно: минимум QLIKE в g = 0
    y = W.walk(x1.assign(rv=x1["F"]), D, kind)
    assert (y["active"] == 1).any() and bits(y["F"], y["X1"]) and (y["g"] == 0).all()


@pytest.mark.parametrize("kind", W.KINDS)
def test_outside_area_and_before_activation_and_before_inclusion_f_equals_x1(kind) -> None:
    x1, D = SZ.make(500, 1, p_e=0.15)
    y = W.walk(x1, D, kind)
    out = (D["e"] == 0) | (y["d"] < W.ACTIVATION).to_numpy() | (y["active"] == 0).to_numpy()
    assert bits(y["F"][out], y["X1"][out]) and (y["active"][y["d"] < W.ACTIVATION] == 0).all() and (y["active"] == 1).any()
    sp = SZ.make(500, 2, p_e=0.02)                             # почти нет экстремальных пар: < 30 закрытых ⇒ не включена
    z = W.walk(sp[0], sp[1], kind)
    assert (z["active"] == 0).all() and bits(z["F"], z["X1"]) and z["n_ext"].max() < FT.MIN_EXT


def test_c_zero_loss_and_gradient_equal_zb_bitwise_and_fit_equals_zb_formula() -> None:
    x1, D = SZ.make(400, 3)
    e = D["e"] == 1
    u, s, G = D["u"][e], D["s"][e], D["G"][e]
    for th in ((0.1, 0.2), (-0.3, 0.05), (0.0, 0.0)):
        v3, g3 = FZ.loss_grad(np.array([th[0], th[1], 0.0]), u, s, G, 0.01)
        v2, g2 = FT.lin_loss_grad(np.array(th), u, s, 0.01)
        assert v3 == v2 and bits(g3[:2], g2)                 # та же функция потерь побитно
    for lam in FT.LAMS:
        t3, t2 = FZ.fit_ze(u, s, G, lam, c_bounds=(0.0, 0.0)), FT.fit_lin(u, s, lam)
        assert t3[2] == 0.0 and np.allclose(t3[:2], t2, atol=1e-5, rtol=0)      # формула и потеря — побитно (выше); путь L-BFGS-B при 3 и 2 переменных различается на ~1e-7


def test_walk_with_c_fixed_zero_equals_zb_walk(monkeypatch) -> None:
    x1, D = SZ.make(500, 4)
    zb = ZW.walk(x1, dict(D, Z=SY.make(500, 4)[1]["Z"]), "ZB")
    monkeypatch.setattr(FZ, "C_BOUNDS", (0.0, 0.0))
    ze = W.walk(x1, D, "ZE")
    assert (ze["c"].dropna() == 0).all() and np.allclose(ze["F"], zb["F"], rtol=1e-5, atol=0) and (ze["active"] == zb["active"]).all()


def test_known_answer_recovers_b_a_c_on_area() -> None:
    x1, D = SZ.make(4000, 5, p_e=0.5, shape=8.0)
    y = W.walk(x1, D, "ZE")
    last = y[y["active"] == 1].iloc[-1]
    assert abs(last["b"] - SZ.TRUE["b0"]) < 0.08 and abs(last["a"] - SZ.TRUE["a0"]) < 0.08 and abs(last["c"] - SZ.TRUE["c0"]) < 0.12
    ctrl = W.walk(x1, D, "ZEp")                                # контроль со сдвинутым G c не получает того же
    assert ctrl[ctrl["active"] == 1].iloc[-1]["c"] < last["c"] - 0.2


@pytest.mark.parametrize("c0,bound", [(-0.6, 0.0), (2.0, 1.0)])
def test_c_bound_zero_one_and_mutation_without_bound_is_caught(monkeypatch, c0, bound) -> None:
    x1, D = SZ.make(3000, 6, truth=dict(b0=0.1, a0=0.3, c0=c0), p_e=0.5, shape=8.0)
    honest = W.walk(x1, D, "ZE")["c"].dropna()
    assert honest.min() >= 0.0 and honest.max() <= 1.0 and abs(honest.iloc[-1] - bound) < 1e-6
    monkeypatch.setattr(FZ, "C_BOUNDS", (-5.0, 5.0))           # мутация: без границы [0, 1]
    mut = W.walk(x1, D, "ZE")["c"].dropna()
    assert mut.min() < 0.0 or mut.max() > 1.0


def test_zep_uses_g_shifted_252_and_mutation_without_shift_is_caught(monkeypatch) -> None:
    x1, D = SZ.make(900, 7, p_e=0.4)
    ze, zp = W.walk(x1, D, "ZE"), W.walk(x1, D, "ZEp")
    assert bits(zp["G"], np.roll(D["G"], 252)) and not np.allclose(ze["F"], zp["F"])
    monkeypatch.setattr(FE, "shift_g", lambda G: np.asarray(G, float))          # мутация: ZEp без сдвига
    assert np.allclose(W.walk(x1, D, "ZEp")["F"], ze["F"])


def test_pit_closed_plus_one_mutation_is_caught(monkeypatch) -> None:
    """Пара дня D (t⁺ > D) в месяц переоценки D не входит в пул: подмена её rv не меняет g месяца; мутация «закрыто + 1» меняет."""
    x1, D = SZ.make(500, 8, p_e=0.4)
    st = months(x1)
    a = next(p for p in st[3:] if D["e"][p] == 1)
    b = st[st.index(a) + 1]
    x2, D2 = x1.copy(), dict(D, u=D["u"].copy())
    x2.loc[a, "rv"] *= 50.0
    D2["u"][a] *= 50.0
    assert bits(W.walk(x1, D, "ZE")["g"][a:b], W.walk(x2, D2, "ZE")["g"][a:b])
    monkeypatch.setattr(ZW, "closed_date", lambda tp, d: d)       # мутация: пара видна в день прогноза d
    assert not bits(W.walk(x1, D, "ZE")["g"][a:b], W.walk(x2, D2, "ZE")["g"][a:b])


def test_validation_block_is_last_60_extreme_pairs_and_mutation_all_pairs_is_caught(monkeypatch) -> None:
    x1, D = SZ.make(700, 9, p_e=0.4)
    seen = []
    orig = FT.select
    monkeypatch.setattr(FT, "select", lambda tr, val, fit, score, om=(1.0,): (seen.append((tr, val)), orig(tr, val, fit, score, om))[1])
    W.walk(x1, D, "ZE")
    assert seen and all((D["e"][v] == 1).all() and len(v) == min(FT.V_EXT, (len(tr) + len(v)) // 2) for tr, v in seen if len(v) > 0)
    seen.clear()
    monkeypatch.setattr(FT, "val_block_b", lambda pool, ext, v: FT.val_idx(pool, v))        # мутация: последние V пар пула — все дни
    W.walk(x1, D, "ZE")
    assert any((D["e"][v] == 0).any() for _, v in seen)
