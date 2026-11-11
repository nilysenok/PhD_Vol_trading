"""W9, сверки определения на реальных рядах и PIT-терцили: F_cfg8 = C; ln F8 = ln H + G; D = D_H + D_G; ell = ln(M/H); мутация «D_G = G» (ошибка Д-1) ловится
сверкой; PIT-терцили используют только прошлые даты, мутация «по всей выборке» ловится; боевые группы признака W9D2 не меняются от будущих значений."""
from __future__ import annotations

import numpy as np
import pytest

from models_v2.s16 import x1_w9_diag as D
from models_v2.s16 import x1_w9_diag_core as K


@pytest.fixture(scope="module")
def real():
    x, chk = K.load(with_z=False)
    return x, chk


def test_definition_checks_on_real_series(real) -> None:
    x, chk = real
    assert len(x) == 2198
    assert chk["C_eq_F8"] <= 1e-8 and chk["lnF8_eq_lnH_G"] <= 1e-8 and chk["ell_eq_ln_M_H"] <= 1e-8
    assert chk["D_eq_DH_DG"] <= 1e-8 and chk["lnX1_eq"] <= 1e-8 and chk["H_eq_K1"] <= 1e-10
    # то же независимо от кода сверки
    assert np.max(np.abs(x["F8"] / x["C"] - 1)) <= 1e-8
    assert np.max(np.abs(np.log(x["F8"]) - np.log(x["H"]) - x["G"])) <= 1e-8
    assert np.max(np.abs(x["D"] - x["DH"] - x["DG"])) <= 1e-8
    assert np.max(np.abs(x["DH"] + (1 - x["w"]) * x["ell"])) == 0


def test_mutation_dg_equals_g_is_caught(real) -> None:
    x, _ = real
    bad = x["D"] - x["DH"] - x["G"]                                            # мутация Д-1: D_G = G
    assert np.max(np.abs(bad)) > 1e-3                                          # тождество D = D_H + D_G рвётся на реальных рядах
    y = x.copy()
    y["G"] = y["G"] + y["w"] * y["ell"]                                        # «G уже не включает якорь»: сверка ln F8 = ln H + G падает
    with pytest.raises(SystemExit):
        K.definition_checks(y)


def test_mutation_swapped_sign_of_dh_is_caught(real) -> None:
    x, _ = real
    y = x.copy()
    y["ell"] = -y["ell"]                                                       # перепутанный знак ℓ
    with pytest.raises(SystemExit):
        K.definition_checks(y)


def test_pit_terciles_use_only_past_and_whole_sample_mutation_caught() -> None:
    rng = np.random.default_rng(1)
    v = rng.normal(size=300)
    g0 = K.pit_terciles(v)
    assert (g0[:K.MIN_PAST] == -1).all() and (g0[K.MIN_PAST:] >= 0).all()
    w = v.copy()
    w[200:] = rng.normal(size=100) * 50                                        # меняем только будущее относительно i = 199
    g1 = K.pit_terciles(w)
    assert np.array_equal(g0[:200], g1[:200])                                  # группы дней до 200 не зависят от значений после

    def whole_sample(a):                                                       # мутация: пороги по всей выборке
        q1, q2 = np.quantile(a, [1 / 3, 2 / 3])
        return np.where(a <= q1, 0, np.where(a <= q2, 1, 2))

    assert not np.array_equal(whole_sample(v)[:200], whole_sample(w)[:200])    # такая мутация зависит от будущего — тест её отличает
    assert not np.array_equal(whole_sample(v)[K.MIN_PAST:200], g0[K.MIN_PAST:200])


def test_real_feature_groups_ignore_future(real) -> None:
    x, _ = real
    v = D.feat_raw(x, "W9D2_encshare")
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
    g = D.groups_of("W9D2_encshare", v)
    cut = 1500
    v2 = v.copy()
    v2[cut:] = v2[cut:][::-1]                                                  # переставили будущее
    g2 = D.groups_of("W9D2_encshare", v2)
    assert np.array_equal(g[:cut], g2[:cut]) and not np.array_equal(g[cut:], g2[cut:])
    assert set(np.unique(g)) == {-1, 0, 1, 2}
    ga = D.groups_of("W9D1_agree", D.feat_raw(x, "W9D1_agree"))               # agree — без порогов, группы 0/1 по знаку произведения
    assert set(np.unique(ga)) <= {0, 1}
