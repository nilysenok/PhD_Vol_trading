"""Ж-214, этап V: диагностика сезонности и шума. Известный ответ на синтетике: белый шум -> сигнатура ~1, АК ~0; шум микроструктуры
(доходности MA(1), АК ≈ −0,5) -> RV10/RV20 заметно > 1; U-образный профиль -> отношение первого слота > 1. Мутация: перепутанные RV10/RV20."""
import numpy as np
import pandas as pd

from models_v2.s9 import hv2_diagv as DV, hv2_seas as Z

N = 400
IDX = pd.bdate_range("2016-01-04", periods=N)


def returns(noise_sd=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.1, (N, Z.NSLOT))
    if noise_sd:
        e = rng.normal(0, noise_sd, (N, Z.NSLOT + 1))
        r = r + e[:, 1:] - e[:, :-1]                      # наблюдаемая цена = эффективная + шум -> MA(1)
    return r


def test_signature_white_noise_vs_microstructure_noise():
    w = DV.signature(returns(), IDX)
    n = DV.signature(returns(noise_sd=0.1), IDX)
    assert (w["rv10_over_rv20"].between(0.97, 1.03)).all() and (w["ac1"].abs() < 0.02).all()
    assert (n["rv10_over_rv20"] > 1.3).all() and (n["ac1"] < -0.3).all()
    swapped = 1.0 / n["rv10_over_rv20"]                    # мутант (перепутано направление отношения): шум не виден
    assert not (swapped > 1.3).any()


def test_hac_slope_known_answer_and_sign_mutation():
    from models_v2.s9 import hv2_info as I
    rng = np.random.default_rng(4)
    z = rng.normal(size=1500)
    e = 0.3 * z + rng.normal(size=1500)
    b, t, p = I.nw_t(z, e)
    assert 0.2 < b < 0.4 and t > 6 and p < 1e-6
    b0, t0, p0 = I.nw_t(z, rng.normal(size=1500))
    assert abs(t0) < 3.5 and p0 > 1e-4
    assert I.nw_t(z, -e)[0] < 0                          # перепутанный знак ошибки меняет знак наклона


def test_profile_summary_known_answer_u_shape():
    rng = np.random.default_rng(1)
    shape = np.ones(Z.NSLOT)
    shape[0], shape[-1] = 5.0, 2.0
    M2 = (rng.normal(0, 1, (N, Z.NSLOT)) ** 2) * shape
    pt = DV.profile_table(M2, IDX)
    assert abs(pt[pt["year"] == "all"]["share"].sum() - 1.0) < 1e-12
    s = DV.profile_summary(pt)
    assert (s["first_over_median"].between(4.0, 6.5)).all() and (s["corr_with_all"] > 0.9).all()
    assert (s["last_over_median"].between(1.4, 2.7)).all()
